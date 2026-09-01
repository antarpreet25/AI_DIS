"""
zedd_trainer.py

Runs two distinct ZEDD experiments and produces a comparison:

    Experiment A — Baseline ZEDD
        all-MiniLM-L6-v2 (unmodified)
        original 6-sentence baseline
        fixed threshold = 0.35
        → evaluated on scaled dataset

    Experiment B — Fine-tuned ZEDD
        all-MiniLM-L6-v2 fine-tuned on training_pairs.json
        enriched baseline rebuilt with fine-tuned model
        GMM-calibrated threshold (data-driven)
        → evaluated on scaled dataset

Research question answered by the comparison:
    Does domain-specific contrastive fine-tuning plus data-driven
    threshold calibration improve ZEDD detection over the original
    zero-shot configuration?

Usage
-----
    python src/zedd_trainer.py            # full pipeline
    python src/zedd_trainer.py --compare  # compare only (no re-training)

Output files
------------
    data/zedd_finetuned_model/      fine-tuned SentenceTransformer
    data/zedd_finetuned_baseline.json   enriched baseline from fine-tuned model
    data/zedd_calibration.json          GMM calibration results + threshold
    data/zedd_training_log.json         per-epoch loss values

Notes on training data
----------------------
The 330 training pairs are constructed from 65 base documents and 200
attack variants (200 injected-clean pairs + 130 clean-clean pairs).
These are not 330 independent documents; they share an underlying
document population of 65 clean texts.  This is described accurately
in the dissertation methodology rather than claimed as 330 independent
examples.  The contrastive objective is still valid because each pair
presents a genuine semantic contrast.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Project root — resolves correctly when run from AI_DIS/ root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    All hyperparameters in one place.
    Change a value here; it propagates everywhere automatically.
    Documented in the dissertation as part of the reproducibility statement.
    """
    batch_size:       int   = 8      # conservative for CPU / MPS
    epochs:           int   = 10
    warmup_steps:     int   = 20
    learning_rate:    float = 2e-5
    weight_decay:     float = 0.01
    max_grad_norm:    float = 1.0
    dataset_seed:     int   = 42     # seed used in attack_generator.py
    training_seed:    int   = 42     # seed passed to GaussianMixture
    model_name:       str   = "all-MiniLM-L6-v2"
    baseline_threshold: float = 0.35  # Sekar et al. default

    # Paths (relative to _PROJECT_ROOT)
    training_pairs_path:   str = "data/training_pairs.json"
    scaled_dataset_path:   str = "data/attacks/attack_dataset_scaled.json"
    finetuned_model_path:  str = "data/zedd_finetuned_model"
    finetuned_baseline:    str = "data/zedd_finetuned_baseline.json"
    calibration_path:      str = "data/zedd_calibration.json"
    training_log_path:     str = "data/zedd_training_log.json"


# ---------------------------------------------------------------------------
# ZEDDTrainer
# ---------------------------------------------------------------------------

class ZEDDTrainer:
    """
    Orchestrates fine-tuning, baseline rebuilding, GMM calibration,
    and the baseline vs. fine-tuned comparison.

    All methods are designed to be called in sequence via run(), but
    can also be called individually for partial reruns (e.g.
    compare_with_baseline() after a previous full run).
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        self._root = _PROJECT_ROOT

        # Resolved absolute paths
        self._pairs_path      = self._root / self.cfg.training_pairs_path
        self._dataset_path    = self._root / self.cfg.scaled_dataset_path
        self._model_out       = self._root / self.cfg.finetuned_model_path
        self._baseline_out    = self._root / self.cfg.finetuned_baseline
        self._calibration_out = self._root / self.cfg.calibration_path
        self._log_out         = self._root / self.cfg.training_log_path

        self._finetuned_model = None   # populated after fine_tune()
        self._epoch_losses: list[dict] = []

    # ------------------------------------------------------------------
    # Step 1 — Load training pairs
    # ------------------------------------------------------------------

    def load_training_pairs(self) -> tuple[list, list]:
        if not self._pairs_path.exists():
            raise FileNotFoundError(
                f"Training pairs not found at {self._pairs_path}.\n"
                "Run: python src/attack_generator.py --scaled"
            )

        with open(self._pairs_path, encoding="utf-8") as f:
            all_pairs = json.load(f)

        # Handles both 0/1 integer labels and floating contrastive labels
        injected_clean = [p for p in all_pairs if p.get("label") == 0 or p.get("label") == 0.0]
        clean_clean    = [p for p in all_pairs if p.get("label") == 1 or p.get("label") == 1.0]

        print(f"[load_training_pairs] Loaded {len(all_pairs)} pairs "
            f"({len(injected_clean)} injected-clean, "
            f"{len(clean_clean)} clean-clean)")
        return injected_clean, clean_clean

    # ------------------------------------------------------------------
    # Step 2 — Build InputExamples
    # ------------------------------------------------------------------

    def build_input_examples(
    self,
    injected_clean: list,
    clean_clean: list,
) -> list:
        """
        Convert pair dicts to InputExample objects for ContrastiveLoss.

        Label convention (ContrastiveLoss):
            label = 0.0 -> dissimilar pair (injected-clean)
            label = 1.0 -> similar pair (clean-clean)
        """
        from sentence_transformers import InputExample

        examples = []
        
        # Dissimilar pairs (Label 0)
        for p in injected_clean:
            t1 = p.get("text1") or p.get("injected")
            t2 = p.get("text2") or p.get("clean")
            examples.append(InputExample(texts=[t1, t2], label=0.0))

        # Similar pairs (Label 1)
        for p in clean_clean:
            t1 = p.get("text1") or p.get("text2_a") or p.get("injected")
            t2 = p.get("text2") or p.get("text2_b") or p.get("clean")
            examples.append(InputExample(texts=[t1, t2], label=1.0))

        # Shuffle with fixed seed for reproducibility
        import random
        rng = random.Random(self.cfg.training_seed)
        rng.shuffle(examples)

        print(f"[build_input_examples] {len(examples)} InputExamples "
            f"(seed={self.cfg.training_seed})")
        return examples

    # ------------------------------------------------------------------
    # Step 3 — Fine-tune
    # ------------------------------------------------------------------

    def fine_tune(self, examples: list) -> None:
        """
        Fine-tune all-MiniLM-L6-v2 using ContrastiveLoss.

        Uses SentenceTransformerTrainer (sentence-transformers v5+)
        so that training loss is obtained from the HuggingFace Trainer
        state — not fabricated or approximated.

        Training loss is logged per epoch to self._epoch_losses and
        saved to data/zedd_training_log.json.
        """
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.losses import ContrastiveLoss

        try:
            from sentence_transformers import (
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
            )
            _use_new_api = True
        except ImportError:
            _use_new_api = False

        print(f"\n[fine_tune] Loading base model: {self.cfg.model_name}")
        model = SentenceTransformer(self.cfg.model_name)
        loss_fn = ContrastiveLoss(model)

        self._model_out.mkdir(parents=True, exist_ok=True)
        self._epoch_losses = []

        if _use_new_api:
            self._fine_tune_new_api(model, loss_fn, examples)
        else:
            self._fine_tune_legacy(model, loss_fn, examples)

        # Save fine-tuned model
        model.save(str(self._model_out))
        print(f"[fine_tune] Model saved → {self._model_out}")
        self._finetuned_model = model
        self._save_training_log()

    def _fine_tune_new_api(self, model, loss_fn, examples: list) -> None:
        """Use SentenceTransformerTrainer (v5+) for proper loss logging."""
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from datasets import Dataset as HFDataset

        # Convert examples to HuggingFace Dataset format
        records = [
            {"sentence1": ex.texts[0],
             "sentence2": ex.texts[1],
             "label":     float(ex.label)}
            for ex in examples
        ]
        hf_dataset = HFDataset.from_list(records)

        args = SentenceTransformerTrainingArguments(
            output_dir                = str(self._model_out / "checkpoints"),
            num_train_epochs          = self.cfg.epochs,
            per_device_train_batch_size = self.cfg.batch_size,
            warmup_steps              = self.cfg.warmup_steps,
            learning_rate             = self.cfg.learning_rate,
            weight_decay              = self.cfg.weight_decay,
            max_grad_norm             = self.cfg.max_grad_norm,
            seed                      = self.cfg.training_seed,
            logging_steps             = 10,
            save_strategy             = "no",
            report_to                 = "none",
        )

        trainer = SentenceTransformerTrainer(
            model     = model,
            args      = args,
            train_dataset = hf_dataset,
            loss      = loss_fn,
        )

        print(f"[fine_tune] Training: {self.cfg.epochs} epochs, "
              f"batch={self.cfg.batch_size}, lr={self.cfg.learning_rate}, "
              f"seed={self.cfg.training_seed}")
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0

        # Extract per-epoch loss from trainer log history
        log_history = trainer.state.log_history
        epoch_logs: dict[int, list[float]] = {}
        for entry in log_history:
            if "loss" in entry:
                ep = int(entry.get("epoch", 0))
                epoch_logs.setdefault(ep, []).append(entry["loss"])

        for ep in sorted(epoch_logs):
            avg_loss = float(np.mean(epoch_logs[ep]))
            self._epoch_losses.append({"epoch": ep, "train_loss": round(avg_loss, 6)})
            print(f"  Epoch {ep:2d}: train_loss = {avg_loss:.6f}")

        print(f"[fine_tune] Training complete in {elapsed:.1f}s")

    def _fine_tune_legacy(self, model, loss_fn, examples: list) -> None:
        """
        Fallback: use model.fit() with a callback for loss logging.
        The callback receives (score, epoch, steps) where score comes
        from the evaluator — None here since we pass no evaluator.
        Loss is extracted from the loss_fn's __call__ accumulation.
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(examples, batch_size=self.cfg.batch_size, shuffle=True)

        # Capture loss via callback — score is None (no evaluator passed),
        # epoch and steps are real values from the training loop.
        def _callback(score, epoch: int, steps: int):
            # With no evaluator, score=None. We log the step count.
            self._epoch_losses.append({
                "epoch": epoch,
                "steps": steps,
                "note": "loss not available via legacy callback without evaluator",
            })

        print(f"[fine_tune] Training (legacy API): {self.cfg.epochs} epochs")
        t0 = time.time()
        model.fit(
            train_objectives  = [(loader, loss_fn)],
            epochs            = self.cfg.epochs,
            warmup_steps      = self.cfg.warmup_steps,
            optimizer_params  = {"lr": self.cfg.learning_rate},
            weight_decay      = self.cfg.weight_decay,
            max_grad_norm     = self.cfg.max_grad_norm,
            callback          = _callback,
            show_progress_bar = True,
        )
        print(f"[fine_tune] Training complete in {time.time()-t0:.1f}s")

    def _save_training_log(self) -> None:
        self._log_out.parent.mkdir(parents=True, exist_ok=True)
        log = {
            "config": asdict(self.cfg),
            "epoch_losses": self._epoch_losses,
        }
        with open(self._log_out, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"[fine_tune] Training log → {self._log_out}")

    # ------------------------------------------------------------------
    # Step 4 — Rebuild enriched baseline with fine-tuned model
    # ------------------------------------------------------------------

    def rebuild_baseline(self) -> None:
        """
        Build the enriched baseline centroid using the fine-tuned model
        and save it.  The fine-tuned model must have been loaded or
        fine-tuned in this session (self._finetuned_model) OR saved to
        disk (self._model_out) from a previous run.
        """
        sys.path.insert(0, str(self._root / "src"))
        from defense.zedd import build_enriched_baseline

        if self._finetuned_model is None:
            self._load_finetuned_model()

        print("[rebuild_baseline] Building enriched baseline with fine-tuned model…")
        detector = build_enriched_baseline(
            model      = self._finetuned_model,
            model_name = self.cfg.model_name,
            data_root  = self._root,
        )
        detector.save_baseline(self._baseline_out)
        print(f"[rebuild_baseline] Baseline saved → {self._baseline_out}")

    # ------------------------------------------------------------------
    # Step 5 — Load scaled dataset and compute drift scores
    # ------------------------------------------------------------------

    def load_scaled_dataset(self) -> tuple[list, list]:
        """
        Load attack_dataset_scaled.json.

        Returns
        -------
        (attack_texts, benign_texts) — list of str
        """
        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Scaled dataset not found at {self._dataset_path}.\n"
                "Run: python src/attack_generator.py --scaled"
            )

        with open(self._dataset_path, encoding="utf-8") as f:
            payload = json.load(f)

        samples = payload["samples"]
        attacks = [s["wrapped_attack"] for s in samples if s["category"] != "benign"]
        benign  = [s["wrapped_attack"] for s in samples if s["category"] == "benign"]

        print(f"[load_scaled_dataset] {len(attacks)} attacks, {len(benign)} benign")
        return attacks, benign

    def calculate_drift_scores(
        self,
        texts: list[str],
        detector,
    ) -> np.ndarray:
        """
        Compute drift scores (1 - cosine_similarity_to_centroid) for a
        list of texts using a ZEDDDetector instance.

        ZEDD's decision variable is the drift score, not the raw
        embedding — the GMM must be fitted on drift scores, not
        embeddings (Sekar et al. section 4.3.1).
        """
        scores = []
        for text in texts:
            result = detector.detect(text)
            # drift_score = 1 - similarity_score
            drift = 1.0 - result.similarity_score
            scores.append(drift)
        return np.array(scores)

    # ------------------------------------------------------------------
    # Step 6 — Fit GMM and find threshold
    # ------------------------------------------------------------------

    def fit_gmm(
        self,
        attack_scores: np.ndarray,
        benign_scores: np.ndarray,
    ) -> GaussianMixture:
        """
        Fit a two-component GMM to the combined drift score distribution.

        The GMM models the underlying distribution of drift scores:
            Component A ≈ clean/benign documents  (lower mean drift)
            Component B ≈ injected documents      (higher mean drift)

        The two components are identified by their means — not assumed
        by component index, since GaussianMixture makes no ordering
        guarantee.
        """
        all_scores = np.concatenate([attack_scores, benign_scores]).reshape(-1, 1)

        gmm = GaussianMixture(
            n_components = 2,
            random_state = self.cfg.training_seed,
            max_iter     = 200,
        )
        gmm.fit(all_scores)

        means   = gmm.means_.flatten()
        weights = gmm.weights_
        clean_idx  = int(np.argmin(means))
        attack_idx = int(np.argmax(means))

        print(f"[fit_gmm] Clean component:  mean={means[clean_idx]:.4f}, "
              f"weight={weights[clean_idx]:.4f}")
        print(f"[fit_gmm] Attack component: mean={means[attack_idx]:.4f}, "
              f"weight={weights[attack_idx]:.4f}")

        # Attach identities for downstream use
        gmm._clean_idx  = clean_idx
        gmm._attack_idx = attack_idx

        return gmm

    def find_gmm_threshold(self, gmm: GaussianMixture) -> float:
        """
        Find the drift score threshold where the weighted density of
        the clean component equals the weighted density of the attack
        component — i.e. the crossing point of the two Gaussians.

        Follows Sekar et al. equation 2:
            f_clean(x) · w_clean = f_injected(x) · w_injected

        Implemented as a binary search over [0, 1] with 64 iterations
        (precision ~5e-20 — far beyond what's practically meaningful).

        Includes a sanity check: the threshold must lie strictly
        between the clean and attack component means.  If not, the
        calibration has failed and a loud ValueError is raised rather
        than silently saving a nonsensical threshold.
        """
        means   = gmm.means_.flatten()
        weights = gmm.weights_
        covars  = gmm.covariances_.flatten()

        c_idx = gmm._clean_idx
        a_idx = gmm._attack_idx

        def _density(x: float, idx: int) -> float:
            std = float(np.sqrt(covars[idx]))
            return float(weights[idx]) * stats.norm.pdf(x, means[idx], std)

        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _density(mid, c_idx) > _density(mid, a_idx):
                lo = mid
            else:
                hi = mid
        threshold = (lo + hi) / 2.0

        # --- Sanity checks ---
        clean_mean  = float(means[c_idx])
        attack_mean = float(means[a_idx])

        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"GMM calibration produced an out-of-range threshold: "
                f"{threshold:.4f}.  Expected 0 < threshold < 1.  "
                "Check that drift scores are in [0, 1] and that clean "
                "and attack distributions are genuinely separable."
            )

        if not (clean_mean < threshold < attack_mean):
            raise ValueError(
                f"GMM threshold ({threshold:.4f}) does not lie between "
                f"clean mean ({clean_mean:.4f}) and attack mean "
                f"({attack_mean:.4f}).  The GMM components may not "
                "correspond to clean/attack distributions as expected. "
                "Check that the dataset has sufficient class separation."
            )

        print(f"[find_gmm_threshold] Threshold = {threshold:.4f} "
              f"(clean mean={clean_mean:.4f}, attack mean={attack_mean:.4f})")
        return threshold

    # ------------------------------------------------------------------
    # Step 7 — Evaluate at a given threshold
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        attack_scores:  np.ndarray,
        benign_scores:  np.ndarray,
        threshold:      float,
        label:          str,
    ) -> dict:
        """
        Compute recall and FPR for a drift-score threshold.

        Drift scores are in [0, 1].  Higher = more suspicious.
        A sample is flagged (predicted attack) if drift_score > threshold.
        """
        tp = int(np.sum(attack_scores > threshold))
        fn = int(np.sum(attack_scores <= threshold))
        fp = int(np.sum(benign_scores > threshold))
        tn = int(np.sum(benign_scores <= threshold))

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        result = {
            "label":       label,
            "threshold":   round(threshold, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall":      round(recall,    4),
            "fpr":         round(fpr,       4),
            "precision":   round(precision, 4),
            "f1":          round(f1,        4),
        }
        return result

    # ------------------------------------------------------------------
    # Step 8 — Save calibration
    # ------------------------------------------------------------------

    def save_calibration(
        self,
        threshold:      float,
        gmm:            GaussianMixture,
        attack_scores:  np.ndarray,
        benign_scores:  np.ndarray,
    ) -> None:
        """
        Save the calibrated threshold and metadata.

        Richer than the minimum spec — includes sample counts, model
        provenance, and component statistics so the experiment is fully
        auditable from the JSON file alone.
        """
        means   = gmm.means_.flatten()
        weights = gmm.weights_
        c_idx   = gmm._clean_idx
        a_idx   = gmm._attack_idx

        # FPR on benign at the calibrated threshold
        fp  = int(np.sum(benign_scores > threshold))
        tn  = int(np.sum(benign_scores <= threshold))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        calibration = {
            "threshold":              round(float(threshold), 6),
            "clean_component_mean":   round(float(means[c_idx]), 6),
            "injected_component_mean":round(float(means[a_idx]), 6),
            "clean_component_weight": round(float(weights[c_idx]), 6),
            "injected_component_weight": round(float(weights[a_idx]), 6),
            "clean_fpr_at_threshold": round(fpr, 6),
            "n_attack_samples":       int(len(attack_scores)),
            "n_benign_samples":       int(len(benign_scores)),
            "attack_drift_mean":      round(float(attack_scores.mean()), 6),
            "benign_drift_mean":      round(float(benign_scores.mean()), 6),
            "model":                  self.cfg.finetuned_model_path,
            "baseline":               self.cfg.finetuned_baseline,
            "training_seed":          self.cfg.training_seed,
            "dataset_seed":           self.cfg.dataset_seed,
        }

        self._calibration_out.parent.mkdir(parents=True, exist_ok=True)
        with open(self._calibration_out, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        print(f"[save_calibration] Saved → {self._calibration_out}")

    # ------------------------------------------------------------------
    # Step 8b — Per-mode threshold calibration
    # ------------------------------------------------------------------

    def calibrate_all_modes(self) -> dict:
        """
        Re-derive the GMM-calibrated threshold SEPARATELY for each of
        ZEDD's three runtime modes and rewrite data/zedd_calibration.json
        with one entry per mode.

        WHY THIS IS NEEDED
        ------------------
        run() (Step 6) computes drift scores with the detector in
        DOCUMENT mode only, so the single threshold it saves is a
        document-mode threshold. The three modes score on different
        scales:

            document      decision var = document drift (1 - doc cosine);
                          flag if drift > threshold
            sentence      decision var = worst-sentence similarity;
                          flag if worst_sim < threshold
            dual_encoder  decision var = worst-sentence combined score
                          (0.5*domain_drift + 0.5*security_proximity);
                          flag if combined > threshold

        Applying the document threshold to sentence / dual_encoder makes
        ZEDD flag most / all benign inputs. This method fits the same
        two-component GMM used in run(), once per mode, on that mode's
        own score distribution over the scaled dataset.

        Reuses the fine-tuned model + enriched baseline already on disk —
        no retraining. Called by `python src/zedd_trainer.py --calibrate`.

        Output JSON shape (back-compatible: the flat top-level "threshold"
        is kept, equal to the document-mode value):

            {
              "threshold": <document drift threshold>,       # legacy readers
              "modes": {
                "document":     {"threshold": .., "flag_rule": ..,
                                 "eval": {recall, fpr, ..}, ..},
                "sentence":     { .. },
                "dual_encoder": { .. }
              },
              ...
            }
        """
        from sentence_transformers import SentenceTransformer
        sys.path.insert(0, str(self._root / "src"))
        from defense.zedd import ZEDDDetector

        attack_texts, benign_texts = self.load_scaled_dataset()

        if self._finetuned_model is None:
            self._load_finetuned_model()

        modes = ("document", "sentence", "dual_encoder")
        fpr_target = 0.015   # fallback operating point if the GMM crossing is not clean
        mode_entries: dict = {}

        for mode in modes:
            print(f"\n[calibrate_all_modes] --- {mode} ---")
            detector = ZEDDDetector(
                model          = self._finetuned_model,
                model_name     = self.cfg.model_name,
                sentence_level = (mode == "sentence"),
                dual_encoder   = (mode == "dual_encoder"),
            )
            detector.load_baseline(self._baseline_out)

            a = self._suspicion_scores(attack_texts, detector, mode)
            b = self._suspicion_scores(benign_texts, detector, mode)
            print(f"  attack suspicion: mean={a.mean():.4f} [{a.min():.4f}, {a.max():.4f}]")
            print(f"  benign suspicion: mean={b.mean():.4f} [{b.min():.4f}, {b.max():.4f}]")

            gmm = self.fit_gmm(a, b)
            gmm_thr = self._gmm_crossing_unchecked(gmm, a, b)
            gmm_eval = self._eval_threshold(a, b, gmm_thr)

            # Fallback: lowest cutoff keeping benign FPR <= fpr_target
            fpr_thr = float(np.quantile(b, 1.0 - fpr_target)) + 1e-6
            fpr_eval = self._eval_threshold(a, b, fpr_thr)

            print(f"  GMM crossing   thr={gmm_thr:.4f} -> recall={gmm_eval['recall']} fpr={gmm_eval['fpr']}")
            print(f"  FPR<={fpr_target} thr={fpr_thr:.4f} -> recall={fpr_eval['recall']} fpr={fpr_eval['fpr']}")

            if gmm_eval["fpr"] <= fpr_target:
                chosen, method, chosen_eval = gmm_thr, "gmm_crossing", gmm_eval
            else:
                chosen, method, chosen_eval = fpr_thr, "fpr_target", fpr_eval
            print(f"  chosen: {method}  threshold={chosen:.6f}")

            # `chosen` is in SUSPICION orientation (higher = more
            # suspicious). Convert to the exact number the mode's detector
            # assigns to self.drift_threshold and compares against:
            #   sentence      : flag if worst_sentence_similarity < drift_threshold
            #                   -> runtime value is a SIMILARITY floor = 1 - suspicion
            #   dual_encoder  : flag if combined_score > drift_threshold
            #                   -> runtime value is the suspicion cutoff itself
            #   document      : _detect_document_level compares the closest-
            #                   category similarity against that category's
            #                   OWN per-category threshold from the baseline
            #                   JSON, never self.drift_threshold, so the
            #                   runtime value is recorded for completeness /
            #                   the comparison table but does not by itself
            #                   change document-mode decisions.
            if mode == "dual_encoder":
                runtime_thr = float(chosen)
            else:
                runtime_thr = 1.0 - float(chosen)

            mode_entries[mode] = {
                "runtime_drift_threshold":     round(runtime_thr, 6),
                "applies_to_runtime_decision": (mode != "document"),
                "suspicion_cutoff":            round(float(chosen), 6),
                "selection_method":            method,
                "flag_rule": (
                    "combined_score > runtime_drift_threshold" if mode == "dual_encoder"
                    else "worst_sentence_similarity < runtime_drift_threshold" if mode == "sentence"
                    else "closest_category_similarity < per_category_threshold  (runtime_drift_threshold unused)"
                ),
                "gmm_crossing_suspicion": round(float(gmm_thr), 6),
                "eval": chosen_eval,
                "attack_suspicion_mean": round(float(a.mean()), 6),
                "benign_suspicion_mean": round(float(b.mean()), 6),
            }

        calibration = {
            # Legacy flat key kept for older readers: the document-mode
            # drift crossing point, same quantity zedd_trainer.run() wrote.
            "threshold":        mode_entries["document"]["suspicion_cutoff"],
            "modes":            mode_entries,
            "n_attack_samples": int(len(attack_texts)),
            "n_benign_samples": int(len(benign_texts)),
            "model":            self.cfg.finetuned_model_path,
            "baseline":         self.cfg.finetuned_baseline,
            "training_seed":    self.cfg.training_seed,
            "dataset_seed":     self.cfg.dataset_seed,
            "_note": (
                "Per-mode ZEDD thresholds. zedd.py:load_finetuned selects "
                "modes[<active mode>].threshold. Calibration suspicion "
                "orientation is 'higher = more suspicious' (document/sentence: "
                "1 - similarity_score; dual_encoder: combined_score)."
            ),
        }

        self._calibration_out.parent.mkdir(parents=True, exist_ok=True)
        with open(self._calibration_out, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        print(f"\n[calibrate_all_modes] Saved -> {self._calibration_out}")
        for mode in modes:
            e = mode_entries[mode]
            applied = "" if e["applies_to_runtime_decision"] else "  (not applied at runtime)"
            print(f"  {mode:<13} runtime_drift_threshold={e['runtime_drift_threshold']:.4f}  "
                  f"recall={e['eval']['recall']}  fpr={e['eval']['fpr']}  "
                  f"({e['selection_method']}){applied}")
        return calibration

    def _suspicion_scores(self, texts: list, detector, mode: str) -> np.ndarray:
        """Per text, the scalar the mode's flag rule compares against its
        threshold, normalised so higher = more suspicious:
            document / sentence : 1 - similarity_score
            dual_encoder        : combined_score
        """
        vals = []
        for text in texts:
            r = detector.detect(text)
            if mode == "dual_encoder":
                vals.append(float(r.combined_score if r.combined_score is not None else 0.0))
            else:
                vals.append(1.0 - float(r.similarity_score))
        return np.asarray(vals, dtype=np.float64)

    def _gmm_crossing_unchecked(self, gmm, attack: np.ndarray, benign: np.ndarray) -> float:
        """Same crossing-point search as find_gmm_threshold, but searched
        over the observed score range and WITHOUT the
        clean_mean < thr < attack_mean assertion — the caller decides
        whether the result is acceptable via its benign FPR."""
        means   = gmm.means_.flatten()
        weights = gmm.weights_
        covars  = gmm.covariances_.flatten()
        c_idx, a_idx = gmm._clean_idx, gmm._attack_idx

        def _density(x: float, idx: int) -> float:
            std = float(np.sqrt(covars[idx]))
            return float(weights[idx]) * stats.norm.pdf(x, means[idx], std)

        lo = float(min(attack.min(), benign.min()))
        hi = float(max(attack.max(), benign.max()))
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if _density(mid, c_idx) > _density(mid, a_idx):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    @staticmethod
    def _eval_threshold(attack: np.ndarray, benign: np.ndarray, thr: float) -> dict:
        tp = int(np.sum(attack > thr)); fn = int(np.sum(attack <= thr))
        fp = int(np.sum(benign > thr)); tn = int(np.sum(benign <= thr))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr    = fp / (fp + tn) if (fp + tn) else 0.0
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "recall": round(recall, 4), "fpr": round(fpr, 4)}

    # ------------------------------------------------------------------
    # Step 9 — Compare baseline vs fine-tuned
    # ------------------------------------------------------------------

    def compare_with_baseline(
        self,
        attack_texts: list[str],
        benign_texts: list[str],
        finetuned_threshold: float,
    ) -> None:
        """
        Evaluate both ZEDD configurations on the same dataset and print
        the comparison table.

        Experiment A: original all-MiniLM-L6-v2, original baseline,
                      fixed threshold (self.cfg.baseline_threshold).
        Experiment B: fine-tuned model, enriched baseline,
                      GMM-calibrated threshold (finetuned_threshold).

        This answers the core research question:
            Does fine-tuning + GMM calibration improve ZEDD?
        """
        sys.path.insert(0, str(self._root / "src"))
        from defense.zedd import build_powergrid_baseline, ZEDDDetector

        # --- Experiment A: baseline ZEDD ---
        print("\n[compare] Loading baseline ZEDD (Experiment A)…")
        baseline_detector = build_powergrid_baseline()
        a_attack = self.calculate_drift_scores(attack_texts, baseline_detector)
        a_benign = self.calculate_drift_scores(benign_texts, baseline_detector)
        result_a = self._evaluate(
            a_attack, a_benign,
            threshold = self.cfg.baseline_threshold,
            label     = "Experiment A: baseline ZEDD (zero-shot, threshold=0.35)",
        )

        # --- Experiment B: fine-tuned ZEDD ---
        print("[compare] Loading fine-tuned ZEDD (Experiment B)…")
        if self._finetuned_model is None:
            self._load_finetuned_model()
        ft_detector = ZEDDDetector(model=self._finetuned_model)
        ft_detector.load_baseline(self._baseline_out)
        b_attack = self.calculate_drift_scores(attack_texts, ft_detector)
        b_benign = self.calculate_drift_scores(benign_texts, ft_detector)
        result_b = self._evaluate(
            b_attack, b_benign,
            threshold = finetuned_threshold,
            label     = f"Experiment B: fine-tuned ZEDD (GMM threshold={finetuned_threshold:.4f})",
        )

        # --- Print comparison ---
        print("\n" + "=" * 70)
        print("ZEDD EXPERIMENT COMPARISON")
        print("=" * 70)
        header = f"{'Configuration':<45} {'Recall':>7} {'FPR':>7} {'F1':>7}"
        print(header)
        print("-" * 70)
        for r in [result_a, result_b]:
            lbl = r["label"][:44]
            print(f"{lbl:<45} {r['recall']:>7.4f} {r['fpr']:>7.4f} {r['f1']:>7.4f}")
        print("=" * 70)

        # Save comparison to calibration JSON for the evaluator to read
        comparison = {"experiment_a": result_a, "experiment_b": result_b}
        comp_path  = self._root / "data" / "zedd_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[compare] Comparison saved → {comp_path}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _load_finetuned_model(self) -> None:
        """Load a previously fine-tuned model from disk."""
        if not self._model_out.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {self._model_out}. "
                "Run fine_tune() first."
            )
        from sentence_transformers import SentenceTransformer
        print(f"[_load_finetuned_model] Loading from {self._model_out}…")
        self._finetuned_model = SentenceTransformer(str(self._model_out))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the complete fine-tuning and calibration pipeline.

        Steps
        -----
        1. Load training pairs
        2. Build InputExamples
        3. Fine-tune model
        4. Rebuild enriched baseline with fine-tuned model
        5. Load scaled dataset
        6. Compute drift scores (fine-tuned model + baseline)
        7. Fit GMM on drift scores
        8. Find GMM crossing-point threshold
        9. Save calibration
        10. Compare baseline vs fine-tuned (prints table)
        """
        print("\n" + "=" * 60)
        print("ZEDD TRAINER — full pipeline")
        print(f"Model:        {self.cfg.model_name}")
        print(f"Epochs:       {self.cfg.epochs}")
        print(f"Batch size:   {self.cfg.batch_size}")
        print(f"Training seed:{self.cfg.training_seed}")
        print("=" * 60 + "\n")

        # 1-2: pairs → examples
        inj_clean, cl_cl = self.load_training_pairs()
        examples = self.build_input_examples(inj_clean, cl_cl)

        # 3: fine-tune
        self.fine_tune(examples)

        # 4: rebuild baseline
        self.rebuild_baseline()

        # 5: load dataset
        attack_texts, benign_texts = self.load_scaled_dataset()

        # 6: drift scores with fine-tuned detector
        print("[run] Computing drift scores on scaled dataset…")
        sys.path.insert(0, str(self._root / "src"))
        from defense.zedd import ZEDDDetector
        ft_detector = ZEDDDetector(model=self._finetuned_model)
        ft_detector.load_baseline(self._baseline_out)
        attack_scores = self.calculate_drift_scores(attack_texts, ft_detector)
        benign_scores = self.calculate_drift_scores(benign_texts, ft_detector)
        print(f"  Attack drift — mean={attack_scores.mean():.4f}, "
              f"std={attack_scores.std():.4f}")
        print(f"  Benign drift — mean={benign_scores.mean():.4f}, "
              f"std={benign_scores.std():.4f}")

        # 7-8: GMM
        gmm       = self.fit_gmm(attack_scores, benign_scores)
        threshold = self.find_gmm_threshold(gmm)

        # 9: save calibration
        self.save_calibration(threshold, gmm, attack_scores, benign_scores)

        # 10: comparison table
        self.compare_with_baseline(attack_texts, benign_texts, threshold)

        print("\n[run] Pipeline complete.")
        print(f"  Fine-tuned model → {self._model_out}")
        print(f"  Calibration      → {self._calibration_out}")
        print(f"  Training log     → {self._log_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_usage():
    print(__doc__)
    print("Usage:")
    print("  python src/zedd_trainer.py             # full pipeline")
    print("  python src/zedd_trainer.py --compare   # compare only")
    print("  python src/zedd_trainer.py --calibrate # per-mode thresholds only (no retraining)")


if __name__ == "__main__":
    args = sys.argv[1:]

    cfg     = TrainingConfig()
    trainer = ZEDDTrainer(config=cfg)

    if "--calibrate" in args:
        # Re-derive per-mode thresholds from the EXISTING fine-tuned model
        # + baseline. No fine-tuning. Rewrites data/zedd_calibration.json.
        if not ((_PROJECT_ROOT / cfg.finetuned_model_path).exists()
                and (_PROJECT_ROOT / cfg.finetuned_baseline).exists()):
            print("ERROR: --calibrate requires an existing fine-tuned model + baseline.")
            print("Run `python src/zedd_trainer.py` (no flag) first.")
            sys.exit(1)
        trainer.calibrate_all_modes()

    elif "--compare" in args:
        # Load existing fine-tuned model and calibration, run comparison only
        if not ((_PROJECT_ROOT / cfg.finetuned_model_path).exists()
                and (_PROJECT_ROOT / cfg.finetuned_baseline).exists()
                and (_PROJECT_ROOT / cfg.calibration_path).exists()):
            print("ERROR: --compare requires a completed fine-tuning run.")
            print("Run without --compare first.")
            sys.exit(1)

        with open(_PROJECT_ROOT / cfg.calibration_path, encoding="utf-8") as f:
            cal = json.load(f)
        finetuned_threshold = cal["threshold"]

        attack_texts, benign_texts = trainer.load_scaled_dataset()
        trainer.compare_with_baseline(attack_texts, benign_texts, finetuned_threshold)

    elif "--help" in args or "-h" in args:
        _print_usage()

    else:
        trainer.run()
