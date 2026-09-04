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
import random
import sys
import time
from contextlib import contextmanager
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

    def calibrate_all_modes(
        self,
        calib_attack_texts: Optional[list] = None,
        calib_benign_texts: Optional[list] = None,
        calibration_source: str = "evaluation_set",
    ) -> dict:
        """
        Re-derive the GMM-calibrated threshold SEPARATELY for each of
        ZEDD's three runtime modes and rewrite the calibration JSON with
        one entry per mode.

        calib_attack_texts / calib_benign_texts: if given, thresholds are
        fitted on THESE (a dataset disjoint from the evaluation set — the
        data-leakage fix). If omitted, falls back to the scaled evaluation
        set, which is the legacy behaviour and IS leaky (train == calibrate
        == test). The v3 flow (prepare_v3 / finish_v3) always passes a
        disjoint calibration set generated by zedd_corpus.

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

        if calib_attack_texts is not None and calib_benign_texts is not None:
            attack_texts, benign_texts = calib_attack_texts, calib_benign_texts
            print(f"[calibrate_all_modes] calibrating on '{calibration_source}' "
                  f"({len(attack_texts)} attacks, {len(benign_texts)} benign) — "
                  f"disjoint from the evaluation set.")
        else:
            attack_texts, benign_texts = self.load_scaled_dataset()
            calibration_source = "evaluation_set"
            print("[calibrate_all_modes] WARNING: calibrating on the EVALUATION "
                  "set (leaky). Pass a disjoint calibration set to fix this.")

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
                # Document mode's Stage-1 pre-filter would independently
                # flag inputs during suspicion scoring, so the GMM would
                # be fitted on a decision variable the calibrated floor
                # never actually governs. Disable it here so the fit sees
                # the real closest-centroid drift. (sentence/dual paths
                # don't use the pre-filter.)
                pre_filter_enabled = (mode != "document"),
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

            # The GMM crossing is only trustworthy when it lands strictly
            # between the two component means AND achieves a low benign
            # FPR. A degenerate fit (crossing outside [clean_mean,
            # attack_mean]) produced the v3 document floor of -0.216,
            # which sat below both clusters.
            c_mean = float(min(gmm.means_.flatten()))
            a_mean = float(max(gmm.means_.flatten()))
            gmm_between = c_mean < gmm_thr < a_mean

            print(f"  GMM crossing   thr={gmm_thr:.4f} "
                  f"(between means: {gmm_between}) -> recall={gmm_eval['recall']} fpr={gmm_eval['fpr']}")
            print(f"  FPR<={fpr_target} thr={fpr_thr:.4f} -> recall={fpr_eval['recall']} fpr={fpr_eval['fpr']}")

            if gmm_between and gmm_eval["fpr"] <= fpr_target:
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
            "calibration_source": calibration_source,
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

    @contextmanager
    def _temp_paths(self, **overrides):
        """Temporarily point instance path attributes (e.g. _model_out,
        _baseline_out, _calibration_out) somewhere else for the duration
        of a `with` block, then restore them — lets run_v2() reuse
        fine_tune()/calibrate_all_modes() unmodified while writing to v2
        output paths instead of the production ones."""
        saved = {k: getattr(self, k) for k in overrides}
        for k, v in overrides.items():
            setattr(self, k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # v2 pipeline — sentence-level contrastive pairs + document-register
    # baseline (see zedd_corpus.py's module docstring for the rationale)
    # ------------------------------------------------------------------

    def run_v2(self) -> dict:
        """
        Trains a SECOND fine-tuned model on document-level pairs (the
        existing data/training_pairs.json) PLUS sentence-level contrastive
        pairs (zedd_corpus.generate_sentence_pairs — isolated injected
        lines vs clean lines from newly generated, seed-disjoint
        documents), and builds its baseline from zedd_corpus's templated
        document-register corpus instead of sensor_generator.py telemetry.

        Writes to SEPARATE v2 paths — data/zedd_finetuned_model_v2/,
        data/zedd_finetuned_baseline_v2.json, data/zedd_calibration_v2.json
        — and does NOT touch the production v1 artifacts. main.py keeps
        using v1 until a human promotes v2 (see promote_v2() below), so a
        v2 run that turns out worse never regresses the running pipeline.

        Does not require v1's fine-tuned model — v2 trains from the base
        all-MiniLM-L6-v2 model on the combined pair set, independent of
        v1's weights, so the two runs cannot compound each other's error.
        """
        sys.path.insert(0, str(self._root / "src"))
        import zedd_corpus
        from defense.zedd import build_document_register_baseline

        v2_model = self._root / "data" / "zedd_finetuned_model_v2"
        v2_baseline = self._root / "data" / "zedd_finetuned_baseline_v2.json"
        v2_calibration = self._root / "data" / "zedd_calibration_v2.json"
        v2_pairs_path = self._root / "data" / "training_pairs_augmented.json"

        print("\n[run_v2] Step 1/5 — building augmented training pairs "
              "(document-level + sentence-level)…")
        inj_clean, cl_cl = self.load_training_pairs()
        base_pairs = inj_clean + cl_cl
        sentence_pairs = zedd_corpus.generate_sentence_pairs()
        combined = base_pairs + sentence_pairs
        random.Random(self.cfg.training_seed).shuffle(combined)
        zedd_corpus.save_pairs(combined, v2_pairs_path)
        print(f"[run_v2] combined: {len(base_pairs)} document-level + "
              f"{len(sentence_pairs)} sentence-level = {len(combined)} pairs")

        print("\n[run_v2] Step 2/5 — fine-tuning (-> v2 model path)…")
        pos = [p for p in combined if p.get("label") in (0, 0.0)]
        neg = [p for p in combined if p.get("label") in (1, 1.0)]
        examples = self.build_input_examples(pos, neg)
        with self._temp_paths(_model_out=v2_model):
            self.fine_tune(examples)   # sets self._finetuned_model, saves to v2_model

        print("\n[run_v2] Step 3/5 — building document-register baseline (v2)…")
        detector = build_document_register_baseline(
            model=self._finetuned_model, model_name=self.cfg.model_name,
        )
        v2_baseline.parent.mkdir(parents=True, exist_ok=True)
        detector.save_baseline(v2_baseline)
        print(f"[run_v2] baseline saved -> {v2_baseline}")

        print("\n[run_v2] Step 4/5 — per-mode calibration (v2)…")
        with self._temp_paths(_baseline_out=v2_baseline, _calibration_out=v2_calibration):
            calibration = self.calibrate_all_modes()

        print("\n[run_v2] Step 5/5 — done. NOT promoted to production yet.")
        print(f"  model       -> {v2_model}")
        print(f"  baseline    -> {v2_baseline}")
        print(f"  calibration -> {v2_calibration}")
        print("  Run `python src/zedd_trainer.py --validate-v2` to compare "
              "v1 vs v2 before promoting.")
        return calibration

    def validate_v2(self) -> dict:
        """
        Score BOTH v1 (production) and v2 (candidate) on the scaled
        dataset, all three modes each, and print a side-by-side table.
        Read-only — never writes anything, never promotes anything. This
        is the gate before promote_v2().
        """
        sys.path.insert(0, str(self._root / "src"))
        from sentence_transformers import SentenceTransformer
        from defense.zedd import ZEDDDetector

        attack_texts, benign_texts = self.load_scaled_dataset()

        configs = {
            "v1": (self._model_out, self._baseline_out, self._calibration_out),
            "v2": (
                self._root / "data" / "zedd_finetuned_model_v2",
                self._root / "data" / "zedd_finetuned_baseline_v2.json",
                self._root / "data" / "zedd_calibration_v2.json",
            ),
        }
        results: dict = {}
        for tag, (model_path, baseline_path, cal_path) in configs.items():
            if not (model_path.exists() and baseline_path.exists() and cal_path.exists()):
                print(f"[validate_v2] {tag}: missing artifacts at {model_path} — skipping.")
                continue
            model = SentenceTransformer(str(model_path))
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            results[tag] = {}
            for mode in ("document", "sentence", "dual_encoder"):
                det = ZEDDDetector(
                    model=model, model_name=self.cfg.model_name,
                    sentence_level=(mode == "sentence"),
                    dual_encoder=(mode == "dual_encoder"),
                )
                det.load_baseline(baseline_path)
                modes_cal = cal.get("modes", {})
                if mode in modes_cal and "runtime_drift_threshold" in modes_cal[mode]:
                    rt = float(modes_cal[mode]["runtime_drift_threshold"])
                    det.drift_threshold = rt
                    if mode == "document":
                        det.document_similarity_floor = rt
                tp = fn = fp = tn = 0
                for t in attack_texts:
                    r = det.detect(t)
                    tp += r.flagged; fn += (not r.flagged)
                for t in benign_texts:
                    r = det.detect(t)
                    fp += r.flagged; tn += (not r.flagged)
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                results[tag][mode] = {"recall": round(recall, 4), "fpr": round(fpr, 4),
                                       "tp": tp, "fn": fn, "fp": fp, "tn": tn}

        print("\n" + "=" * 70)
        print("v1 (production) vs v2 (candidate) — standalone ZEDD, all 3 modes")
        print("=" * 70)
        print(f"{'mode':<14}{'v1 recall':>11}{'v1 fpr':>9}   {'v2 recall':>11}{'v2 fpr':>9}")
        for mode in ("document", "sentence", "dual_encoder"):
            r1 = results.get("v1", {}).get(mode, {})
            r2 = results.get("v2", {}).get(mode, {})
            print(f"{mode:<14}"
                  f"{r1.get('recall','-'):>11}{r1.get('fpr','-'):>9}   "
                  f"{r2.get('recall','-'):>11}{r2.get('fpr','-'):>9}")
        return results

    def promote_v2(self) -> None:
        """
        Copy v2 artifacts over the production v1 paths, after backing up
        the current v1 artifacts as *_v1_backup. Call only after
        validate_v2() shows v2 is an improvement (document mode in
        particular must not have regressed).
        """
        import shutil

        v2_model = self._root / "data" / "zedd_finetuned_model_v2"
        v2_baseline = self._root / "data" / "zedd_finetuned_baseline_v2.json"
        v2_calibration = self._root / "data" / "zedd_calibration_v2.json"
        for p in (v2_model, v2_baseline, v2_calibration):
            if not p.exists():
                raise FileNotFoundError(f"promote_v2: missing {p} — run run_v2() first.")

        backups = []
        for prod, backup_suffix in (
            (self._model_out, "_v1_backup"),
            (self._baseline_out, "_v1_backup.json"),
            (self._calibration_out, "_v1_backup.json"),
        ):
            if prod.exists():
                backup = prod.with_name(prod.stem + backup_suffix) if prod.is_file() else Path(str(prod) + backup_suffix)
                if prod.is_dir():
                    if backup.exists():
                        shutil.rmtree(backup)
                    shutil.copytree(prod, backup)
                else:
                    shutil.copy2(prod, backup)
                backups.append(backup)

        if self._model_out.exists():
            shutil.rmtree(self._model_out)
        shutil.copytree(v2_model, self._model_out)
        shutil.copy2(v2_baseline, self._baseline_out)
        shutil.copy2(v2_calibration, self._calibration_out)

        print("[promote_v2] v2 promoted to production paths:")
        print(f"  {self._model_out}")
        print(f"  {self._baseline_out}")
        print(f"  {self._calibration_out}")
        print(f"[promote_v2] previous v1 artifacts backed up: {backups}")

    # ------------------------------------------------------------------
    # v3 pipeline — leakage-free, GPU-friendly split
    #   prepare_v3()  : generate all data locally (fast, CPU)
    #   <fine-tune on Colab GPU using data/v3/train_pairs.json>
    #   finish_v3()   : build baseline + calibrate (on the DISJOINT
    #                   calibration set) + validate — locally, fast
    #   validate_v3() : v1backup / current / v3, on eval set AND holdout
    #   promote_v3()  : copy v3 over production, back up current
    # ------------------------------------------------------------------

    _V3_DIR_NAME = "v3"

    def _v3_paths(self) -> dict:
        d = self._root / "data" / self._V3_DIR_NAME
        return {
            "dir":            d,
            "train_pairs":    d / "train_pairs.json",
            "calib_set":      d / "calibration_set.json",
            "baseline_corpus": d / "baseline_corpus_meta.json",
            "model":          self._root / "data" / "zedd_finetuned_model_v3",
            "baseline":       self._root / "data" / "zedd_finetuned_baseline_v3.json",
            "calibration":    self._root / "data" / "zedd_calibration_v3.json",
        }

    def prepare_v3(self) -> None:
        """
        Generate every input the v3 retrain needs, all from seeds disjoint
        from the evaluation set, and write them under data/v3/. Fast (no
        GPU, no fine-tuning). After this, fine-tune on Colab using
        data/v3/train_pairs.json, then run finish_v3().
        """
        sys.path.insert(0, str(self._root / "src"))
        import zedd_corpus

        p = self._v3_paths()
        p["dir"].mkdir(parents=True, exist_ok=True)

        print("[prepare_v3] generating training pairs (document + sentence level, "
              "every attack template covered)…")
        pairs = zedd_corpus.generate_training_pairs()
        zedd_corpus.save_json(pairs, p["train_pairs"])
        print(f"  {zedd_corpus.summarize_pairs(pairs)}")
        print(f"  -> {p['train_pairs']}")

        print("\n[prepare_v3] generating disjoint calibration dataset "
              "(own seed — this is the data-leakage fix)…")
        calib = zedd_corpus.generate_calibration_dataset()
        zedd_corpus.save_json(calib, p["calib_set"])
        print(f"  {len(calib['attacks'])} attacks + {len(calib['benign'])} benign "
              f"-> {p['calib_set']}")

        print("\n[prepare_v3] recording baseline corpus config "
              "(the corpus itself is regenerated deterministically in finish_v3)…")
        zedd_corpus.save_json(
            {
                "train_seed": zedd_corpus._TRAIN_SEED,
                "calib_seed": zedd_corpus._CALIB_SEED,
                "balanced_corpus_counts": zedd_corpus.BALANCED_CORPUS_COUNTS,
                "docs_per_category": zedd_corpus.DEFAULT_DOCS_PER_CATEGORY,
                "min_uses_per_template": zedd_corpus.DEFAULT_MIN_USES_PER_TEMPLATE,
            },
            p["baseline_corpus"],
        )
        print(f"  -> {p['baseline_corpus']}")

        print("\n[prepare_v3] DONE. Next:")
        print("  1. Fine-tune on Colab (GPU) using data/v3/train_pairs.json")
        print("     — see COLAB_GUIDE.md, step by step.")
        print("  2. Put the downloaded model at data/zedd_finetuned_model_v3/")
        print("  3. python src/zedd_trainer.py --finish-v3")

    def finish_v3(self, rebuild_baseline: bool = True) -> dict:
        """
        After the Colab fine-tune: build the v3 baseline from the balanced
        register corpus, calibrate all three modes ON THE DISJOINT
        CALIBRATION SET (never the eval set), and validate. Writes only
        *_v3 paths — promote_v3() moves them into production.

        rebuild_baseline=False (CLI: --recalibrate-v3) reuses the existing
        data/zedd_finetuned_baseline_v3.json and only re-runs calibration
        + validation — for iterating on threshold/calibration logic
        without paying the ~20k-embedding baseline rebuild again.
        """
        sys.path.insert(0, str(self._root / "src"))
        from sentence_transformers import SentenceTransformer
        from defense.zedd import build_document_register_baseline
        import zedd_corpus

        p = self._v3_paths()
        if not p["model"].exists():
            raise FileNotFoundError(
                f"v3 model not found at {p['model']}. Fine-tune on Colab first "
                "(COLAB_GUIDE.md), then unzip it there."
            )
        if not p["calib_set"].exists():
            raise FileNotFoundError(
                f"{p['calib_set']} missing — run `python src/zedd_trainer.py --prepare-v3` first."
            )

        print(f"[finish_v3] loading v3 model from {p['model']} …")
        self._finetuned_model = SentenceTransformer(str(p["model"]))

        if rebuild_baseline or not p["baseline"].exists():
            print("[finish_v3] building v3 baseline from the balanced register corpus…")
            detector = build_document_register_baseline(
                model=self._finetuned_model, model_name=self.cfg.model_name,
            )
            detector.save_baseline(p["baseline"])
            print(f"  -> {p['baseline']}")
        else:
            print(f"[finish_v3] reusing existing baseline {p['baseline']} "
                  "(rebuild_baseline=False)")

        print("[finish_v3] per-mode calibration on the DISJOINT calibration set…")
        calib = json.loads(p["calib_set"].read_text(encoding="utf-8"))
        with self._temp_paths(_baseline_out=p["baseline"], _calibration_out=p["calibration"]):
            calibration = self.calibrate_all_modes(
                calib_attack_texts=calib["attacks"],
                calib_benign_texts=calib["benign"],
                calibration_source="v3_disjoint_calibration_set",
            )

        print("\n[finish_v3] validating v3 vs current production…")
        self.validate_v3()

        print("\n[finish_v3] DONE. v3 written (NOT promoted):")
        print(f"  model       -> {p['model']}")
        print(f"  baseline    -> {p['baseline']}")
        print(f"  calibration -> {p['calibration']}")
        print("  Promote with: python src/zedd_trainer.py --promote-v3")
        return calibration

    def validate_v3(self, include_v1_backup: bool = False) -> dict:
        """
        Score `current` production and the `v3` candidate on BOTH the
        evaluation set (seed 42) and the disjoint holdout (the calibration
        set, seed 70123). Read-only. The holdout column is the honest one:
        no config was trained or calibrated on it.

        Pass include_v1_backup=True (CLI: --with-v1) to also score the
        pre-v2 backup — off by default because it triples the runtime and
        its numbers are already in the run logs.
        """
        sys.path.insert(0, str(self._root / "src"))
        from sentence_transformers import SentenceTransformer
        from defense.zedd import ZEDDDetector

        p = self._v3_paths()
        eval_attacks, eval_benign = self.load_scaled_dataset()
        holdout = None
        if p["calib_set"].exists():
            hj = json.loads(p["calib_set"].read_text(encoding="utf-8"))
            holdout = (hj["attacks"], hj["benign"])

        configs = {
            "current": (self._model_out, self._baseline_out, self._calibration_out),
            "v3": (p["model"], p["baseline"], p["calibration"]),
        }
        if include_v1_backup:
            configs = {
                "v1_backup": (
                    self._root / "data" / "zedd_finetuned_model_v1_backup",
                    self._root / "data" / "zedd_finetuned_baseline_v1_backup.json",
                    self._root / "data" / "zedd_calibration_v1_backup.json",
                ),
                **configs,
            }

        def score(model, baseline_path, cal, mode, attacks, benign):
            det = ZEDDDetector(
                model=model, model_name=self.cfg.model_name,
                sentence_level=(mode == "sentence"),
                dual_encoder=(mode == "dual_encoder"),
            )
            det.load_baseline(baseline_path)
            mc = cal.get("modes", {})
            if mode in mc and "runtime_drift_threshold" in mc[mode]:
                rt = float(mc[mode]["runtime_drift_threshold"])
                det.drift_threshold = rt
                if mode == "document":
                    det.document_similarity_floor = rt
            tp = sum(1 for t in attacks if det.detect(t).flagged)
            fp = sum(1 for t in benign if det.detect(t).flagged)
            rec = tp / len(attacks) if attacks else 0.0
            fpr = fp / len(benign) if benign else 0.0
            return round(rec, 4), round(fpr, 4)

        results: dict = {}
        for tag, (mp, bp, cp) in configs.items():
            if not (Path(mp).exists() and Path(bp).exists() and Path(cp).exists()):
                print(f"[validate_v3] {tag}: artifacts missing — skipped.")
                continue
            model = SentenceTransformer(str(mp))
            cal = json.loads(Path(cp).read_text(encoding="utf-8"))
            results[tag] = {}
            for mode in ("document", "sentence", "dual_encoder"):
                row = {"eval": score(model, bp, cal, mode, eval_attacks, eval_benign)}
                if holdout:
                    row["holdout"] = score(model, bp, cal, mode, holdout[0], holdout[1])
                results[tag][mode] = row

        print("\n" + "=" * 78)
        print("ZEDD standalone — recall / fpr   (holdout = disjoint, the honest column)")
        print("=" * 78)
        hdr = f"{'mode':<13}"
        for tag in results:
            hdr += f"{tag+' eval':>18}"
            if holdout:
                hdr += f"{tag+' holdout':>18}"
        print(hdr)
        for mode in ("document", "sentence", "dual_encoder"):
            line = f"{mode:<13}"
            for tag in results:
                e = results[tag][mode]["eval"]
                line += f"{f'{e[0]:.2f}/{e[1]:.2f}':>18}"
                if holdout:
                    h = results[tag][mode]["holdout"]
                    line += f"{f'{h[0]:.2f}/{h[1]:.2f}':>18}"
            print(line)
        return results

    def promote_v3(self) -> None:
        """Copy v3 artifacts over production (backing up current as
        *_pre_v3_backup first). Run only after validate_v3() looks good."""
        import shutil
        p = self._v3_paths()
        for k in ("model", "baseline", "calibration"):
            if not p[k].exists():
                raise FileNotFoundError(f"promote_v3: missing {p[k]} — run --finish-v3 first.")

        for prod, suffix in (
            (self._model_out, "_pre_v3_backup"),
            (self._baseline_out, "_pre_v3_backup.json"),
            (self._calibration_out, "_pre_v3_backup.json"),
        ):
            if not prod.exists():
                continue
            backup = (prod.with_name(prod.stem + suffix) if prod.is_file()
                      else Path(str(prod) + suffix))
            if prod.is_dir():
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.copytree(prod, backup)
            else:
                shutil.copy2(prod, backup)

        if self._model_out.exists():
            shutil.rmtree(self._model_out)
        shutil.copytree(p["model"], self._model_out)
        shutil.copy2(p["baseline"], self._baseline_out)
        shutil.copy2(p["calibration"], self._calibration_out)
        print("[promote_v3] v3 promoted to production. Previous artifacts backed "
              "up as *_pre_v3_backup.")

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
    print("  python src/zedd_trainer.py --v2          # (legacy) train candidate v2 locally")
    print("  python src/zedd_trainer.py --validate-v2 # (legacy) compare v1 vs v2")
    print("  python src/zedd_trainer.py --promote-v2  # (legacy) promote v2")
    print()
    print("  v3 (leakage-free, GPU-friendly) workflow:")
    print("  python src/zedd_trainer.py --prepare-v3  # 1. generate all data locally (fast) -> data/v3/")
    print("  <fine-tune on Colab GPU using data/v3/train_pairs.json — see COLAB_GUIDE.md>")
    print("  python src/zedd_trainer.py --finish-v3      # 2. baseline + calibrate (disjoint) + validate")
    print("  python src/zedd_trainer.py --recalibrate-v3 # re-calibrate + validate, reuse existing v3 baseline")
    print("  python src/zedd_trainer.py --validate-v3    # compare current / v3 (eval + holdout); add --with-v1")
    print("  python src/zedd_trainer.py --promote-v3  # 3. copy v3 over production (backs up current)")


if __name__ == "__main__":
    args = sys.argv[1:]

    cfg     = TrainingConfig()
    trainer = ZEDDTrainer(config=cfg)

    if "--prepare-v3" in args:
        trainer.prepare_v3()

    elif "--finish-v3" in args:
        trainer.finish_v3()

    elif "--recalibrate-v3" in args:
        trainer.finish_v3(rebuild_baseline=False)

    elif "--validate-v3" in args:
        trainer.validate_v3(include_v1_backup=("--with-v1" in args))

    elif "--promote-v3" in args:
        trainer.promote_v3()

    elif "--v2" in args:
        trainer.run_v2()

    elif "--validate-v2" in args:
        trainer.validate_v2()

    elif "--promote-v2" in args:
        trainer.promote_v2()

    elif "--calibrate" in args:
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
