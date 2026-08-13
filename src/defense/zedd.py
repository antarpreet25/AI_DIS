"""
defense/zedd.py

Layer 3 of the defense stack: Zero-Shot Embedding Drift Detection (ZEDD),
based on Sekar et al. (2026).

CONCEPTUAL BASIS
-----------------
Convert input text to a semantic embedding, compare it against a baseline
representing normal power-grid language, and flag inputs whose similarity
to that baseline falls below a threshold. No training, no labeled attack
examples, no gradient access required — this is a genuinely zero-shot,
training-free detector, unlike Layer 2's API-constrained adaptation of
Chen et al.

NOVEL CONTRIBUTION: CATEGORY-AWARE TWO-STAGE CENTROID PRE-FILTERING
---------------------------------------------------------------------
A naive ZEDD implementation maintains ONE global centroid for "normal
text" and compares every input against it. That collapses genuinely
different registers of normal language (a terse sensor reading and a
discursive maintenance report are both "normal" but semantically distant
from each other) into a single point, which forces either a loose
threshold (misses real drift) or a tight one (flags legitimate variety).

This module instead maintains one centroid PER CATEGORY of normal content
(sensor_readings, maintenance_reports, operator_notes, system_alerts),
each with its OWN configurable drift_threshold — tight for formulaic
sensor text, looser for varied prose — and runs detection in two stages:

    Stage 1 (pre-filter, cheap): compare the input against all category
        centroids — just N dot products for N categories. If it is not
        sufficiently close to ANY category (below pre_filter_threshold
        against all of them), flag it immediately as drift without doing
        any further work. This is the fast rejection path.

    Stage 2 (detailed): for inputs that pass Stage 1, take the CLOSEST
        category and compare against THAT category's specific
        drift_threshold for the final determination.

The pre_filter_enabled toggle exists specifically so this two-stage
design can be ablated: run the full evaluation with pre_filter_enabled
=True and =False, holding everything else constant, and compare
detection accuracy and speed. That comparison is the empirical
justification for this being a genuine contribution rather than just
restating Sekar et al. with extra steps.

BASELINE IS STATIC BY DESIGN
------------------------------
This module deliberately does NOT implement incremental baseline
updating (no "learn from new normal traffic over time" mechanism). A
detector that updates its notion of "normal" from live traffic is
vulnerable to baseline poisoning: an attacker who can influence what the
system sees as normal (e.g. by slowly drifting inputs) can walk the
baseline toward accepting their attack as normal. A static, offline-built
baseline that only changes via an explicit, reviewed build_baseline() or
load_baseline() call does not have this vulnerability. Document this as
a deliberate security property, not a missing feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Log file location (same pattern as filtering.py / defensive_tokens.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZEDD_LOG_PATH = _PROJECT_ROOT / "data" / "logs" / "zedd_log.jsonl"

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
KNOWN_CATEGORIES = ("sensor_readings", "maintenance_reports", "operator_notes", "system_alerts")


def _safe_text(value) -> str:
    """Coerce arbitrary input into a string without raising — same
    defensive pattern used throughout the defense stack."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return ""


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, guarding against a zero-norm vector rather
    than dividing by zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, safe against zero vectors on either side."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ZEDDResult:
    """Outcome of screening a single piece of text for embedding drift."""
    flagged: bool
    similarity_score: float          # similarity to the closest category centroid
    closest_category: Optional[str]
    pre_filter_passed: bool
    stage: int                       # 0 = not flagged, 1 = caught at pre-filter, 2 = caught at detailed check
    reason: Optional[str]
    confidence: float                # 0.0 when not flagged, matches filtering.py's convention

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ZEDDDetector:
    """
    Zero-Shot Embedding Drift Detection with category-aware two-stage
    centroid pre-filtering.

    Usage:
        zedd = ZEDDDetector()
        zedd.build_baseline(normal_texts, categories,
                             category_thresholds={"sensor_readings": 0.30})
        result = zedd.detect(input_text)
        zedd.save_baseline("data/baseline/zedd_baseline.json")

        # later / elsewhere:
        zedd2 = ZEDDDetector()
        zedd2.load_baseline("data/baseline/zedd_baseline.json")

    Ablation (the empirical case for the two-stage design):
        zedd_full = ZEDDDetector(pre_filter_enabled=True)
        zedd_no_prefilter = ZEDDDetector(pre_filter_enabled=False)
        # load the SAME baseline into both, run the SAME evaluation set
        # through both, compare accuracy and wall-clock time.
    """

    def __init__(
        self,
        drift_threshold: float = 0.35,
        pre_filter_threshold: float = 0.25,
        pre_filter_enabled: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
        model=None,
        log_path: Optional[Path] = None,
    ):
        self.drift_threshold = drift_threshold
        self.pre_filter_threshold = pre_filter_threshold
        self.pre_filter_enabled = pre_filter_enabled
        self._model_name = model_name
        # Allow injecting a pre-loaded model so multiple ZEDDDetector
        # instances (e.g. the pre_filter_enabled=True/False ablation pair)
        # can share ONE loaded SentenceTransformer instead of each paying
        # the load cost separately.
        self._model = model

        # category -> {"centroid": np.ndarray, "threshold": float}
        self.category_centroids: dict = {}
        self.global_centroid: Optional[np.ndarray] = None
        self._baseline_built = False

        self.log_path = Path(log_path) if log_path else ZEDD_LOG_PATH

    # ------------------------------------------------------------------
    # Model loading (lazy — no download/load cost until actually needed)
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it with "
                    "`pip install sentence-transformers` — it should already be "
                    "in requirements.txt for this project."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single string and L2-normalize the result."""
        model = self._get_model()
        raw = model.encode([text])[0]
        vec = np.asarray(raw, dtype=np.float64)
        return _normalize_vector(vec)

    def _embed_many(self, texts: list) -> np.ndarray:
        """Embed a batch of strings (more efficient than one-at-a-time
        during baseline building) and L2-normalize each row."""
        model = self._get_model()
        raw = model.encode(list(texts))
        vecs = np.asarray(raw, dtype=np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return vecs / norms

    # ------------------------------------------------------------------
    # Baseline construction
    # ------------------------------------------------------------------

    def build_baseline(
        self,
        normal_texts: list,
        categories: list,
        category_thresholds: Optional[dict] = None,
    ) -> None:
        """
        Build the category centroids (and the global centroid) from a
        parallel list of texts and category labels.

        category_thresholds: optional dict mapping category name -> a
        drift_threshold specific to that category (e.g. tighter for
        formulaic sensor text, looser for varied maintenance prose).
        Categories not present in this dict fall back to self.drift_threshold.

        This is a SETUP-time function, not a runtime detection path — it
        is allowed to raise on genuinely malformed configuration (e.g.
        mismatched list lengths), unlike detect(), which must never raise
        on live input.
        """
        if len(normal_texts) != len(categories):
            raise ValueError(
                f"normal_texts (len={len(normal_texts)}) and categories "
                f"(len={len(categories)}) must be the same length."
            )
        if not normal_texts:
            raise ValueError("normal_texts is empty — nothing to build a baseline from.")

        category_thresholds = category_thresholds or {}

        # Group texts by category
        texts_by_category: dict = {}
        for text, category in zip(normal_texts, categories):
            texts_by_category.setdefault(category, []).append(_safe_text(text))

        self.category_centroids = {}
        for category, texts in texts_by_category.items():
            embeddings = self._embed_many(texts)          # shape (n, 384), rows unit-norm
            centroid = _normalize_vector(embeddings.mean(axis=0))
            threshold = category_thresholds.get(category, self.drift_threshold)
            self.category_centroids[category] = {"centroid": centroid, "threshold": threshold}

        # Global centroid = mean of the per-category centroids (equal
        # weight per category, NOT weighted by example count per category)
        stacked = np.stack([info["centroid"] for info in self.category_centroids.values()])
        self.global_centroid = _normalize_vector(stacked.mean(axis=0))

        self._baseline_built = True

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, input_text) -> ZEDDResult:
        """
        Screen input_text for embedding drift against the baseline.

        Raises:
            RuntimeError: if called before build_baseline() or
                load_baseline() has established a baseline. This is a
                deliberate, clear failure rather than letting a bare
                numpy/AttributeError surface from deep in the method.
        """
        if not self._baseline_built:
            raise RuntimeError(
                "ZEDDDetector.detect() was called before a baseline was "
                "established. Call build_baseline(normal_texts, categories) "
                "or load_baseline(path) first — see build_powergrid_baseline() "
                "for a ready-made starting baseline."
            )

        text = _safe_text(input_text)
        vec = self._embed(text)

        category_sims = {
            category: _cosine_similarity(vec, info["centroid"])
            for category, info in self.category_centroids.items()
        }
        closest_category = max(category_sims, key=category_sims.get) if category_sims else None
        closest_sim = category_sims[closest_category] if closest_category is not None else 0.0

        # --- Stage 1: cheap pre-filter against ALL category centroids ---
        if self.pre_filter_enabled:
            if closest_sim < self.pre_filter_threshold:
                result = ZEDDResult(
                    flagged=True,
                    similarity_score=round(closest_sim, 4),
                    closest_category=closest_category,
                    pre_filter_passed=False,
                    stage=1,
                    reason=(
                        f"pre_filter: input far from all category centroids "
                        f"(closest={closest_category}, similarity={closest_sim:.3f} "
                        f"< pre_filter_threshold={self.pre_filter_threshold})"
                    ),
                    confidence=round(min(0.95, 0.5 + (self.pre_filter_threshold - closest_sim)), 2),
                )
                self._log(result)
                return result
            pre_filter_passed = True
        else:
            # Stage 1 skipped entirely — proceed straight to Stage 2. This
            # branch is what makes the pre_filter_enabled=False ablation run
            # possible: same Stage 2 logic, just without the cheap early exit.
            pre_filter_passed = True

        # --- Stage 2: detailed check against the CLOSEST category's OWN threshold ---
        if closest_category is None:
            result = ZEDDResult(
                flagged=True,
                similarity_score=0.0,
                closest_category=None,
                pre_filter_passed=pre_filter_passed,
                stage=2,
                reason="stage2: no category centroids available for comparison (baseline has no categories)",
                confidence=0.5,
            )
            self._log(result)
            return result

        category_threshold = self.category_centroids[closest_category]["threshold"]

        if closest_sim < category_threshold:
            result = ZEDDResult(
                flagged=True,
                similarity_score=round(closest_sim, 4),
                closest_category=closest_category,
                pre_filter_passed=pre_filter_passed,
                stage=2,
                reason=(
                    f"stage2: similarity to closest category '{closest_category}' "
                    f"({closest_sim:.3f}) below its drift_threshold ({category_threshold:.3f})"
                ),
                confidence=round(min(0.95, 0.5 + (category_threshold - closest_sim)), 2),
            )
            self._log(result)
            return result

        # Not flagged
        result = ZEDDResult(
            flagged=False,
            similarity_score=round(closest_sim, 4),
            closest_category=closest_category,
            pre_filter_passed=pre_filter_passed,
            stage=0,
            reason=None,
            confidence=0.0,
        )
        self._log(result)
        return result

    # ------------------------------------------------------------------
    # Baseline persistence
    # ------------------------------------------------------------------

    def save_baseline(self, path) -> None:
        """Save category centroids, the global centroid, thresholds, and
        metadata to a JSON file. Embeddings are serialized as plain lists."""
        if not self._baseline_built:
            raise RuntimeError("No baseline to save — call build_baseline() first.")

        payload = {
            "model_name": self._model_name,
            "drift_threshold": self.drift_threshold,
            "pre_filter_threshold": self.pre_filter_threshold,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "categories": {
                category: {
                    "centroid": info["centroid"].tolist(),
                    "threshold": info["threshold"],
                }
                for category, info in self.category_centroids.items()
            },
            "global_centroid": (
                self.global_centroid.tolist() if self.global_centroid is not None else None
            ),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_baseline(self, path) -> None:
        """
        Restore centroids, per-category thresholds, and the global
        centroid from a file written by save_baseline().

        Note: this restores drift_threshold/pre_filter_threshold/category
        thresholds from the FILE (so a saved baseline is reproducible
        regardless of what an instance happened to be constructed with),
        but deliberately does NOT touch self.pre_filter_enabled — that is
        a runtime experiment toggle, not a property of the baseline data,
        which is exactly what makes the ablation comparison clean: load
        the SAME baseline into two detectors and only vary pre_filter_enabled.
        """
        path = Path(path)
        with open(path, "r") as f:
            payload = json.load(f)

        loaded_model_name = payload.get("model_name")
        if loaded_model_name and loaded_model_name != self._model_name:
            raise ValueError(
                f"Baseline at {path} was built with model '{loaded_model_name}', "
                f"but this detector is configured for '{self._model_name}'. "
                "Embeddings from different models are not comparable — use a "
                "detector configured with the same model_name that built this baseline."
            )

        self.category_centroids = {
            category: {
                "centroid": np.array(info["centroid"], dtype=np.float64),
                "threshold": info["threshold"],
            }
            for category, info in payload["categories"].items()
        }
        self.global_centroid = (
            np.array(payload["global_centroid"], dtype=np.float64)
            if payload.get("global_centroid") is not None
            else None
        )
        self.drift_threshold = payload.get("drift_threshold", self.drift_threshold)
        self.pre_filter_threshold = payload.get("pre_filter_threshold", self.pre_filter_threshold)
        self._baseline_built = True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, result: ZEDDResult) -> None:
        """Append one JSON line per detect() call — every call is logged,
        not just flagged ones, matching defensive_tokens.py's pattern
        (this layer's overhead/behavior on CLEAN traffic is just as
        relevant to the evaluation as its behavior on attacks)."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "flagged": result.flagged,
                "stage": result.stage,
                "similarity_score": result.similarity_score,
                "closest_category": result.closest_category,
                "pre_filter_passed": result.pre_filter_passed,
                "reason": result.reason,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Logging must never crash detection itself.
            pass

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """
        Read the ZEDD log and return aggregate counts, following the same
        defensive pattern as filtering.py / defensive_tokens.py:
        same method name/signature, all-zero structure if the log doesn't
        exist yet, malformed lines skipped rather than raising.

        Structure:
            {
                "total_detections": int,
                "total_flagged": int,
                "by_stage": {"0": count, "1": count, "2": count},
                "by_closest_category": {category: count, ...},
                "avg_similarity_score": float,
            }
        """
        stats = {
            "total_detections": 0,
            "total_flagged": 0,
            "by_stage": {},
            "by_closest_category": {},
            "avg_similarity_score": 0.0,
        }

        if not self.log_path.exists():
            return stats

        total_similarity = 0.0

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total_detections"] += 1

                if entry.get("flagged"):
                    stats["total_flagged"] += 1

                stage = str(entry.get("stage", "unknown"))
                stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1

                category = entry.get("closest_category") or "none"
                stats["by_closest_category"][category] = (
                    stats["by_closest_category"].get(category, 0) + 1
                )

                total_similarity += entry.get("similarity_score", 0.0) or 0.0

        if stats["total_detections"] > 0:
            stats["avg_similarity_score"] = round(
                total_similarity / stats["total_detections"], 4
            )

        return stats


# ---------------------------------------------------------------------------
# Convenience baseline builder
# ---------------------------------------------------------------------------

# Hardcoded example texts representative of normal power-grid language,
# covering all four categories. Deliberately several examples per category
# (not just one) so each centroid reflects a small cluster rather than a
# single, fragile point. This is a REASONABLE starting baseline for
# development and demos — for the dissertation's actual evaluation runs,
# consider building the baseline from a larger, more systematically
# sampled set of normal texts (e.g. drawn from sensor_generator.py's
# "normal" scenario descriptions plus a wider set of maintenance-report
# style text) so the reported numbers reflect a baseline you can defend
# in the methodology chapter, not just this convenience default.
_POWERGRID_BASELINE_TEXTS = {
    "sensor_readings": [
        "Sensor T-12 reading: voltage 32.8kV, load 58%, temperature 51.3C, vibration 0.9mm/s.",
        "Transformer B-4 oil moisture at 9ppm, dissolved gas levels nominal, frequency 50.02Hz.",
        "Load reading 64%, top-oil temperature 55.1C, no anomalies detected in dissolved gas panel.",
        "Voltage 33.1kV, frequency 49.98Hz, vibration steady at 0.7mm/s across last interval.",
        "CO2 1490ppm, CH4 19.8ppm, H2 14.6ppm - dissolved gas readings within Condition 1 range.",
        "Substation B-7 telemetry: load 47%, temperature 42.0C, voltage 33.0kV, all nominal.",
    ],
    "maintenance_reports": [
        "Routine inspection of substation B-7 completed; bushings, tap changer, and cooling fans "
        "all within spec. No corrective action required.",
        "Technician replaced worn gasket on cooling radiator during scheduled maintenance window; "
        "oil sample sent for lab analysis.",
        "Annual dissolved gas analysis sampling performed on transformer T-12; results pending "
        "from laboratory, expected within 5 business days.",
        "Tap changer contact wear inspected and found acceptable per manufacturer guidelines; "
        "next inspection scheduled for Q3.",
        "Cooling fan motor on unit 4 showed minor bearing noise; flagged for replacement during "
        "next planned outage.",
        "Quarterly infrared thermography scan of busbar connections completed with no hotspots identified.",
    ],
    "operator_notes": [
        "Operator handover: all systems nominal at shift change, no outstanding alarms.",
        "Noted slight increase in ambient temperature affecting cooling fan duty cycle; monitoring continues.",
        "Load forecast for tomorrow indicates peak demand around 14:00; recommend proactive tap adjustment.",
        "No unusual readings observed during overnight shift; log reviewed and closed.",
        "Reminder logged to schedule oil sample collection next week per maintenance calendar.",
        "Weather advisory noted for high winds tomorrow; will monitor vibration readings more closely.",
    ],
    "system_alerts": [
        "ALERT: Tap changer operation completed successfully, position changed from 5 to 6.",
        "WARNING: Ambient temperature sensor reading near upper calibration bound, verify sensor calibration.",
        "ALERT: Scheduled maintenance window starting in 30 minutes for substation B-7.",
        "NOTICE: Dissolved gas sampling valve accessed for routine lab sample collection.",
        "ALERT: Communication link to remote sensor node restored after brief interruption.",
        "WARNING: Backup generator test scheduled for 02:00, brief telemetry gap expected.",
    ],
}


def build_powergrid_baseline(
    pre_filter_enabled: bool = True,
    drift_threshold: float = 0.35,
    pre_filter_threshold: float = 0.25,
    category_thresholds: Optional[dict] = None,
    model=None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ZEDDDetector:
    """
    Convenience function: build a ready-to-use ZEDDDetector from hardcoded
    example texts representative of normal power-grid language, without
    requiring a separate data-collection step. This is what main.py calls
    to initialize the system.

    For anything beyond development/demo use, consider building the
    baseline from a larger, more representative text sample and saving it
    with save_baseline() so it doesn't need to be reconstructed (and
    re-embedded) on every run.
    """
    detector = ZEDDDetector(
        drift_threshold=drift_threshold,
        pre_filter_threshold=pre_filter_threshold,
        pre_filter_enabled=pre_filter_enabled,
        model_name=model_name,
        model=model,
    )

    normal_texts = []
    categories = []
    for category, texts in _POWERGRID_BASELINE_TEXTS.items():
        normal_texts.extend(texts)
        categories.extend([category] * len(texts))

    detector.build_baseline(normal_texts, categories, category_thresholds=category_thresholds)
    return detector


def build_enriched_baseline(
    model=None,
    model_name: str = DEFAULT_MODEL_NAME,
    pre_filter_enabled: bool = True,
    drift_threshold: float = 0.35,
    pre_filter_threshold: float = 0.25,
    category_thresholds: Optional[dict] = None,
    data_root=None,
) -> "ZEDDDetector":
    """
    Build a ZEDDDetector baseline from real generated sensor data rather
    than the handful of hand-written example sentences in
    build_powergrid_baseline(). Each category centroid is computed from
    40-100 real texts instead of 6, making it far more representative of
    the full spread of legitimate power-grid language.

    WHY THIS IMPROVES DETECTION
    ----------------------------
    ZEDD flags inputs whose cosine similarity to the closest category
    centroid falls below drift_threshold. A centroid computed from 6
    hand-written examples is a coarse approximation of the true centre of
    that category's embedding cluster. With 40-100 real examples the
    centroid is more accurate, which means:
      - Legitimate documents score higher (they're closer to a better
        centroid), reducing false positive risk.
      - Attack documents embedded in legitimate-sounding wrappers may
        score differently depending on how much the wrapper dominates the
        embedding — this is investigated empirically by running the
        ZEDD diagnostic after building this baseline.

    DATA SOURCES (all produced by sensor_generator.py --all)
    ----------------------------------------------------------
    sensor_readings   ← normal.json + gradual_load_increase.json (~98 texts)
    maintenance_reports ← tap_changer_stress.json (~49 texts)
    operator_notes    ← developing_thermal_fault.json + arcing_fault.json
    system_alerts     ← grid_disturbance.json + false_data_injection.json

    GRACEFUL DEGRADATION
    ---------------------
    If any log file is missing (e.g. sensor_generator.py hasn't been run
    yet), that file is skipped with a printed warning. If an entire
    category ends up with zero texts after loading all available files,
    the category falls back to the corresponding hand-written texts from
    _POWERGRID_BASELINE_TEXTS so the baseline always has at least one
    data point per category.

    FIELD NAMES
    -----------
    Uses the actual field names from sensor_generator.py's JSON output:
    voltage_kv, load_pct, temperature_c, vibration_mm_s, frequency_hz,
    oil_moisture_ppm. The dissolved_gas_ppm field is a nested dict and is
    intentionally excluded — sentence-level embeddings of dissolved gas
    values add noise rather than signal for this purpose.

    Args:
        model: optional pre-loaded SentenceTransformer instance (shared
            with RAGMemory to avoid loading the model twice).
        model_name: model to load if model is None.
        pre_filter_enabled: passed to ZEDDDetector — set False for
            ablation testing without the two-stage pre-filter.
        drift_threshold: global threshold; per-category overrides via
            category_thresholds take precedence.
        pre_filter_threshold: stage-1 cutoff below which an input is
            flagged without proceeding to stage-2.
        category_thresholds: optional per-category drift thresholds,
            e.g. {"sensor_readings": 0.30, "maintenance_reports": 0.40}.
        data_root: project root Path. Defaults to two levels above this
            file (i.e. AI_DIS/).

    Returns:
        A fully initialized ZEDDDetector with the enriched baseline built
        and ready to call detect() on.
    """
    root = Path(data_root) if data_root else Path(__file__).resolve().parents[2]
    logs_dir = root / "data" / "logs"

    # ------------------------------------------------------------------
    # File-to-category mapping
    # ------------------------------------------------------------------
    FILE_CATEGORY_MAP = {
        "sensor_readings": [
            "normal.json",
            "gradual_load_increase.json",
        ],
        "maintenance_reports": [
            "tap_changer_stress.json",
        ],
        "operator_notes": [
            "developing_thermal_fault.json",
            "arcing_fault.json",
        ],
        "system_alerts": [
            "grid_disturbance.json",
            "false_data_injection.json",
        ],
    }

    def _reading_to_text(r: dict) -> str:
        """Convert one sensor reading dict to a natural-language sentence.
        Uses only scalar sensor fields — dissolved_gas_ppm is a nested
        dict and is deliberately excluded."""
        s = r.get("sensors", {})
        parts = []
        if "voltage_kv" in s:
            parts.append(f"Voltage {s['voltage_kv']:.1f}kV")
        if "load_pct" in s:
            parts.append(f"load {s['load_pct']:.0f}%")
        if "temperature_c" in s:
            parts.append(f"temperature {s['temperature_c']:.1f}C")
        if "vibration_mm_s" in s:
            parts.append(f"vibration {s['vibration_mm_s']:.2f}mm/s")
        if "frequency_hz" in s:
            parts.append(f"frequency {s['frequency_hz']:.2f}Hz")
        if "oil_moisture_ppm" in s:
            parts.append(f"oil moisture {s['oil_moisture_ppm']:.1f}ppm")
        base = ", ".join(parts) + "." if parts else ""
        description = r.get("description", "")
        return f"{base} {description}".strip() if description else base

    def _load_log(filename: str) -> list:
        """Load readings from one log file. Returns [] on any error."""
        path = logs_dir / filename
        if not path.exists():
            print(f"[build_enriched_baseline] WARNING: {filename} not found "
                  f"at {path} — skipping.")
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            readings = data.get("readings", [])
            if not readings:
                print(f"[build_enriched_baseline] WARNING: {filename} has no "
                      f"'readings' key or empty list — skipping.")
            return readings
        except Exception as exc:
            print(f"[build_enriched_baseline] WARNING: failed to load "
                  f"{filename}: {type(exc).__name__}: {exc} — skipping.")
            return []

    # ------------------------------------------------------------------
    # Build per-category text lists from log files
    # ------------------------------------------------------------------
    category_texts: dict = {cat: [] for cat in FILE_CATEGORY_MAP}

    for category, filenames in FILE_CATEGORY_MAP.items():
        for filename in filenames:
            readings = _load_log(filename)
            texts = [_reading_to_text(r) for r in readings if _reading_to_text(r)]
            category_texts[category].extend(texts)
        print(f"[build_enriched_baseline] {category}: "
              f"{len(category_texts[category])} texts loaded from files.")

    # ------------------------------------------------------------------
    # Fallback: any category with zero texts gets the hand-written
    # examples from _POWERGRID_BASELINE_TEXTS so the baseline always
    # has at least something per category.
    # ------------------------------------------------------------------
    for category, texts in category_texts.items():
        if not texts:
            fallback = _POWERGRID_BASELINE_TEXTS.get(category, [])
            print(f"[build_enriched_baseline] WARNING: {category} has no data "
                  f"from log files — falling back to {len(fallback)} "
                  f"hand-written baseline texts.")
            category_texts[category] = list(fallback)

    # ------------------------------------------------------------------
    # Build and return the detector
    # ------------------------------------------------------------------
    detector = ZEDDDetector(
        drift_threshold=drift_threshold,
        pre_filter_threshold=pre_filter_threshold,
        pre_filter_enabled=pre_filter_enabled,
        model_name=model_name,
        model=model,
    )

    all_texts = []
    all_categories = []
    for category, texts in category_texts.items():
        all_texts.extend(texts)
        all_categories.extend([category] * len(texts))

    detector.build_baseline(
        all_texts,
        all_categories,
        category_thresholds=category_thresholds,
    )

    total = len(all_texts)
    print(f"[build_enriched_baseline] Baseline built from {total} total texts "
          f"across {len(category_texts)} categories.")
    return detector
