"""
main.py

The pipeline conductor. Wires all five defense layers together in
sequence and decides what happens at each stage. Holds NO detection
logic of its own — every blocking/flagging decision is made by calling
into a defense/ module or grid_agent.py; main.py only sequences those
calls and interprets their boolean results. If you find yourself writing
anything here that looks like pattern matching, threshold comparison, or
embedding math, it belongs in one of the defense files instead.

PIPELINE ORDER (see run() below)
-----------------------------------
    input_text
      -> Layer 1 (filtering.screen_input)      -- may block, stores attack if so
      -> Layer 2 (defensive_tokens.apply_adaptive) -- always runs if Layer 1 passed, never blocks
      -> Layer 3 (zedd.detect, on ORIGINAL text)     -- may block, stores attack if so
      -> Layer 4 (rag_memory.check, on ORIGINAL text) -- may block, does NOT re-store (already known)
      -> agent call (grid_agent.analyze_readings)
      -> Layer 1 output (filtering.screen_output)     -- may block, does NOT store (agent output, not an input attack pattern)
      -> Layer 5 (human_loop.evaluate)                -- never blocks the PIPELINE, gates the agent's tool_call

NO EXPLICIT LOGGING CODE IN THIS FILE
----------------------------------------
Every layer already logs its own calls internally (filtering.py,
defensive_tokens.py, zedd.py, rag_memory.py, human_loop.py each write
their own JSONL log on every call). The pipeline spec's "log" step at
each stage is therefore already satisfied simply by calling the layer —
there is no separate logging call for main.py to make.

WHY A CATEGORY-CLASSIFICATION FALLBACK IS NEEDED FOR ZEDD-CAUGHT ATTACKS
-----------------------------------------------------------------------------
RAGMemory organizes attack memory by ATTACK TYPE (direct_injection,
indirect_injection, escalation, encoding_obfuscation). ZEDDDetector
organizes ITS categories by CONTENT TYPE (sensor_readings,
maintenance_reports, operator_notes, system_alerts) — a completely
different taxonomy. So when ZEDD (not filtering) is what flags an input,
there is no direct way to know which RAG category to store it under.
Rather than write new classification logic here (forbidden by this
file's own design principle), _store_blocked_attack() calls filtering.py's
classify_attack_category() — a small, reusable method added to Layer 1
specifically to answer this question by reusing its EXISTING pattern
checks, without requiring a block. If even that returns nothing (expected
for attacks specifically designed to evade pattern matching — arguably
the most interesting case ZEDD exists to catch), this falls back to
'indirect_injection', on the reasoning that an attack evading every
keyword-based category by definition fits the "indirect" category's own
definition better than a guess at 'direct_injection'/'escalation'/
'encoding_obfuscation' would.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sensor_generator import build_scenarios, generate_readings, get_rolling_window
from grid_agent import analyze_readings, GridAssessment
from defense.filtering import InputOutputFilter, FilterResult
from defense.defensive_tokens import DefensiveTokens
from defense.zedd import ZEDDDetector, ZEDDResult, build_enriched_baseline
from defense.rag_memory import RAGMemory, RAGResult, ATTACK_CATEGORIES
from defense.human_loop import HumanLoopGate, HumanLoopResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZEDD_BASELINE_PATH = _PROJECT_ROOT / "data" / "zedd_baseline.json"
DEFAULT_CHROMADB_PATH = _PROJECT_ROOT / "data" / "chromadb"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# See module docstring's "WHY A CATEGORY-CLASSIFICATION FALLBACK..." section.
_UNCLASSIFIED_ATTACK_FALLBACK_CATEGORY = "indirect_injection"

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    blocked: bool
    blocked_by: Optional[str]  # 'filtering' / 'zedd' / 'rag_memory' / 'filtering_output' / None
    filter_result: Optional[FilterResult]
    zedd_result: Optional[ZEDDResult]
    rag_result: Optional[RAGResult]
    assessment: Optional[GridAssessment]
    human_loop_result: Optional[HumanLoopResult]
    tokens_added: int
    pipeline_duration_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    """
    Usage:
        pipeline = Pipeline()
        result = pipeline.run(input_text, sensor_readings, n_window=10)

        # Evaluation helpers:
        result = pipeline.run_attack_sample(sample, sensor_readings)
        results = pipeline.run_evaluation(dataset, sensor_readings)

    A NOTE ON filter_result: the pipeline runs filtering TWICE (input and
    output). PipelineResult has a single filter_result field, not separate
    input/output fields, per the given interface — this holds whichever
    filter check was RELEVANT to how the run concluded: the input-stage
    FilterResult if blocked there, the output-stage FilterResult if it got
    that far (whether or not that one blocked). This is a deliberate
    interpretation of an interface that doesn't disambiguate the two
    itself; flagging it explicitly rather than leaving it implicit.
    """

    def __init__(
        self,
        zedd_baseline_path=None,
        chromadb_path=None,
        model_name: str = DEFAULT_MODEL_NAME,
        model=None,
    ):
        self.zedd_baseline_path = Path(zedd_baseline_path) if zedd_baseline_path else DEFAULT_ZEDD_BASELINE_PATH
        chromadb_path = Path(chromadb_path) if chromadb_path else DEFAULT_CHROMADB_PATH

        # Load ONE SentenceTransformer instance, shared between ZEDD and
        # RAG memory — avoids loading the model twice. `model=` allows a
        # caller (tests, evaluator.py running multiple Pipelines) to
        # inject an already-loaded model instead.
        self._shared_model = model if model is not None else self._load_shared_model(model_name)

        # Layer 1
        self.filter = InputOutputFilter()
        # Layer 2
        self.tokens = DefensiveTokens()

        # Layer 3 — load an existing baseline if one has been saved,
        # otherwise build the convenience baseline and save it so the
        # next run doesn't have to rebuild it.
        if self.zedd_baseline_path.exists():
            self.zedd = ZEDDDetector(model=self._shared_model, model_name=model_name)
            self.zedd.load_baseline(self.zedd_baseline_path)
        else:
            self.zedd = build_enriched_baseline(model=self._shared_model, model_name=model_name)
            self.zedd.save_baseline(self.zedd_baseline_path)

        # Layer 4 — RAGMemory auto-populates its centroids from any
        # existing ChromaDB data at construction time on its own; nothing
        # extra needed here (see rag_memory.py's __init__).
        self.rag = RAGMemory(chromadb_path=chromadb_path, model=self._shared_model, model_name=model_name)

        # Layer 5
        self.gate = HumanLoopGate()

        # Lazy-created on first multimodal sample — see _get_multimodal_extractor()
        self._multimodal_extractor = None

    @staticmethod
    def _load_shared_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers`."
            ) from exc
        return SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(self, input_text: str, sensor_readings: list, n_window: int = 10) -> PipelineResult:
        start = time.perf_counter()

        # --- Layer 1: input filtering ---
        filter_result = self.filter.screen_input(input_text)
        if filter_result.blocked:
            self._store_blocked_attack(input_text, filter_reason=filter_result.reason, source_layer="filtering")
            return self._finish(
                start, blocked=True, blocked_by="filtering",
                filter_result=filter_result, zedd_result=None, rag_result=None,
                assessment=None, human_loop_result=None, tokens_added=0,
            )

        # --- Layer 2: defensive tokens ---
        # layer1_flagged is always False here by construction: Layer 1
        # already returned above if it had flagged anything, so this call
        # never reaches strict mode via that argument within this pipeline.
        secured_input = self.tokens.apply_adaptive(input_text, layer1_flagged=False)
        token_stats = self.tokens.analyze_application(input_text, secured_input)
        tokens_added = token_stats["tokens_added"]

        # --- Layer 3: ZEDD, on the ORIGINAL input, not the token-wrapped version ---
        zedd_result = self.zedd.detect(input_text)
        if zedd_result.flagged:
            self._store_blocked_attack(input_text, filter_reason=None, source_layer="zedd")
            return self._finish(
                start, blocked=True, blocked_by="zedd",
                filter_result=filter_result, zedd_result=zedd_result, rag_result=None,
                assessment=None, human_loop_result=None, tokens_added=tokens_added,
            )

        # --- Layer 4: RAG memory, on the ORIGINAL input ---
        rag_result = self.rag.check(input_text)
        if rag_result.flagged:
            # No store here — a match means this is ALREADY in memory;
            # storing again would just create a near-duplicate embedding.
            return self._finish(
                start, blocked=True, blocked_by="rag_memory",
                filter_result=filter_result, zedd_result=zedd_result, rag_result=rag_result,
                assessment=None, human_loop_result=None, tokens_added=tokens_added,
            )

        # --- Agent call ---
        rolling_window = get_rolling_window(sensor_readings, n=n_window)
        assessment = analyze_readings(rolling_window, user_context=secured_input)

        # --- Layer 1, output side ---
        output_filter_result = self.filter.screen_output(assessment.recommendation)
        if output_filter_result.blocked:
            return self._finish(
                start, blocked=True, blocked_by="filtering_output",
                filter_result=output_filter_result, zedd_result=zedd_result, rag_result=rag_result,
                assessment=assessment, human_loop_result=None, tokens_added=tokens_added,
            )

        # --- Layer 5: human-in-the-loop gate ---
        human_loop_result = self.gate.evaluate(assessment.tool_call, assessment)

        return self._finish(
            start, blocked=False, blocked_by=None,
            filter_result=output_filter_result, zedd_result=zedd_result, rag_result=rag_result,
            assessment=assessment, human_loop_result=human_loop_result, tokens_added=tokens_added,
        )

    def _finish(self, start_time: float, **fields) -> PipelineResult:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return PipelineResult(pipeline_duration_ms=elapsed_ms, **fields)

    def _store_blocked_attack(
        self, text: str, filter_reason: Optional[str], source_layer: str
    ) -> None:
        """
        Store a confirmed-blocked attack in RAG memory. See module
        docstring's "WHY A CATEGORY-CLASSIFICATION FALLBACK..." section
        for how the category is chosen when filter_reason doesn't already
        supply one (i.e. when ZEDD, not filtering, made the block decision).
        """
        category = None
        if filter_reason:
            # filtering.py's reason format is "<category>: <detail>" —
            # the category prefix IS a valid ATTACK_CATEGORIES value when
            # filtering itself is what blocked this.
            candidate = filter_reason.split(":")[0].strip()
            if candidate in ATTACK_CATEGORIES:
                category = candidate

        if category is None:
            category = self.filter.classify_attack_category(text)

        if category is None:
            category = _UNCLASSIFIED_ATTACK_FALLBACK_CATEGORY

        try:
            self.rag.store_attack(text=text, attack_category=category, source_layer=source_layer)
        except Exception:
            # Storage is best-effort here — a storage failure must not
            # invalidate a blocking decision that has already been made
            # correctly and is about to be returned to the caller.
            pass

    # ------------------------------------------------------------------
    # Batch / evaluation helpers
    # ------------------------------------------------------------------

    def run_attack_sample(self, sample, sensor_readings: list, n_window: int = 10) -> PipelineResult:
        """
        Convenience wrapper around run() using sample.wrapped_attack as
        the input text. Storage into RAG memory on a filtering/ZEDD block
        already happens INSIDE run() (see _store_blocked_attack above) —
        this method deliberately does NOT add a second store_attack()
        call, which would otherwise create a duplicate embedding every
        single time the same sample is evaluated.

        Multimodal samples: if sample.image_path is set, the input text
        actually run through the pipeline is EXTRACTED from that image
        via MultimodalExtractor, not sample.wrapped_attack directly — the
        image is the real attack delivery mechanism, and every
        downstream layer should see what the agent would actually
        receive. If extraction fails (returns ""), falls back to
        sample.wrapped_attack with a logged warning, so one failed
        extraction doesn't silently drop a sample from evaluation.
        """
        input_text = sample.wrapped_attack
        image_path = getattr(sample, "image_path", None)
        if image_path:
            extractor = self._get_multimodal_extractor()
            extracted = extractor.extract(image_path)
            if extracted:
                input_text = extracted
            else:
                logger.warning(
                    "Multimodal extraction returned empty text for %s (image_path=%s) "
                    "— falling back to sample.wrapped_attack.",
                    getattr(sample, "attack_id", "<unknown>"), image_path,
                )
        return self.run(input_text, sensor_readings, n_window=n_window)

    def _get_multimodal_extractor(self):
        """Lazy, cached — avoids constructing an Anthropic client for
        every sample when most samples in a dataset aren't multimodal."""
        if self._multimodal_extractor is None:
            from multimodal_extractor import MultimodalExtractor
            self._multimodal_extractor = MultimodalExtractor()
        return self._multimodal_extractor

    def run_evaluation(self, dataset: list, sensor_readings: list, n_window: int = 10) -> list:
        """Run every sample in dataset through the pipeline, in order, and
        return all results. dataset may freely mix attack and benign
        AttackSample entries — both flow through run_attack_sample() identically."""
        return [
            self.run_attack_sample(sample, sensor_readings, n_window=n_window)
            for sample in dataset
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    scenarios = build_scenarios()

    parser = argparse.ArgumentParser(
        description="Run a single input through the full defense pipeline against a sensor scenario."
    )
    parser.add_argument("--scenario", choices=sorted(scenarios.keys()), default="normal")
    parser.add_argument("--attack", type=str, required=True, help="Input text to run through the pipeline.")
    parser.add_argument("--duration-min", type=float, default=60.0, help="Simulated sensor scenario duration.")
    parser.add_argument("--n-window", type=int, default=10, help="Rolling window size passed to the agent.")
    parser.add_argument("--seed", type=int, default=42, help="Sensor generation seed, for reproducibility.")
    args = parser.parse_args()

    scenario = scenarios[args.scenario]
    sensor_readings = generate_readings(
        scenario, duration_min=args.duration_min, interval_min=5.0, seed=args.seed
    )

    pipeline = Pipeline()
    result = pipeline.run(args.attack, sensor_readings, n_window=args.n_window)

    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    _cli()