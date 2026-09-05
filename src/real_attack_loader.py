"""
real_attack_loader.py

Loads REAL, externally-authored prompt-injection examples from public
datasets and converts them into AttackTemplate-compatible objects, so
they flow through the EXISTING generation pipeline (AttackInserter +
ExtendedDocumentFactory + the dedup pass in dataset_scale_v2.py)
unchanged. This is deliberately a small, additive module — no change to
main.py, zedd.py, rag_memory.py, or the insertion mechanism itself; only
the SOURCE of raw_phrase content changes.

WHY THESE TWO SOURCES, AND WHY NOT OTHERS TRIED
-------------------------------------------------
- deepset/prompt-injections (HF, open, 203 positive examples): a small,
  well-known, cited benchmark of genuinely realistic attacker phrasing
  ("Forget all previous tasks...", "Ignore the above instructions...").
  Used verbatim, no filtering needed — every positive example reads as a
  real prompt injection attempt.
- jayavibhav/prompt-injection (HF, open, 129k positive examples): huge,
  but inspection showed most of it is noisy/augmented — typo-corrupted
  strings ("ovverirde prioirty taks") and flowery narrative jailbreak
  prose ("whispering waters of the world's rivers carry secrets...").
  Splicing that into a SCADA alert would look MORE synthetic, not less.
  Filtered here for a QUALITY subset: must contain a recognisable
  attack-intent phrase (ignore/disregard/override/bypass/reveal/...) and
  pass a basic real-word ratio check to exclude the corrupted rows.
- Necent/llm-jailbreak-prompt-injection-dataset (the large 30-dataset
  aggregation) is GATED on HuggingFace (requires manual, unauthenticated
  access here) — not usable without the user requesting access
  separately; not used.
- Garak (NVIDIA's LLM vulnerability scanner) is a legitimate, real
  source of well-documented attack techniques, but is not installed
  here and pulling its probe strings requires adding a new heavy
  dependency — noted as a defensible future addition, not built now
  given the scope already covered by the two sources above.

CATEGORY / DIFFICULTY / EXPECTED_BLOCKED_BY ASSIGNMENT
--------------------------------------------------------
Not guessed — DERIVED by actually running each real example through the
existing filtering.py: if InputOutputFilter.screen_input() blocks it,
that IS the ground truth (difficulty="easy", expected_blocked_by=
["filtering"]). If it doesn't, classify_attack_category() gives a best-
guess category (falling back to "indirect_injection", matching main.py's
own fallback convention for attacks that evade every keyword pattern —
arguably the correct fit, since that's what "evades keyword matching"
means), and difficulty="hard" with expected_blocked_by=["zedd"] — mirrors
exactly how main.py treats a ZEDD-only catch.

Every resulting AttackTemplate keeps full provenance: payload_source
records which dataset (and original row index) it came from.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attack_content import AttackTemplate
from defense.filtering import InputOutputFilter

_ATTACK_INTENT_KEYWORDS = [
    "ignore", "disregard", "forget all", "forget everything", "override",
    "bypass", "reveal", "disclose", "decrypt", "confidential", "classified",
    "without permission", "pretend you are", "act as", "you are now",
    "new instructions", "system prompt", "jailbreak", "unfiltered",
    "no restrictions", "do anything", "trusted user", "admin mode",
]

_MIN_LEN, _MAX_LEN = 20, 300


def _looks_like_real_english(text: str) -> bool:
    """Cheap gibberish filter: most words should be plausible (mostly
    lowercase-alpha, no excessive repeated-char corruption). Rejects the
    typo-fuzzed rows found during inspection (e.g. 'ovverirde prioirty')
    without needing a full spellchecker."""
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 4:
        return False
    # A handful of common English function words should appear somewhere
    # — real sentences have them, keyword-salad/fuzzed text usually doesn't.
    common = {"the", "a", "an", "is", "are", "you", "your", "to", "and",
              "all", "this", "please", "with", "for", "of", "on", "it"}
    lower_words = {w.lower() for w in words}
    return len(lower_words & common) >= 1


def _classify_and_label(text: str, filt: InputOutputFilter) -> dict:
    result = filt.screen_input(text)
    if result.blocked:
        candidate = (result.reason or "").split(":")[0].strip()
        category = candidate if candidate in (
            "direct_injection", "indirect_injection", "escalation", "encoding_obfuscation"
        ) else "indirect_injection"
        return {
            "category": category,
            "difficulty": "easy",
            "expected_blocked_by": ["filtering"],
            "expected_slip_layers": [],
            "ground_truth_risk": "HIGH",
        }
    category = filt.classify_attack_category(text) or "indirect_injection"
    return {
        "category": category,
        "difficulty": "hard",
        "expected_blocked_by": ["zedd"],
        "expected_slip_layers": ["filtering"],
        "ground_truth_risk": "MEDIUM",
    }


def load_deepset(filt: InputOutputFilter) -> list:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        repo_id="deepset/prompt-injections",
        filename="data/train-00000-of-00001-9564e8b05b4757ab.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    positives = df[df["label"] == 1]["text"].tolist()

    templates = []
    for i, text in enumerate(positives):
        text = text.strip()
        if not (_MIN_LEN <= len(text) <= 2000):
            continue
        labels = _classify_and_label(text, filt)
        templates.append(AttackTemplate(
            technique=f"real_deepset_{i:04d}",
            category=labels["category"],
            raw_phrase=text,
            difficulty=labels["difficulty"],
            expected_blocked_by=labels["expected_blocked_by"],
            expected_slip_layers=labels["expected_slip_layers"],
            ground_truth_risk=labels["ground_truth_risk"],
            insertion_hint="body",
        ))
    print(f"[real_attack_loader] deepset/prompt-injections: {len(templates)} usable "
          f"(of {len(positives)} positive rows)")
    return [(t, "real_deepset") for t in templates]


def load_jayavibhav(filt: InputOutputFilter, max_rows: int = 5000) -> list:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        repo_id="jayavibhav/prompt-injection",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(path)
    positives = df[df["label"] == 1]["text"]

    kept = []
    for i, text in positives.items():
        text = str(text).strip()
        if not (_MIN_LEN <= len(text) <= _MAX_LEN):
            continue
        lower = text.lower()
        if not any(kw in lower for kw in _ATTACK_INTENT_KEYWORDS):
            continue
        if not _looks_like_real_english(text):
            continue
        kept.append((i, text))
        if len(kept) >= max_rows:
            break

    templates = []
    for i, text in kept:
        labels = _classify_and_label(text, filt)
        templates.append(AttackTemplate(
            technique=f"real_jayavibhav_{i}",
            category=labels["category"],
            raw_phrase=text,
            difficulty=labels["difficulty"],
            expected_blocked_by=labels["expected_blocked_by"],
            expected_slip_layers=labels["expected_slip_layers"],
            ground_truth_risk=labels["ground_truth_risk"],
            insertion_hint="body",
        ))
    print(f"[real_attack_loader] jayavibhav/prompt-injection: {len(templates)} usable "
          f"after quality filter (scanned toward {max_rows} cap, "
          f"{len(positives)} total positive rows available)")
    return [(t, "real_jayavibhav") for t in templates]


def load_all_real_templates(max_jayavibhav: int = 5000) -> dict:
    """Returns {category: [(AttackTemplate, payload_source), ...]}."""
    import tempfile
    # Classifying thousands of real examples through screen_input() would
    # otherwise write thousands of entries into the PRODUCTION filter log
    # (data/logs/filter_log.jsonl) — this is dataset-construction-time
    # classification, not real pipeline traffic, so it gets its own
    # throwaway log path instead.
    scratch_log = Path(tempfile.gettempdir()) / "real_attack_loader_filter_log.jsonl"
    filt = InputOutputFilter(log_path=scratch_log)
    all_templates = load_deepset(filt) + load_jayavibhav(filt, max_rows=max_jayavibhav)

    by_category: dict = {}
    for template, source in all_templates:
        by_category.setdefault(template.category, []).append((template, source))

    print("[real_attack_loader] real templates by category: " +
          ", ".join(f"{c}={len(v)}" for c, v in by_category.items()))
    return by_category


if __name__ == "__main__":
    by_cat = load_all_real_templates()
    for cat, items in by_cat.items():
        print(f"\n--- {cat} ({len(items)}) ---")
        for t, src in items[:3]:
            print(f"  [{src}] {t.raw_phrase[:100]!r}")
