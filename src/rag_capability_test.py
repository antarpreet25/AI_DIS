"""
rag_capability_test.py

Isolates RAG memory's actual incremental contribution: catching a
PARAPHRASE of an attack already blocked once, that filtering and ZEDD
do NOT independently catch on their own.

WHY THIS IS A SEPARATE TEST, NOT PART OF THE MAIN EVAL SET
-------------------------------------------------------------
The main scaled dataset (dataset_scale_v2.py) explicitly DISCARDS
anything with cosine similarity >=0.85 to an already-accepted sample in
its category — that's the dedup pass that keeps the dataset from being
padded with near-duplicates. But RAG's entire purpose is catching
NEAR-DUPLICATES of attacks already seen. Testing RAG on a
dedup-maximized dataset was always going to show zero contribution —
not because RAG doesn't work, but because the main dataset was built to
exclude exactly the case RAG is for. This script builds the
complementary, deliberately-paraphrase-heavy test the main set cannot.

METHOD
------
For N seed attacks (drawn from the v2 dataset):
  1. Generate a PARAPHRASE of the seed's payload:
       - compositional seeds: resample the SAME slot pattern with
         different slot values (a natural, controlled paraphrase of the
         same underlying technique)
       - fixed/real seeds: a light synonym-substitution rewrite
  2. Simulate the real pipeline order: feed the SEED through filtering
     -> ZEDD; if blocked, store its payload in RAG memory (exactly as
     main.py's _store_blocked_attack does).
  3. Feed the PARAPHRASE through filtering -> ZEDD -> RAG (fresh, not
     used to build memory) and record which layer(s) would have caught
     it.
  4. Attribute the catch to RAG specifically only when filtering AND
     ZEDD both pass the paraphrase but RAG flags it — i.e. only when RAG
     is the reason the paraphrase doesn't reach the agent, not just any
     case where RAG happens to also fire.

No API calls, no cost — filtering and RAG run locally; ZEDD needs the
fine-tuned model loaded once.
"""

from __future__ import annotations

import json
import random
import re
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attack_content_v2 import CompositionalAttackFactory, _build_compositional_patterns
from defense.filtering import InputOutputFilter
from defense.zedd import ZEDDDetector
from defense.rag_memory import RAGMemory, ATTACK_CATEGORIES

N_SEEDS_PER_CATEGORY = 15
SEED = 424242

_SYNONYM_MAP = {
    "ignore": "disregard", "disregard": "set aside", "previous": "prior",
    "instructions": "directives", "please": "kindly", "immediately": "right away",
    "override": "bypass", "bypass": "circumvent", "approve": "authorise",
    "all": "every", "confirm": "verify", "report": "log", "forget": "set aside",
    "now": "at this time", "stop": "halt", "continue": "proceed",
}


def _light_paraphrase(text: str, rng: random.Random) -> str:
    """Cheap, deterministic-ish paraphrase for fixed/real seeds: swap a
    handful of common words for synonyms and reorder if there's an
    'and' clause. Not meant to be a good paraphrase generator — just
    enough surface variation to be a genuine near-duplicate, not a
    verbatim copy, matching what a real attacker's light reword would
    produce."""
    words = text.split()
    out = []
    for w in words:
        bare = re.sub(r"[^A-Za-z]", "", w).lower()
        if bare in _SYNONYM_MAP and rng.random() < 0.7:
            repl = _SYNONYM_MAP[bare]
            out.append(w.replace(bare, repl) if bare in w.lower() else repl)
        else:
            out.append(w)
    return " ".join(out)


def build_pairs(dataset_path: Path, rng: random.Random) -> list:
    """Returns [(category, seed_payload, paraphrase_payload, seed_kind), ...]"""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]
    attacks = [s for s in data if s["category"] != "benign"]

    patterns_by_cat = {}
    for p in _build_compositional_patterns():
        patterns_by_cat.setdefault(p.category, []).append(p)

    pairs = []
    for cat in ATTACK_CATEGORIES:
        cat_samples = [s for s in attacks if s["category"] == cat]
        rng.shuffle(cat_samples)
        taken = 0
        for s in cat_samples:
            if taken >= N_SEEDS_PER_CATEGORY:
                break
            seed_payload = (s.get("raw_attack") or "").strip()
            if not seed_payload:
                continue
            if s.get("template_kind") == "compositional" and s.get("slot_values"):
                # Find the pattern by technique prefix and resample slots.
                prefix = "_".join(s["attack_type"].split("_")[:-1])
                matches = [p for p in patterns_by_cat.get(cat, [])
                           if s["attack_type"].startswith(p.technique_prefix)]
                if matches:
                    pattern = matches[0]
                    new_phrase, _ = pattern.sample(rng)
                    if new_phrase != seed_payload:
                        pairs.append((cat, seed_payload, new_phrase, "compositional"))
                        taken += 1
                        continue
            # fixed / real seeds -> light paraphrase
            paraphrase = _light_paraphrase(seed_payload, rng)
            if paraphrase != seed_payload:
                pairs.append((cat, seed_payload, paraphrase, s.get("template_kind", "fixed")))
                taken += 1
    return pairs


def main():
    rng = random.Random(SEED)
    v2_path = _ROOT / "data" / "attacks" / "attack_dataset_scaled_v2.json"
    pairs = build_pairs(v2_path, rng)
    print(f"[rag_capability_test] built {len(pairs)} seed/paraphrase pairs "
          f"({N_SEEDS_PER_CATEGORY} target per category)")

    filt = InputOutputFilter(log_path=Path(tempfile.gettempdir()) / "rag_test_filter_log.jsonl")

    print("[rag_capability_test] loading ZEDD (fine-tuned v3, sentence mode — the "
          "production-recommended configuration)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(_ROOT / "data" / "zedd_finetuned_model"))
    zedd = ZEDDDetector(model=model, sentence_level=True)
    zedd.load_baseline(_ROOT / "data" / "zedd_finetuned_baseline.json")
    cal = json.loads((_ROOT / "data" / "zedd_calibration.json").read_text(encoding="utf-8"))
    zedd.drift_threshold = float(cal["modes"]["sentence"]["runtime_drift_threshold"])

    tmp = Path(tempfile.mkdtemp())
    rag = RAGMemory(chromadb_path=tmp / "chromadb", model=model,
                     log_path=tmp / "rag_log.jsonl")

    results = {"rag_attributable": [], "already_caught_without_rag": [], "missed_entirely": []}
    stored = 0
    for cat, seed_payload, paraphrase, kind in pairs:
        # --- Step 1: seed goes through filtering -> ZEDD, store if blocked ---
        seed_filter = filt.screen_input(seed_payload)
        seed_blocked = seed_filter.blocked
        if not seed_blocked:
            seed_zedd = zedd.detect(seed_payload)
            seed_blocked = seed_zedd.flagged
        if seed_blocked:
            rag.store_attack(text=seed_payload, attack_category=cat,
                              source_layer="filtering" if seed_filter.blocked else "zedd",
                              payload=seed_payload)
            stored += 1

        # --- Step 2: does the PARAPHRASE get caught, and by what? ---
        para_filter = filt.screen_input(paraphrase)
        para_zedd_flagged = False
        if not para_filter.blocked:
            para_zedd_flagged = zedd.detect(paraphrase).flagged
        caught_without_rag = para_filter.blocked or para_zedd_flagged

        para_rag_flagged = False
        if not caught_without_rag:
            try:
                para_rag_flagged = rag.check(paraphrase).flagged
            except Exception:
                pass  # transient chroma hiccup on a very sparse category collection

        entry = {"category": cat, "kind": kind, "seed": seed_payload[:80], "paraphrase": paraphrase[:80]}
        if caught_without_rag:
            results["already_caught_without_rag"].append(entry)
        elif para_rag_flagged:
            results["rag_attributable"].append(entry)
        else:
            results["missed_entirely"].append(entry)

    total = len(pairs)
    print(f"\nseeds stored into RAG memory: {stored}/{total}")
    print(f"\nRESULTS (of {total} seed/paraphrase pairs):")
    print(f"  paraphrase already caught by filtering/ZEDD alone : {len(results['already_caught_without_rag'])}")
    print(f"  paraphrase caught ONLY because of RAG memory      : {len(results['rag_attributable'])}")
    print(f"  paraphrase missed by all three layers             : {len(results['missed_entirely'])}")

    print("\n--- examples: RAG-attributable catches ---")
    for e in results["rag_attributable"][:8]:
        print(f"  [{e['category']}/{e['kind']}] seed={e['seed']!r}")
        print(f"      paraphrase={e['paraphrase']!r}")

    out = _ROOT / "data" / "results" / "rag_capability_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_pairs": total, "n_stored": stored,
        "caught_without_rag": len(results["already_caught_without_rag"]),
        "rag_attributable": len(results["rag_attributable"]),
        "missed_entirely": len(results["missed_entirely"]),
        "details": results,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
