"""
dataset_scale_v2.py

Builds the scaled-v2 evaluation dataset: proportionally larger than the
original 265-sample attack_dataset_scaled.json, with enforced diversity
rather than raw duplication —

  - combines the ORIGINAL 180 fixed AttackTemplateLibrary templates with
    NEW compositional (slot-based) templates (attack_content_v2.py) so no
    single injection phrase is reused hundreds of times,
  - spreads documents across 14 registers (the original 9 + 5 new ones),
  - runs every candidate through a LOCAL, STOCK (not fine-tuned — dedup
    should be a property of the text, independent of any one detector's
    embedding space) all-MiniLM-L6-v2 embedding and discards anything
    with cosine similarity >=0.85 to an already-accepted sample in the
    same category,
  - tags every sample with full generation provenance (doc_type,
    template technique, template_kind, slot_values, nearest-neighbour
    similarity, seed).

Does NOT touch or regenerate attack_dataset_scaled.json (seed 42) — that
file must stay exactly reproducible from the untouched attack_content.py
/ attack_pipeline.py. Output: attack_dataset_scaled_v2.json, a fresh,
independent, seed-90 file. Schema-compatible with AttackSample (the new
provenance fields all have defaults), so `AttackGenerator.load()` reads
it with zero code changes.

Usage:
    python src/dataset_scale_v2.py [--attacks-per-category N] [--benign N]
                                    [--seed N] [--sim-threshold F]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from attack_content import AttackTemplateLibrary
from attack_content_v2 import ExtendedDocumentFactory, CompositionalAttackFactory, ALL_DOC_TYPES_V2
from attack_pipeline import AttackInserter
from attack_generator import AttackSample
from real_attack_loader import load_all_real_templates

DEFAULT_SEED = 90            # disjoint from the eval seed (42) and from ZEDD's own
                              # training/calibration seeds (90210 / 70123) — v2 is a
                              # different concern (a bigger EVALUATION set) with no
                              # relationship to those, but still deliberately distinct.
DEFAULT_ATTACKS_PER_CATEGORY = 750
DEFAULT_BENIGN = 1000
DEFAULT_SIM_THRESHOLD = 0.85
OVERSHOOT = 1.5               # generate 1.5x target per round before dedup
MAX_ROUNDS = 8                # give up topping up after this many rounds

CATEGORIES = ("direct_injection", "indirect_injection", "escalation", "encoding_obfuscation")


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed_many(model, texts):
    if not texts:
        return np.zeros((0, 384))
    E = np.asarray(model.encode(list(texts), batch_size=64, show_progress_bar=False), dtype=np.float64)
    norms = np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-10, None)
    return E / norms


def _greedy_dedup_select(candidates: list, embeddings: np.ndarray, threshold: float,
                          already_accepted_emb: np.ndarray, target: int) -> tuple:
    """
    Greedily walk `candidates` in order, accepting each one unless its
    cosine similarity to something already accepted (this round OR from
    a previous round, via already_accepted_emb) is >= threshold.
    Returns (accepted_indices, accepted_embeddings_matrix, max_sim_per_accepted).
    """
    accepted_idx = []
    max_sims = []
    accepted_block = [already_accepted_emb] if already_accepted_emb.shape[0] else []
    for i, emb in enumerate(embeddings):
        if len(accepted_idx) >= target:
            break
        if accepted_block:
            pool = np.vstack(accepted_block)
            sims = pool @ emb
            max_sim = float(sims.max()) if sims.size else 0.0
        else:
            max_sim = 0.0
        if max_sim < threshold:
            accepted_idx.append(i)
            max_sims.append(max_sim)
            accepted_block.append(emb.reshape(1, -1))
    final_emb = np.vstack(accepted_block) if accepted_block else np.zeros((0, embeddings.shape[1]))
    return accepted_idx, final_emb, max_sims


def build_attack_category(
    category: str, target: int, seed: int, sim_threshold: float, embedder,
    real_templates: Optional[list] = None,
) -> list:
    """Returns a list of dicts (AttackSample-shaped) for one category.

    real_templates: optional [(AttackTemplate, payload_source), ...] from
    real_attack_loader.py — mixed into the pool alongside the fixed and
    compositional (synthetic) templates, tagged with their real source so
    the final dataset can be sliced by provenance (see AttackSample.
    payload_source)."""
    rng = random.Random(seed)
    inserter = AttackInserter(rng=rng)
    factory = ExtendedDocumentFactory(rng=rng)

    fixed_templates = AttackTemplateLibrary().by_category(category)
    # One CompositionalAttackFactory instance persists across rounds so its
    # _seen_phrases set keeps growing — each round asks it for MORE fresh
    # phrasings rather than cycling a stale pre-built list (that bug meant
    # a fixed ~500-string pool was exhausted within round 1 alone, so
    # rounds 2+ only ever re-drew exact duplicates and accepted nothing).
    comp_factory = CompositionalAttackFactory(rng=random.Random(seed + 1))

    accepted: list = []
    accepted_emb = np.zeros((0, 384))
    doc_types = list(ALL_DOC_TYPES_V2)
    round_no = 0
    draw_i = 0
    # (template, kind, slot_values, payload_source)
    fixed_pool = [(t, "fixed", {}, "synthetic") for t in fixed_templates]
    real_pool = [(t, "real", {}, src) for t, src in (real_templates or [])]
    rng.shuffle(real_pool)  # real examples interleave rather than cluster together

    while len(accepted) < target and round_no < MAX_ROUNDS:
        round_no += 1
        n_needed = target - len(accepted)
        n_draw = int(n_needed * OVERSHOOT) + 5

        # Fresh compositional phrasings for THIS round (factory remembers
        # every phrase it has ever emitted, so these are new combinations,
        # not repeats of round 1's).
        fresh_comp = comp_factory.generate(category, n_draw)
        round_pool = (
            fixed_pool
            + real_pool
            + [(t, "compositional", sv, "synthetic") for t, sv in fresh_comp]
        )
        if not round_pool:
            break

        batch_meta = []
        batch_wrapped = []
        batch_payload = []
        for _ in range(n_draw):
            template, kind, slot_values, source = round_pool[draw_i % len(round_pool)]
            draw_i += 1
            doc_type = doc_types[draw_i % len(doc_types)]
            doc = getattr(factory, f"generate_{doc_type}")()
            wrapped, clean = inserter.insert(doc, template)
            batch_meta.append({
                "template": template, "kind": kind, "slot_values": slot_values,
                "doc_type": doc_type, "clean": clean, "source": source,
            })
            batch_wrapped.append(wrapped)
            batch_payload.append(template.get_injection_text())

        # Dedup on the PAYLOAD (the injected line), not the whole wrapped
        # document — the host document's boilerplate (dates, substation
        # IDs, structure) dominates a whole-document embedding, so two
        # attacks with completely different injected phrases still read
        # as near-identical at the document level (same failure mode as
        # the RAG whole-document poisoning bug found earlier). What "no
        # near-duplicate attacks" should mean is no near-duplicate
        # ATTACK SIGNATURES; the host template repeating is fine and
        # realistic (attackers reuse known document formats).
        emb = _embed_many(embedder, batch_payload)
        idx, accepted_emb, max_sims = _greedy_dedup_select(
            batch_payload, emb, sim_threshold, accepted_emb, target - len(accepted) + len(accepted)
        )
        # idx are indices into THIS batch; only take as many as still needed
        remaining_slots = target - len(accepted)
        for j, i in enumerate(idx[:remaining_slots]):
            m = batch_meta[i]
            t = m["template"]
            accepted.append({
                "attack_id": f"v2_{category}_{len(accepted)+1:04d}",
                "category": category,
                "attack_type": t.technique,
                "raw_attack": t.get_injection_text(),
                "wrapped_attack": batch_wrapped[i],
                "clean_counterpart": m["clean"],
                "expected_blocked_by": t.expected_blocked_by,
                "expected_slip_layers": t.expected_slip_layers,
                "difficulty": t.difficulty,
                "ground_truth_risk": t.ground_truth_risk,
                "image_path": None,
                "doc_type": m["doc_type"],
                "template_kind": m["kind"],
                "slot_values": m["slot_values"],
                "payload_source": m["source"],
                "nearest_neighbor_similarity": round(max_sims[j], 4),
                "generation_seed": seed,
                "dataset_version": "v2",
            })
        print(f"  [{category}] round {round_no}: drew {n_draw}, "
              f"accepted {len(idx[:remaining_slots])}, total {len(accepted)}/{target}")

    if len(accepted) < target:
        print(f"  [{category}] WARNING: only reached {len(accepted)}/{target} "
              f"after {MAX_ROUNDS} rounds (template/doc-type combinatorial "
              f"space exhausted at this dedup threshold).")
    return accepted


def build_benign(target: int, seed: int, sim_threshold: float, embedder) -> list:
    rng = random.Random(seed + 1000)
    factory = ExtendedDocumentFactory(rng=rng)
    doc_types = list(ALL_DOC_TYPES_V2)

    accepted: list = []
    accepted_emb = np.zeros((0, 384))
    round_no = 0
    draw_i = 0
    while len(accepted) < target and round_no < MAX_ROUNDS:
        round_no += 1
        n_needed = target - len(accepted)
        n_draw = int(n_needed * OVERSHOOT) + 5
        batch_docs = []
        for _ in range(n_draw):
            doc_type = doc_types[draw_i % len(doc_types)]
            draw_i += 1
            batch_docs.append((doc_type, getattr(factory, f"generate_{doc_type}")()))
        texts = [d.text for _, d in batch_docs]
        emb = _embed_many(embedder, texts)
        idx, accepted_emb, max_sims = _greedy_dedup_select(
            texts, emb, sim_threshold, accepted_emb, target - len(accepted) + len(accepted)
        )
        remaining_slots = target - len(accepted)
        for j, i in enumerate(idx[:remaining_slots]):
            doc_type, doc = batch_docs[i]
            accepted.append({
                "attack_id": f"v2_benign_{len(accepted)+1:04d}",
                "category": "benign",
                "attack_type": "benign",
                "raw_attack": "",
                "wrapped_attack": doc.text,
                "clean_counterpart": None,
                "expected_blocked_by": [],
                "expected_slip_layers": [],
                "difficulty": "n/a",
                "ground_truth_risk": "LOW",
                "image_path": None,
                "doc_type": doc_type,
                "template_kind": "fixed",
                "slot_values": {},
                "payload_source": "synthetic",
                "nearest_neighbor_similarity": round(max_sims[j], 4),
                "generation_seed": seed,
                "dataset_version": "v2",
            })
        print(f"  [benign] round {round_no}: drew {n_draw}, "
              f"accepted {len(idx[:remaining_slots])}, total {len(accepted)}/{target}")

    if len(accepted) < target:
        print(f"  [benign] WARNING: only reached {len(accepted)}/{target}.")
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks-per-category", type=int, default=DEFAULT_ATTACKS_PER_CATEGORY)
    ap.add_argument("--benign", type=int, default=DEFAULT_BENIGN)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    ap.add_argument("--out", type=str, default=str(_ROOT / "data" / "attacks" / "attack_dataset_scaled_v2.json"))
    ap.add_argument("--no-real", action="store_true",
                     help="skip real_attack_loader — synthetic-only, for comparison/debugging")
    ap.add_argument("--max-jayavibhav", type=int, default=5000)
    args = ap.parse_args()

    print(f"[dataset_scale_v2] target: {args.attacks_per_category}/category x {len(CATEGORIES)} "
          f"+ {args.benign} benign, seed={args.seed}, sim_threshold={args.sim_threshold}")
    print("[dataset_scale_v2] loading embedder (stock all-MiniLM-L6-v2, NOT the fine-tuned "
          "detector — dedup must be independent of any one detector's embedding space)...")
    embedder = _get_embedder()

    real_by_category = {}
    if not args.no_real:
        print("\n[dataset_scale_v2] loading real attack templates (deepset + jayavibhav)...")
        real_by_category = load_all_real_templates(max_jayavibhav=args.max_jayavibhav)

    t0 = time.time()
    all_samples = []
    for i, cat in enumerate(CATEGORIES):
        print(f"\n[dataset_scale_v2] building {cat}...")
        all_samples += build_attack_category(
            cat, args.attacks_per_category, args.seed + i * 7919, args.sim_threshold, embedder,
            real_templates=real_by_category.get(cat),
        )

    print(f"\n[dataset_scale_v2] building benign...")
    all_samples += build_benign(args.benign, args.seed, args.sim_threshold, embedder)

    # Validate every sample round-trips through AttackSample before saving.
    for s in all_samples:
        AttackSample(**s)

    payload = {
        "num_samples": len(all_samples),
        "samples": all_samples,
        "generation_metadata": {
            "dataset_version": "v2",
            "seed": args.seed,
            "attacks_per_category": args.attacks_per_category,
            "benign_count": args.benign,
            "sim_threshold": args.sim_threshold,
            "doc_types": list(ALL_DOC_TYPES_V2),
            "categories": list(CATEGORIES),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n[dataset_scale_v2] DONE in {elapsed:.0f}s. {len(all_samples)} samples -> {out_path}")

    import collections
    by_cat = collections.Counter(s["category"] for s in all_samples)
    by_doc = collections.Counter(s["doc_type"] for s in all_samples)
    by_kind = collections.Counter(s["template_kind"] for s in all_samples if s["category"] != "benign")
    by_source = collections.Counter(s["payload_source"] for s in all_samples if s["category"] != "benign")
    by_source_cat = collections.Counter(
        (s["category"], s["payload_source"]) for s in all_samples if s["category"] != "benign"
    )
    nn_sims = [s["nearest_neighbor_similarity"] for s in all_samples]
    print("category counts:", dict(by_cat))
    print("doc_type counts:", dict(by_doc))
    print("template_kind counts (attacks only):", dict(by_kind))
    print("payload_source counts (attacks only):", dict(by_source))
    print("payload_source by category:", dict(by_source_cat))
    print(f"nearest-neighbour similarity: mean={np.mean(nn_sims):.3f} max={np.max(nn_sims):.3f} "
          f"(threshold was {args.sim_threshold})")


if __name__ == "__main__":
    main()
