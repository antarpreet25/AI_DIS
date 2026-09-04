"""
zedd_corpus.py

Template-based, seed-disjoint data generation for ZEDD, reusing the
EXISTING templated generators (DocumentFactory, AttackTemplateLibrary,
AttackInserter from attack_content.py / attack_pipeline.py) — no
hand-written one-off samples.

WHAT THIS PRODUCES (all disjoint from the evaluation set)
--------------------------------------------------------
1. A BASELINE CORPUS of clean documents across every document register,
   BALANCED so each of ZEDD's four content categories gets an equal
   document budget (see BALANCED_CORPUS_COUNTS and the note below on why
   uniform per-document-type counts were imbalanced).

2. CONTRASTIVE TRAINING PAIRS at two granularities:
     - document-level : whole injected doc vs whole clean doc
     - sentence-level : one isolated injected line vs one clean line
   Coverage-driven: every one of the AttackTemplateLibrary's ~180
   templates is used at least `min_uses_per_template` times, each in a
   different document context / vocabulary draw, so scaling adds
   QUALITATIVE diversity rather than near-duplicates the model can
   memorise.

3. A CALIBRATION DATASET (attacks + benign) on its OWN seed, used to fit
   the GMM thresholds — so thresholds are never tuned on the evaluation
   set (the data-leakage fix).

THREE DISJOINT SEEDS
--------------------
    evaluation set        : dataset_seed = 42     (attack_generator.py)
    training corpus/pairs  : _TRAIN_SEED  = 90210  (this file)
    calibration dataset    : _CALIB_SEED  = 70123  (this file)

Nothing generated here shares a seed with the evaluation set, so
train / calibrate / test are a clean three-way split.

WHY UNIFORM PER-DOCUMENT-TYPE COUNTS WERE IMBALANCED
---------------------------------------------------
DocumentFactory has 9 document types; ZEDD has 4 content categories, and
the mapping is not 1:1 — four document types (maintenance_report,
maintenance_schedule, technician_note, inspection_report) all belong to
"maintenance_reports", while only one (scada_alert) belongs to
"system_alerts". Generating N documents per *document type* therefore
gave "maintenance_reports" 4N documents and "system_alerts" only N.
BALANCED_CORPUS_COUNTS instead assigns a per-document-type count so each
*category* lands at the same document total.

SCALING
-------
The functions take explicit size arguments; the module-level DEFAULT_*
constants are the only numbers to change to scale everything at once
(toward, e.g., a 25k-sample evaluation set). Bigger baseline corpus =
better centroids (diminishing returns past a few thousand docs); bigger
pair count helps fine-tuning up to a few thousand then saturates — the
lever that matters is TEMPLATE COVERAGE, not raw count.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

_TRAIN_SEED = 90210          # disjoint from attack_generator's dataset_seed=42
_CALIB_SEED = 70123          # disjoint from BOTH the eval set and the training corpus
_PAIR_SAMPLING_SEED = 90211  # separate stream for which-clean-partner-to-pair choices

# Every DocumentFactory document type -> the ZEDD content category it
# belongs to (defense/zedd.py KNOWN_CATEGORIES).
DOC_TYPE_TO_CATEGORY = {
    "sensor_log":              "sensor_readings",
    "dga_report":              "sensor_readings",
    "maintenance_report":      "maintenance_reports",
    "maintenance_schedule":    "maintenance_reports",
    "technician_note":         "maintenance_reports",
    "inspection_report":       "maintenance_reports",
    "operator_handover":       "operator_notes",
    "supplier_communication":  "operator_notes",
    "scada_alert":             "system_alerts",
}

_CATEGORY_TO_DOC_TYPES: dict = {}
for _dt, _cat in DOC_TYPE_TO_CATEGORY.items():
    _CATEGORY_TO_DOC_TYPES.setdefault(_cat, []).append(_dt)

ALL_DOC_TYPES = list(DOC_TYPE_TO_CATEGORY.keys())

# Baseline corpus: aim for ~this many documents PER ZEDD CATEGORY, split
# evenly across the document types feeding that category.
DEFAULT_DOCS_PER_CATEGORY = 460          # 4 categories -> ~1840 documents
# Training pairs: how many times each of the ~180 attack templates is
# instantiated (each in a different document / vocabulary draw).
DEFAULT_MIN_USES_PER_TEMPLATE = 4        # ~180 templates -> ~720 doc x template instantiations
# Calibration dataset size (per class).
DEFAULT_CALIB_ATTACKS = 500
DEFAULT_CALIB_BENIGN = 500

_MIN_SEGMENT_CHARS = 12


def balanced_corpus_counts(docs_per_category: int = DEFAULT_DOCS_PER_CATEGORY) -> dict:
    """Per-document-type counts such that every ZEDD category receives
    ~`docs_per_category` documents in total."""
    counts: dict = {}
    for category, doc_types in _CATEGORY_TO_DOC_TYPES.items():
        per_type = max(1, round(docs_per_category / len(doc_types)))
        for dt in doc_types:
            counts[dt] = per_type
    return counts


BALANCED_CORPUS_COUNTS = balanced_corpus_counts()


def _split_lines(text: str, min_chars: int = _MIN_SEGMENT_CHARS) -> list:
    """Line-level split (documents in this project put one field/sentence
    per line), dropping very short fragments."""
    return [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_chars]


def _factory(seed: int):
    from attack_content import DocumentFactory
    return DocumentFactory(rng=random.Random(seed))


# ---------------------------------------------------------------------------
# 1. Baseline corpus
# ---------------------------------------------------------------------------

def generate_disjoint_corpus(seed: int = _TRAIN_SEED, counts: Optional[dict] = None) -> list:
    """CleanDocuments across all 9 registers, balanced per ZEDD category,
    with a seed disjoint from the evaluation dataset's seed (42)."""
    factory = _factory(seed)
    return factory.generate_batch(counts or BALANCED_CORPUS_COUNTS)


def build_register_baseline_texts(corpus: list) -> dict:
    """
    category -> list[str], mixing whole-document text (keeps document-mode
    centroids representative of full documents) with line-level segments
    (gives sentence-mode centroids real coverage of what legitimate
    fragments in each register look like — the coverage gap that caused
    sentence mode's false positives).
    """
    out = {cat: [] for cat in _CATEGORY_TO_DOC_TYPES}
    for doc in corpus:
        cat = DOC_TYPE_TO_CATEGORY.get(doc.document_type)
        if cat is None:
            continue
        out[cat].append(doc.text)
        out[cat].extend(_split_lines(doc.text))
    return out


# ---------------------------------------------------------------------------
# 2. Contrastive training pairs (document- and sentence-level)
# ---------------------------------------------------------------------------

def _instantiations(seed: int, min_uses_per_template: int):
    """
    Yield (clean_doc, wrapped_doc, template, doc_type) tuples such that
    EVERY attack template is used at least `min_uses_per_template` times,
    each time in a freshly generated document (rotating through all 9
    document types) so there are no near-duplicate contexts.
    """
    from attack_content import AttackTemplateLibrary
    from attack_pipeline import AttackInserter

    rng = random.Random(seed)
    factory = _factory(seed)
    inserter = AttackInserter(rng=rng)
    templates = AttackTemplateLibrary().all()

    plan = templates * min_uses_per_template
    rng.shuffle(plan)

    for i, template in enumerate(plan):
        doc_type = ALL_DOC_TYPES[i % len(ALL_DOC_TYPES)]
        doc = getattr(factory, f"generate_{doc_type}")()
        wrapped, clean = inserter.insert(doc, template)
        yield clean, wrapped, template, doc_type


def generate_training_pairs(
    seed: int = _TRAIN_SEED,
    min_uses_per_template: int = DEFAULT_MIN_USES_PER_TEMPLATE,
    pair_sampling_seed: int = _PAIR_SAMPLING_SEED,
) -> list:
    """
    Build BOTH document-level and sentence-level contrastive pairs from
    the same disjoint instantiations (schema matches
    attack_pipeline.TrainingPairBuilder: {"text1","text2","label",
    "doc_type","granularity"}).

      label 0 (dissimilar):
        - document : (wrapped doc, clean doc)
        - sentence : (injected line, a clean line from the SAME document)
      label 1 (similar):
        - sentence : (clean line, another clean line from the SAME document)

    AttackInserter guarantees wrapped and clean differ by EXACTLY the
    injected line, so the injected line is isolated by an exact line-set
    diff (no fuzzy matching).
    """
    pair_rng = random.Random(pair_sampling_seed)
    pairs: list = []

    for clean, wrapped, template, doc_type in _instantiations(seed, min_uses_per_template):
        # --- document-level dissimilar pair ---
        pairs.append({
            "text1": wrapped, "text2": clean, "label": 0,
            "doc_type": doc_type, "granularity": "document",
        })

        # --- sentence-level pairs ---
        clean_line_set = set(clean.split("\n"))
        injected_lines = [
            ln.strip() for ln in wrapped.split("\n")
            if ln not in clean_line_set and len(ln.strip()) >= _MIN_SEGMENT_CHARS
        ]
        clean_segs = _split_lines(clean)
        if not injected_lines or len(clean_segs) < 2:
            continue

        for inj in injected_lines:
            for partner in pair_rng.sample(clean_segs, k=min(2, len(clean_segs))):
                pairs.append({
                    "text1": inj, "text2": partner, "label": 0,
                    "doc_type": doc_type, "granularity": "sentence",
                })
        # clean-clean "similar" pairs — two per instantiation, to keep the
        # label 0 : label 1 ratio near 2:1 (ContrastiveLoss trains best
        # when it sees a healthy share of positive pairs, not almost all
        # negatives).
        n_pos = min(2, len(clean_segs) // 2)
        for _ in range(n_pos):
            a, b = pair_rng.sample(clean_segs, k=2)
            pairs.append({
                "text1": a, "text2": b, "label": 1,
                "doc_type": doc_type, "granularity": "sentence",
            })

    pair_rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# 3. Calibration dataset (its own seed — the data-leakage fix)
# ---------------------------------------------------------------------------

def generate_calibration_dataset(
    seed: int = _CALIB_SEED,
    n_attacks: int = DEFAULT_CALIB_ATTACKS,
    n_benign: int = DEFAULT_CALIB_BENIGN,
) -> dict:
    """
    {"attacks": [wrapped_text, ...], "benign": [clean_text, ...]} generated
    on a seed disjoint from BOTH the evaluation set and the training
    corpus. GMM thresholds are fitted on THIS, never on the evaluation
    set, so calibration and test do not overlap.

    Attacks rotate through every document type and every attack template
    (round-robin), so the calibration distribution is as diverse as the
    training and evaluation distributions.
    """
    from attack_content import AttackTemplateLibrary
    from attack_pipeline import AttackInserter

    rng = random.Random(seed)
    factory = _factory(seed)
    inserter = AttackInserter(rng=rng)
    templates = AttackTemplateLibrary().all()

    attacks: list = []
    for i in range(n_attacks):
        doc_type = ALL_DOC_TYPES[i % len(ALL_DOC_TYPES)]
        template = templates[i % len(templates)]
        doc = getattr(factory, f"generate_{doc_type}")()
        wrapped, _clean = inserter.insert(doc, template)
        attacks.append(wrapped)

    benign: list = []
    for i in range(n_benign):
        doc_type = ALL_DOC_TYPES[i % len(ALL_DOC_TYPES)]
        benign.append(getattr(factory, f"generate_{doc_type}")().text)

    return {"attacks": attacks, "benign": benign}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_json(obj, path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def summarize_pairs(pairs: list) -> str:
    import collections
    by_gran = collections.Counter(p.get("granularity", "?") for p in pairs)
    by_label = collections.Counter(p["label"] for p in pairs)
    by_doc = collections.Counter(p.get("doc_type", "?") for p in pairs)
    return (
        f"{len(pairs)} pairs | granularity={dict(by_gran)} | "
        f"label={{0:{by_label.get(0,0)}, 1:{by_label.get(1,0)}}} | "
        f"doc_types={dict(by_doc)}"
    )


# ---------------------------------------------------------------------------
# Back-compat shims for the v2 pipeline (run_v2 / build_document_register_baseline).
# New code should use generate_training_pairs / generate_calibration_dataset /
# save_json and the _TRAIN_SEED / _CALIB_SEED names.
# ---------------------------------------------------------------------------

_CORPUS_SEED = _TRAIN_SEED
SENTENCE_PAIR_N_DOCS = 260  # unused by new code; referenced by old run_v2 logs


def generate_sentence_pairs(seed: int = _TRAIN_SEED, n_docs: int = None,
                            pair_sampling_seed: int = _PAIR_SAMPLING_SEED) -> list:
    """Deprecated: sentence-only slice of generate_training_pairs()."""
    return [
        p for p in generate_training_pairs(seed=seed, pair_sampling_seed=pair_sampling_seed)
        if p.get("granularity") == "sentence"
    ]


def save_pairs(pairs: list, path) -> None:
    """Deprecated alias for save_json()."""
    save_json(pairs, path)
    import collections
    n0 = sum(1 for p in pairs if p["label"] in (0, 0.0))
    n1 = sum(1 for p in pairs if p["label"] in (1, 1.0))
    print(f"[zedd_corpus] wrote {len(pairs)} pairs ({n0} dissimilar, {n1} similar) -> {path}")
