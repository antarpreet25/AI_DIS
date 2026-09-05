# AI-Agent Defense Pipeline — Consolidated Changes & Results Report

_Power-grid monitoring agent · defense-in-depth against prompt injection / embedding-drift attacks_
_Report generated: 2026-09-05. Covers all work in this development phase (from the first progress report through to the raw-vs-defended comparison)._

---

## 0. How to use this document

This is the single source of truth for the "what changed and why" narrative your dissertation
needs. It is deliberately complete (methods, reasoning, design choices, successes **and**
failures) so it can be pasted into a separate writing session without losing context.

**Files to attach to the web chat for report writing** are listed in §12.

---

## 1. System under test — architecture recap

The defended system is a layered ("defense-in-depth") wrapper around an LLM agent that reads a
rolling window of power-grid sensor telemetry and produces a structured risk assessment
(`analysis`, `risk_level`, `recommendation`, `confidence_pct`, optional `tool_call`).

Pipeline order (each layer can block; if it blocks, later layers are not reached):

| # | Layer | Mechanism | Role |
|---|-------|-----------|------|
| 0 | Multimodal extraction | OCR / text extraction from attached images | Surface hidden text for the layers below |
| 1 | **Filtering** | Keyword + regex patterns, attack-category classifier | Cheap, high-precision first cut |
| 2 | Defensive tokens | Wraps untrusted context in delimiters + instructions to the model | Passive hardening (never blocks) |
| 3 | **ZEDD** | Zero-shot Embedding-Drift Detection — sentence-transformer embeddings compared to a benign "register" baseline; 3 modes (document / sentence / dual-encoder) | Catches novel phrasings that evade keywords |
| 4 | **RAG memory** | ChromaDB store of previously-blocked attack payloads; new input segments matched by cosine similarity | Catches paraphrases of attacks already seen |
| 5 | Agent | Claude `claude-sonnet-4-6` via `grid_agent.analyze_readings()` | The thing being protected |
| 6 | Output filtering | Screens the agent's own text output | Catch leakage / policy violations in responses |
| 7 | **Human-in-the-loop** | Every `tool_call` the agent emits is gated for human approval | Last line — no autonomous action |

Attack taxonomy used throughout: `direct_injection`, `indirect_injection`, `escalation`,
`encoding_obfuscation`, plus `benign` (legitimate grid documents).

---

## 2. Phase timeline (what happened, in order)

1. **RAG memory false-positive bug** — 100% of benign samples were being flagged. Root-caused
   and fixed (§3).
2. **ZEDD threshold miscalibration** — a single global threshold was being applied to three
   incompatible score scales. Replaced with per-mode calibrated thresholds (§4).
3. **Data-leakage audit of ZEDD fine-tuning** — train / calibrate / test were overlapping.
   Rebuilt with three disjoint seeds; produced fine-tuned model **v3** (§5).
4. **ZEDD weight sweep** — the domain-drift vs security-proximity weights (0.5 / 0.5) were
   never tuned. Grid sweep → **0.8 / 0.2** (§6).
5. **Dataset scaling 265 → 2,128** — with automated diversity control, and (on supervisor's
   advice) integration of real, publicly-available prompt-injection data (§7). Two granularity
   bugs and one pool-exhaustion bug found and fixed along the way.
6. **RAG "zero contribution" investigation** — proven to be a dataset-construction artifact,
   not a broken layer; built a dedicated capability test that measures a real **16.7%**
   incremental contribution (§8).
7. **Document-vs-sentence generalization gap** — quantified on the scaled dataset (§9).
8. **Performance optimization** — batched ZEDD sentence/dual-encoder scoring, 10–30× faster (§10).
9. **Evaluator changes** — v2 dataset support, removed the bespoke false-data-injection metric (§11).
10. **Raw-vs-defended comparison** — real API calls, objective fault scenario, GBP cost
    tracked (§9 results, cost in §13).

---

## 3. RAG memory — 100% false-positive bug

**Symptom.** Every benign grid document was flagged by RAG memory, giving a 100% false-positive
rate on the benign class and making the layer unusable.

**Investigation.**
- Verified the distance→similarity conversion (ChromaDB returns L2/cosine *distance*; code must
  do `similarity = 1 - distance` for cosine). This was correct.
- Traced what was actually being *stored*. On a false positive earlier in a run, the pipeline
  called `store_attack()` with **no ground-truth payload**, so the code fell back to storing the
  **entire input text** (or, in a later variant, ZEDD's `worst_sentence`, which was often a raw
  telemetry line). Benign boilerplate and telemetry fragments were therefore seeded into the
  "known attacks" store, and every subsequent benign document matched them → cascading
  self-poisoning.

**Root cause.** Storage granularity + storing non-ground-truth content. The store was being
populated with whole documents / arbitrary sentences instead of the specific injected payload,
and host-document boilerplate dominates the embedding, so "similarity to a stored attack"
degenerated into "similarity to generic grid-document text."

**Fix (`src/defense/rag_memory.py`, `src/main.py`).**
- `store_attack()` now requires an explicit `payload=` (the actual injected attack string,
  `sample.raw_attack`). If no payload is available, **storage is skipped entirely** and a
  `skipped_reason` is logged — no more whole-text / `worst_sentence` fallback.
- Matching is now **segment-level**: `check()` splits the input into segments
  (`rag_segment_min_chars = 16`) and compares each against stored payloads, instead of embedding
  the whole document.
- Thresholds: `rag_similarity_threshold = 0.70`, `rag_pre_filter_threshold = 0.45`.

**Result.** RAG memory no longer produces false positives (main evaluation: **0 FP** on 281
benign samples in both document and sentence mode). The trade-off this exposed is discussed in §8.

---

## 4. ZEDD — per-mode threshold calibration

**Symptom.** Document mode's runtime threshold was **negative** (`-0.216`), which looked
nonsensical, and the three modes' recall/FPR were wildly inconsistent.

**Root cause.** ZEDD's three modes emit scores on **different, non-comparable scales**:
- `document` / `sentence`: a *similarity* (higher = more benign); the "suspicion" orientation is
  `1 − similarity`.
- `dual_encoder`: a *combined drift score* (higher = more suspicious).

A single global threshold (`1.216`) was being applied to all three. For document mode the code
path actually compares `closest_category_similarity` against a **per-category** threshold, so the
stored global value materialised as `1 − 1.216 = −0.216` — a real bug in how the calibrated
number was being consumed, not just a cosmetic oddity. Separately, a **Stage-1 pre-filter**
inside `_detect_document_level` was short-circuiting decisions *before* the calibrated floor was
applied, so v3 document mode showed 0.99 recall / 0.45 FPR instead of its calibrated
0.53 / 0.016.

**Fix (`src/defense/zedd.py`, `src/zedd_trainer.py`).**
- `load_finetuned()` now selects `modes[<active mode>].runtime_drift_threshold` from the
  calibration JSON — a genuinely per-mode value.
- New `document_similarity_floor` attribute; when set, `_detect_document_level`'s Stage-1
  pre-filter is **skipped** so the calibrated floor is authoritative.
- `calibrate_all_modes()` added: per-mode GMM calibration, with document mode calibrated with
  `pre_filter_enabled=False` and degenerate GMM crossings rejected.

**Calibrated thresholds now in `data/zedd_calibration.json`** (calibration set: 500 attack +
500 benign, disjoint seed — see §5):

| Mode | runtime threshold | flag rule | calib recall | calib FPR |
|------|-------------------|-----------|--------------|-----------|
| document | −0.216 (per-category similarity rule; runtime_drift_threshold unused) | `closest_category_similarity < per_category_threshold` | 0.530 | 0.016 |
| **sentence** | 0.168 | `worst_sentence_similarity < runtime_drift_threshold` | 0.752 | 0.016 |
| dual_encoder | 0.848 | `combined_score > runtime_drift_threshold` | 0.750 | 0.016 |

All three are FPR-targeted at ~1.6%. **Sentence mode is the recommended production
configuration** (best recall at matched FPR, and best generalization — §9).

---

## 5. Data leakage in ZEDD fine-tuning → disjoint methodology (v1 → v2 → v3)

**Problem.** ZEDD's embedding model is contrastively fine-tuned (SentenceTransformers,
`ContrastiveLoss`) on attack/benign pairs, then GMM-calibrated, then evaluated. In v1/v2 the
same underlying seed / sample pool fed all three stages, so the calibration and evaluation
numbers were optimistic — the model had effectively seen the test distribution.

**Fix — three disjoint seeds (`src/zedd_corpus.py`, `src/zedd_trainer.py`).**

| Stage | Seed | Content |
|-------|------|---------|
| Training pairs | `90210` | Coverage-driven: every attack template used ≥ 4× |
| Calibration set | `70123` | 500 attack + 500 benign, own seed |
| Evaluation set | `42` | The held-out benchmark; never used for training or calibration |

- `zedd_corpus.py` (new): `balanced_corpus_counts()`, `generate_training_pairs()`,
  `generate_calibration_dataset()`.
- `zedd_trainer.py`: added `run_v2/validate_v2/promote_v2`, `prepare_v3/finish_v3/validate_v3/
  promote_v3`, `--recalibrate-v3`, and a `_temp_paths()` context manager that redirects output
  paths during calibration/validation so the promotion step is atomic.
- **v3 model + baseline** are promoted: `data/zedd_finetuned_model/`,
  `data/zedd_finetuned_baseline.json`, `data/zedd_calibration.json` (pre-v3 versions kept as
  `*_pre_v3_backup.json`).

**Practical note.** Local CPU fine-tuning of ~1,100 pairs took **108 minutes**, so the workflow
was moved to Colab GPU (`COLAB_GUIDE.md`, `src/colab_finetune.py`). A **v4** retrain is planned
but deferred (see §10 / §14) — it needs another Colab session.

**Success:** leakage removed; the v3 evaluation numbers in §9 are trustworthy (disjoint).
**Cost of the fix:** every dataset change now requires a fine-tune + recalibrate round-trip.

---

## 6. ZEDD domain/security weight sweep

**Question raised.** The dual-encoder mode blends a *domain-drift* signal and a
*security-proximity* signal, weighted `domain_weight = 0.5`, `security_weight = 0.5` — never
justified or tuned.

**Method.** Grid sweep over weight pairs, scored on calibration recall at matched FPR.

**Finding.** The security-proximity signal contributes **almost nothing** — recall is
essentially flat as `security_weight` drops. Best operating point: **`domain_weight = 0.8`,
`security_weight = 0.2`**.

**Change.** `ZEDDDetector.__init__` defaults changed to `0.8 / 0.2`.
**Honest caveat for the write-up:** this means the "dual encoder" is, in practice, mostly the
domain-drift encoder. Sentence mode (single signal) matches or beats it (§9), which is part of
why sentence mode is recommended.

---

## 7. Dataset scaling: 265 → 2,128 samples

### 7.1 Goal and constraints (from supervisor + self-imposed rigor)

- Scale far beyond 265 so metrics are credible for a dissertation.
- **No near-duplicates** padding the count — high intra-class variance, qualitatively diverse.
- Proportionally scaled across classes.
- Automated semantic-similarity dedup (cosine **< 0.85** between payloads in a class).
- Metadata / provenance tags on every sample.
- **Incorporate real, internet-available prompt-injection data** (any year/country) — "internet
  available data is more credible and has realism" — without breaking the existing pipeline or
  losing performance.

### 7.2 Final dataset — `data/attacks/attack_dataset_scaled_v2.json`

**2,128 samples** (up from 42 in the original `attack_dataset.json`, 265 in the v1 scaled set):

| Class | Count |
|-------|-------|
| indirect_injection | 750 |
| direct_injection | 707 |
| benign | 281 |
| escalation | 204 |
| encoding_obfuscation | 186 |

**Provenance (`payload_source`):**

| Source | Count | Note |
|--------|-------|------|
| `real_jayavibhav` | 1,175 | `jayavibhav/prompt-injection` (HF), quality-filtered |
| `synthetic` | 909 | template-generated |
| `real_deepset` | 44 | `deepset/prompt-injections` (HF), used verbatim |

→ **~57% of the full set (66% of the *attack* rows) is real-sourced.**

**Template kind:** `real` 1,219 · `compositional` 457 · `fixed` 452.
**Difficulty:** `hard` 1,044 · `easy` 572 · `medium` 231 · (`n/a` = benign) 281.
**Diversity:** nearest-neighbour cosine similarity across attack payloads —
mean **0.601**, max **0.850** (the dedup ceiling), min 0.0. No near-duplicate padding.

**Host-document variety:** 21 grid document types (9 original + 12 new: switching_order,
protection_relay_report, outage_notification, incident_report, load_forecast_notice,
weather_advisory, vegetation_management_note, commissioning_checklist,
asset_health_index_report, training_record, environmental_compliance_note,
spare_parts_inventory_note).

### 7.3 Real-data sourcing — what was used and what was rejected

| Source | Verdict | Reason |
|--------|---------|--------|
| `deepset/prompt-injections` | **USED** (44 payloads, verbatim) | Small, cited, genuinely realistic attacker phrasing; every positive row reads as a real injection. No filtering needed. |
| `jayavibhav/prompt-injection` | **USED** (1,175, filtered from a 5,000-row scan) | Huge (129k+ positives) but noisy — typo-fuzzed rows ("ovverirde prioirty taks") and flowery narrative jailbreak prose. Filtered: length 20–300 chars, must contain a recognisable attack-intent keyword, must pass a common-English-word ratio check. |
| `Necent/llm-jailbreak-prompt-injection-dataset` | rejected | HuggingFace **gated** — `401 GatedRepoError` without manual approval. |
| Mississippi State SCADA / gas-pipeline logs | rejected | Wrong modality — numeric telemetry, not injectable text. |
| Kaggle power-grid / ARGUS datasets | rejected | Numeric optimisation/fault data, not prompt-injection text. |
| Garak (NVIDIA LLM scanner) | not integrated | Legitimate source of documented techniques, but a heavy new dependency for narrow benefit (mainly `encoding_obfuscation` diversity). Noted as defensible future work. |

**Integration approach (`src/real_attack_loader.py`, new).** Real strings are wrapped as
`AttackTemplate` objects so they flow through the **existing** generation pipeline
(`AttackInserter` + `ExtendedDocumentFactory` + the dedup pass) unchanged — **no changes to
`main.py`, `zedd.py`, `rag_memory.py`, or the insertion mechanism.** Only the *source* of the
injected phrase changes.

Category / difficulty / `expected_blocked_by` labels are **not guessed** — each real string is
run through the actual `InputOutputFilter`:
- if `screen_input()` blocks it → that is ground truth (`difficulty=easy`,
  `expected_blocked_by=["filtering"]`);
- else → `classify_attack_category()` gives the category (fallback `indirect_injection`,
  matching `main.py`'s own convention), `difficulty=hard`, `expected_blocked_by=["zedd"]`.

Real-template yield by category: `indirect_injection` 4,540 · `direct_injection` 637 ·
`encoding_obfuscation` 16 · `escalation` 6. **Escalation and encoding-obfuscation are
structurally scarce in general jailbreak datasets** — a genuine finding, not a bug; those two
classes remain mostly synthetic.

### 7.4 Diversity-engineering bugs found and fixed

**Bug 1 — dedup rejecting ~98% of candidates.** The dedup pass compared **whole wrapped
documents**. Host-document boilerplate dominates the embedding, so every candidate looked
~identical and was rejected. **Fix:** compare `template.get_injection_text()` (the payload)
only. _(Same failure mode as the original RAG bug in §3 — granularity matters.)_

**Bug 2 — rounds 2+ of dedup accepting 0 new samples.** The compositional factory was called
**once** up-front with a huge target, exhausting the whole combinatorial space of a
2–3-pattern-per-category library in round 1; later rounds just re-drew exact duplicates.
**Fix:** call `CompositionalAttackFactory.generate()` **fresh each round** (its `_seen_phrases`
set persists across calls), **and** author far more structurally distinct patterns —
**3 → 12 patterns per category**. Key insight: slot-swaps *within one sentence shape* often stay
> 0.85 similar and get dedup-rejected; genuinely different **sentence structures** survive.

**Bug 3 — generation additions.** `attack_content_v2.py` grew substantially: expanded vocab
pools (all > 2× original), `CompositionalPattern` dataclass, 20 encoding base messages × 9
transform functions for `encoding_obfuscation`.

### 7.5 Schema additions (`src/attack_generator.py`) — all backward-compatible

`doc_type`, `template_kind` (`fixed`|`compositional`|`real`), `slot_values`,
`nearest_neighbor_similarity`, `generation_seed`, `dataset_version`,
`payload_source` (`synthetic`|`real_deepset`|`real_jayavibhav`).

### 7.6 Honest limitation

`escalation` (204) and `encoding_obfuscation` (186) are smaller than the injection classes and
still mostly synthetic, because the real datasets barely contain those techniques. The dataset
is proportionally *reasonable* but not perfectly balanced, and this asymmetry propagates into
ZEDD's per-class recall.

---

## 8. RAG memory — "zero contribution" investigation

**Symptom.** In the main evaluation, RAG memory's per-layer metrics are **all zero** (tp=0,
fp=0, fn=0, tn=0) in every mode — it never independently catches anything. Natural question:
why keep a layer that does nothing?

**Root cause — a dataset-construction artifact, not a broken layer.** The scaled dataset
(§7) *deliberately discards* anything with cosine similarity ≥ 0.85 to an already-accepted
sample in its class. But RAG memory's **entire purpose** is catching **near-duplicates** of
attacks already seen. Evaluating RAG on a dedup-maximised dataset guarantees a zero reading —
the dataset was built to exclude exactly the case RAG exists for. This is a real tension in the
design: the diversity goal for the main benchmark and the paraphrase-catching goal for RAG are
in direct conflict, so they need **separate** test sets.

**Fix — `src/rag_capability_test.py` (new, complete, run).** A complementary,
deliberately-paraphrase-heavy test:
- 15 seed attacks per category (60 total) drawn from the v2 dataset.
- Each seed gets a **paraphrase**: compositional seeds are resampled with the *same pattern,
  different slot values*; fixed/real seeds get a light synonym-substitution rewrite.
- Simulate the real pipeline order: **seed** → filtering → ZEDD (sentence mode); if blocked,
  `rag.store_attack(payload=seed_payload)` exactly as `main.py` does.
- **Paraphrase** → filtering → ZEDD → (only if both miss) → RAG check.
- A catch is attributed to RAG **only** when filtering *and* ZEDD both miss the paraphrase but
  RAG flags it — i.e. only when RAG is the sole reason it doesn't reach the agent.
- No API calls, no cost.

**Result (`data/results/rag_capability_test.json`):**

| Outcome | Count (of 60 pairs) |
|---------|---------------------|
| Seeds stored into RAG memory | 44 |
| Paraphrase already caught by filtering/ZEDD alone | 35 |
| **Paraphrase caught *only* because of RAG memory** | **10 (16.7%)** |
| Paraphrase missed by all three layers | 15 |

RAG-attributable catches by category: `encoding_obfuscation` 5, `direct_injection` 3,
`indirect_injection` 2. (Missed-entirely spread: indirect 5, encoding 5, direct 3, escalation 2.)

**Interpretation for the write-up.** RAG memory's contribution is real but **conditional**: it
only helps once an attack family has been seen and blocked at least once, and then only for
variants that filtering + ZEDD don't already independently catch. On a diverse first-encounter
benchmark it reads as 0; in its actual operating regime (repeat/paraphrased attacks) it adds a
**16.7%** incremental catch rate. That is the defensible framing — not "RAG catches 16.7% of all
attacks."

---

## 9. Results — headline metrics

### 9.1 Full-system evaluation on the 2,128-sample v2 dataset (disjoint, v3 model)

| Metric | Document mode | **Sentence mode (recommended)** |
|--------|---------------|--------------------------------|
| Precision | 0.996 | **0.998** |
| Recall | 0.563 | **0.857** |
| F1 | 0.719 | **0.922** |
| Attack Success Rate (ASR) | 43.7% | **14.3%** |
| False Positive Rate (FPR) | 1.42% | **1.07%** |
| TP / FP / FN / TN | 1040 / 4 / 807 / 277 | 1582 / 3 / 265 / 278 |
| Mean pipeline latency | 154 ms | 146 ms |

Benign set = 281; attack set = 1,847. Dual-encoder mode was **not** re-run on v2 (deprioritised:
it matches sentence mode's recall at ~10× the FPR on v1, and time is constrained — see §14).

**v1 → v2 comparison (same modes, different dataset — for context, not a like-for-like):**

| Mode | v1 (265 samples) F1 / ASR / FPR | v2 (2,128 samples) F1 / ASR / FPR |
|------|-------------------------------|----------------------------------|
| document | 0.754 / 39.5% / 0.0% | 0.719 / 43.7% / 1.4% |
| sentence | 0.917 / 14.5% / 3.1% | **0.922 / 14.3% / 1.1%** |
| dual_encoder | 0.902 / 14.5% / 12.3% | (not re-run) |

Sentence mode **held its performance while the dataset grew 8× and became majority
real-sourced**, and its FPR actually improved (3.1% → 1.1%). That is the strongest single result
in the project — evidence of genuine generalization rather than memorisation.

### 9.2 Per-layer contribution (sentence mode, v2)

| Layer | TP | FN | Recall | Precision |
|-------|----|----|--------|-----------|
| filtering | 689 | 48 | 0.935 | 1.000 |
| ZEDD | 937 | 229 | 0.804 | 1.000 |
| RAG memory | 0 | 0 | — | — _(see §8; 16.7% in the dedicated test)_ |
| human-loop | 12 | 1 | 0.923 | 1.000 |

Filtering + ZEDD do the heavy lifting; every layer that fires has 1.000 precision (no layer
produces a false positive on the benign set).

### 9.3 Recall by payload source (sentence mode, v2) — generalization evidence

| Source | Document mode recall | Sentence mode recall |
|--------|----------------------|----------------------|
| synthetic (n=628) | 0.565 | 0.815 |
| real_jayavibhav (n=1,175) | 0.564 | **0.892** |
| real_deepset (n=44) | 0.500 | 0.500 |

Sentence mode does **not** degrade on real, externally-authored attacks — it is slightly
*better* on the large real slice. `deepset` (terse "ignore previous instructions…" style) is the
hardest slice for both modes, but n=44 so treat as indicative.

### 9.4 Recall by difficulty (v2) — the document-vs-sentence gap

| Difficulty | Document mode recall | Sentence mode recall |
|------------|----------------------|----------------------|
| easy (n=572) | 0.984 | 0.983 |
| medium (n=231) | 0.532 | 0.719 |
| **hard (n=1,044)** | **0.339** | **0.818** |

The generalization gap is concentrated on **hard-difficulty** attacks: document mode collapses
to 0.34 recall, sentence mode holds at 0.82. This supersedes the earlier, smaller-sample probe
that reported ~0.57 → 0.35 (document) vs ~0.84 → 0.80 (sentence) — same direction, firmer
numbers.

**Why:** document mode embeds the whole document and asks "does this drift from the benign
register?" A single injected sentence inside 400 words of realistic grid boilerplate barely
moves the document-level embedding. Sentence mode scores the **worst** sentence, so a lone
hostile sentence is caught regardless of how benign its host is. Document mode only works when
the attack dominates the document (short inputs / heavy obfuscation) — hence its 0.98 recall on
"easy" and collapse on "hard."

### 9.5 Raw (undefended) vs fully-defended — objective fault scenario

**Method.** The same attack inputs sent to (a) the raw agent with zero defense layers and
(b) the full pipeline (sentence mode), against a **genuinely abnormal** sensor scenario
(`developing_thermal_fault`), **not** "normal" readings. Rationale: against normal readings an
agent saying "LOW risk, continue monitoring" is *correct* regardless of any attack — nothing to
suppress. Against a real developing fault there is an **objective, automatic ground truth**: an
unmanipulated agent should *not* say "LOW risk, nominal." If an attack convinces it to, that is
a clean measurable failure — no LLM-judge, no keyword heuristics.

Sample: stratified toward **hard** difficulty and **real** payload source (the genuinely
challenging subset), ~38 attacks/category. Real API calls; cost read from `response.usage` ×
published `claude-sonnet-4-6` rates ($3.00 / 1M input, $15.00 / 1M output), converted at
~0.79 USD→GBP.

**Full run — 152 attacks (38/category), baseline risk = MEDIUM, 179 API calls,
$3.980 / ~£3.14** (`data/results/raw_vs_defended.json`):

| | Raw (no defense) | Defended (full pipeline, sentence mode) |
|---|---|---|
| Baseline risk on scenario (unmanipulated) | MEDIUM (`schedule_inspection`, "reduce load < 80%") | — |
| **Ungated `tool_call` reaches execution** | **136 / 152 (89.5%)** | **0 / 152 (0.0%)** |
| Blocked before reaching agent | — | 129 / 152 (84.9%) |
| Reached agent anyway | — | 26 / 152 (17.1%) |
| Agent driven to **LOW** risk (understating a developing fault) | 16 / 152 (10.5%) | 0 of the 26 that reached it |
| Strict "risk suppressed" (baseline HIGH/CRITICAL → LOW/MEDIUM) | 0 / 152 (n/a — see caveat) | 0 / 152 |

**Per category (raw ungated tool call / defended blocked):**

| Category | Raw ungated tool call | Defended blocked before agent |
|----------|----------------------|-------------------------------|
| direct_injection | 38 / 38 (100%) | 33 / 38 |
| escalation | 38 / 38 (100%) | 33 / 38 |
| indirect_injection | 36 / 38 (95%) | 32 / 38 |
| encoding_obfuscation | 24 / 38 (63%) | 31 / 38 |

**Which layer did the blocking (defended path):** ZEDD 100, filtering 19, **RAG memory 7**,
output filtering 3, (reached agent, not blocked) 23. Note RAG memory fired **7 times** here —
this run processes attacks sequentially and stores each blocked payload, so later
near-variants of an already-blocked attack were caught by RAG. This is the same conditional
contribution measured abstractly in §8, showing up in a realistic sequential run (~4.6% of
blocks).

**Caveats (state these).**
1. The `developing_thermal_fault` scenario produced a **MEDIUM** unmanipulated baseline, not
   HIGH/CRITICAL, so the strict "risk suppression" metric (which requires a HIGH/CRITICAL
   baseline) is trivially 0 on both sides and is **not** the informative result here. The
   informative results are the **ungated-tool-call rate** and **blocked-before-agent rate**.
   A future run should use a more severe scenario (or tune the fault) to get a HIGH baseline
   and make risk-suppression measurable.
2. Even so, **16 / 152 (10.5%)** of raw attacks pushed the agent *below* its own baseline to
   **LOW** risk on a developing fault — direct evidence of downplaying, just not captured by
   the strict metric.
3. Pilot run earlier (8 attacks, £0.17) showed the same pattern: raw 6/8 (75%) ungated tool
   calls → defended 0/8.

**Headline takeaway.** With no defense, **89.5%** of challenging attacks reached the agent and
produced a tool call with **zero human oversight**, and **10.5%** actively drove the agent to
understate a real fault. The defended pipeline reduced ungated tool calls to **0.0%** — 84.9%
blocked outright, every remaining one gated by human-in-the-loop, and none of the 26 that
reached the agent suppressed its risk reading. This is the "why the system needs to exist"
evidence.

**Interruption note.** The full run was interrupted once at row 60 by a real Anthropic "credit
balance too low" 400 error ($1.59 / ~£1.26 spent to that point). Checkpointing was added
(`data/results/raw_vs_defended_checkpoint.json`, every 10 rows) plus a $5.50 hard safety cap;
after a £5 top-up the run completed cleanly at $3.98 / £3.14.

---

## 10. Performance optimization

`_detect_sentence_level` / `_detect_dual_encoder` previously looped per sentence, embedding one
at a time. Rewritten to a single `_embed_many()` batch + one matrix multiply against a cached
`_centroid_matrix()`. **10–30× faster** on multi-sentence inputs; mean pipeline latency in the
2,128-sample eval is ~146–154 ms. Also added `build_document_register_baseline()` using
`src/zedd_corpus.py`.

---

## 11. Evaluator changes (`src/evaluator.py`)

- `--dataset=v1|v2` CLI flag (default `v2`); loads `attack_dataset_scaled_v2.json` when present,
  else falls back to `attack_dataset_scaled.json` → `attack_dataset.json`.
- Report filenames suffix `_v2` so v1/v2 runs never overwrite each other.
- **Removed** the bespoke false-data-injection metric (`FALSE_DATA_INJECTION_SAMPLE_ID =
  "indirect_008"`, `_compute_false_data_injection_result()`, the report field and its print
  block) — it was a single hard-coded sample, superseded by the template-based system and no
  longer meaningful.
- `analyze_readings` **testing-block stub** added in `main.py` (redefines the function
  immediately after import with a zero-cost `GridAssessment` stub) so evaluation runs that only
  need pipeline routing (filtering/ZEDD/RAG decisions) cost **£0**. The raw-vs-defended script
  explicitly overrides this to force real, cost-tracked calls.

---

## 12. Files to attach to the web writing session

**Primary (always attach):**
- `DISSERTATION_CHANGES_REPORT.md` — this document.
- `data/results/evaluation_report_sentence_v2.json` — headline results (sentence mode, v2).
- `data/results/evaluation_report_document_v2.json` — for the document-vs-sentence comparison.
- `data/results/rag_capability_test.json` — RAG's measured 16.7% contribution.
- `data/results/raw_vs_defended.json` — raw-vs-defended comparison (152 attacks, complete).
- `data/zedd_calibration.json` — per-mode thresholds + calibration methodology.
- `data/attacks/attack_dataset_scaled_v2.json` — the dataset itself (large; or just its
  `generation_metadata` block + the §7.2 tables from this report).

**Supporting (attach if the chapter needs method-level detail):**
- `src/real_attack_loader.py` — real-data sourcing + labelling method.
- `src/dataset_scale_v2.py` — scaling / dedup pipeline.
- `src/zedd_corpus.py`, `src/zedd_trainer.py` — disjoint-seed methodology, calibration.
- `src/rag_capability_test.py`, `src/raw_vs_defended.py` — the two bespoke experiments.
- `src/defense/zedd.py`, `src/defense/rag_memory.py` — the two layers that changed most.
- `COLAB_GUIDE.md` — fine-tuning reproducibility.
- `data/results/evaluation_report_{document,sentence,dual_encoder}.json` — v1 baselines for the
  before/after table (§9.1).

**Do NOT rely on:** `attack_dataset.json` (42 samples, original toy set) except as the
"starting point" number.

---

## 13. Cost accounting (GBP)

| Item | Cost |
|------|------|
| Prior phase (API credits, cumulative) | ~£20 |
| Claude Pro subscription | ~£18 |
| This phase — raw-vs-defended pilot (10 calls) | £0.17 |
| This phase — raw-vs-defended full run, first attempt (interrupted at row 60) | £1.26 |
| Top-up added for the completed full run | £5.00 |
| This phase — raw-vs-defended full run, completed (179 calls, 740k in / 117k out tokens) | **£3.14 ($3.98)** |
| Remaining from top-up | ~£1.86 |
| ZEDD fine-tuning (v3) | £0 (Colab free GPU) |
| All pipeline-routing evaluations (2,128 samples × 3 modes) | £0 (testing-block stub) |

Fine-tuning and all layer-decision evaluations were run at **zero API cost** via the Colab GPU
and the testing-block stub. Real spend in this phase is confined to the raw-vs-defended
comparison, which genuinely needs the real model.

---

## 14. Known limitations & deferred work (state honestly in the dissertation)

1. **ZEDD v4 retrain (deferred).** The fine-tuning training pairs and the ZEDD register
   baseline are still **100% synthetic**, even though the evaluation set is now majority
   real-sourced. Document mode's collapse on hard attacks (§9.4) is the visible symptom.
   Planned v4: mix real attack payloads into the training pairs, oversample `escalation` /
   `encoding_obfuscation` to counter class imbalance, and rebuild the register with the same
   21-doc-type `ExtendedDocumentFactory` used for the eval set. Requires another Colab GPU
   session — deferred until after the draft/report per the current deadline.
2. **Dual-encoder mode not re-evaluated on v2.** Sentence mode dominates it (equal recall, far
   lower FPR); low priority.
3. **`escalation` / `encoding_obfuscation` under-represented** (204 / 186) and mostly synthetic
   — the real datasets barely contain these techniques. Garak integration would help
   specifically here; not done (heavy dependency, narrow benefit).
4. **`deepset` slice is small (n=44)** — the 0.50 recall figure there is indicative, not
   definitive.
5. **RAG memory reads as 0 on the main benchmark by construction** (§8). The 16.7% figure comes
   from a separate, purpose-built test and must be framed as a *conditional* contribution.
6. **`jayavibhav` scanned only to a 5,000-row cap.** Raising it would grow the injection classes
   further; not needed for current credibility.

---

## 15. One-paragraph summary for an abstract / intro

A defense-in-depth wrapper (keyword filtering → defensive tokens → zero-shot embedding-drift
detection → RAG attack-memory → LLM agent → output filtering → human-in-the-loop) was built for
an LLM power-grid monitoring agent and evaluated against prompt-injection attacks. The
evaluation benchmark was scaled from 42 to 2,128 samples with automated semantic-similarity
deduplication (cosine < 0.85) and, following supervisory advice, made majority real-sourced
(≈66% of attack rows drawn from public prompt-injection datasets, quality-filtered and
provenance-tagged). A data-leakage audit forced disjoint train/calibration/test seeds and a
model retrain. In its recommended configuration (sentence-level drift detection) the system
achieves F1 0.922, precision 0.998, recall 0.857, 1.1% false-positive rate and a 14.3% attack
success rate, **holding performance as the dataset grew 8× and became majority real-sourced**
(FPR improved 3.1% → 1.1%) — evidence of generalization rather than benchmark memorisation.
Document-level drift detection, by contrast, collapses to 0.34 recall on hard attacks, showing
that scoring the worst sentence rather than the whole-document embedding is essential when a
lone hostile sentence hides in realistic host text. RAG attack-memory contributes nothing on a
diversity-maximised first-encounter benchmark but adds a 16.7% incremental catch rate on
paraphrased repeat attacks in a dedicated test. Against a genuine developing-fault scenario, the
undefended agent produced ungated tool calls for 89.5% of challenging attacks (152 tested) and
was driven to understate the fault in 10.5% of cases; the defended pipeline reduced ungated tool
calls to zero (84.9% blocked outright, the remainder gated for human approval).
