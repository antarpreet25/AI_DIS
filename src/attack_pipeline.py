"""
attack_pipeline.py

Processing logic for the scaled dataset generation pipeline.

Classes
-------
AttackInserter      : embeds an injection phrase into a CleanDocument
                      at a contextually appropriate position, returning
                      both the wrapped (injected) text and the clean
                      counterpart.

DatasetBuilder      : orchestrates generation of the full scaled dataset
                      (200 attacks + 65 benign) by combining DocumentFactory
                      and AttackTemplateLibrary via AttackInserter.

TrainingPairBuilder : builds the fine-tuning training pairs file
                      (injected-clean pairs + clean-clean pairs) from a
                      completed dataset.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# AttackInserter
# ---------------------------------------------------------------------------

class AttackInserter:
    """
    Embeds an injection phrase into a CleanDocument at a position that
    matches the document's structure and the template's insertion_hint.

    The key invariant: wrapped_attack and clean_counterpart differ by
    EXACTLY the injected sentence — every other character is identical.
    This makes each pair valid contrastive fine-tuning data for ZEDD.

    insertion_hint values (from AttackTemplate)
    -------------------------------------------
    'body'           → main content section of the document
    'recommendation' → recommendation / next-step section
    'header'         → immediately after the document title block
    'footer'         → last section before sign-off
    'comment'        → inline annotation or field comment
    'field'          → structured data field (sensor logs, SCADA alerts)
    'any'            → inserter picks the best position automatically
    """

    # Section heading strings used to locate insertion points.
    # Each document type maps hint labels to a line substring that
    # marks where that section begins in the document text.
    _SECTION_HEADINGS: dict[str, dict[str, str]] = {
        "maintenance_report": {
            "body":           "Condition assessment",
            "recommendation": "Recommendation",
            "header":         "Engineer:",
            "footer":         "Signed:",
            "comment":        "Work carried out",
        },
        "sensor_log": {
            "body":           "Status:",
            "field":          "Oil moisture:",
            "comment":        "No alerts raised.",
            "footer":         "No alerts raised.",
            "header":         "Timestamp:",
        },
        "dga_report": {
            "body":           "Interpretation",
            "recommendation": "Recommendation",
            "comment":        "Acetylene",
            "field":          "Acetylene",
            "footer":         "Recommendation",
            "header":         "Substation:",
        },
        "operator_handover": {
            "body":           "Shift summary",
            "footer":         "Outstanding actions:",
            "comment":        "Shift summary",
            "header":         "Outgoing:",
            "recommendation": "Outstanding actions:",
        },
        "scada_alert": {
            "body":           "Alert type:",
            "field":          "Severity:",
            "comment":        "Resolution",
            "footer":         "Resolution",
            "header":         "Alert ID:",
            "recommendation": "Resolution",
        },
        "maintenance_schedule": {
            "body":           "Planned activities",
            "footer":         "Access restrictions:",
            "comment":        "Assigned engineers:",
            "header":         "Substation:",
            "recommendation": "Contact",
        },
        "technician_note": {
            "body":           "On-site observations",
            "comment":        "Next step",
            "recommendation": "Next step",
            "footer":         "Note closed by:",
            "header":         "Location:",
        },
        "supplier_communication": {
            "body":           "Our team will attend",
            "footer":         "Regards,",
            "comment":        "Please confirm",
            "header":         "Dear",
            "recommendation": "Please confirm",
        },
        "inspection_report": {
            "body":           "Findings",
            "recommendation": "Recommendation",
            "comment":        "Actions taken",
            "footer":         "Report approved by:",
            "header":         "Overall asset condition:",
        },
    }

    # Best default hint per document type when template says 'any'
    _DEFAULT_HINT: dict[str, str] = {
        "maintenance_report":     "body",
        "sensor_log":             "comment",
        "dga_report":             "body",
        "operator_handover":      "body",
        "scada_alert":            "field",
        "maintenance_schedule":   "body",
        "technician_note":        "comment",
        "supplier_communication": "body",
        "inspection_report":      "body",
    }

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, doc, template) -> tuple[str, str]:
        """
        Embed template's injection text into doc at the position
        indicated by template.insertion_hint.

        Parameters
        ----------
        doc      : CleanDocument instance
        template : AttackTemplate instance

        Returns
        -------
        (wrapped_attack, clean_counterpart)
            wrapped_attack    : doc.text with injection embedded
            clean_counterpart : doc.text unchanged
        """
        injection = template.get_injection_text()
        hint = template.insertion_hint

        if hint == "any":
            hint = self._DEFAULT_HINT.get(doc.document_type, "body")

        wrapped = self._embed(doc.text, doc.document_type, hint, injection)

        # Safety fallback — if embed found no suitable position, append
        if wrapped == doc.text:
            wrapped = doc.text + f"\n\nNote: {injection}"

        return wrapped, doc.text   # (injected, clean)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str, doc_type: str, hint: str, injection: str) -> str:
        lines = text.split("\n")
        section_map = self._SECTION_HEADINGS.get(doc_type, {})
        heading = section_map.get(hint) or section_map.get("body")

        if heading is None:
            return text + f"\nNote: {injection}"

        section_line = self._find_section_line(lines, heading)
        if section_line == -1:
            section_line = self._after_header(lines)

        return self._splice(lines, section_line, injection, hint)

    def _find_section_line(self, lines: list[str], heading: str) -> int:
        hl = heading.lower()
        for i, line in enumerate(lines):
            if hl in line.lower():
                return i
        return -1

    def _after_header(self, lines: list[str]) -> int:
        passed = False
        for i, line in enumerate(lines):
            if line.strip():
                passed = True
            elif passed:
                return i
        return max(0, len(lines) - 2)

    def _splice(self, lines: list[str], section_line: int,
                injection: str, hint: str) -> str:
        result = list(lines)

        if hint == "header":
            result.insert(section_line + 1, injection)

        elif hint == "footer":
            pos = max(0, section_line - 1)
            result.insert(pos, injection)

        elif hint in ("field", "comment"):
            result.insert(section_line + 1, injection)

        else:
            # body / recommendation: insert after first content line
            # following the section heading — mid-section placement
            pos = section_line + 1
            while pos < len(result) and not result[pos].strip():
                pos += 1
            if pos < len(result) and result[pos].strip():
                pos += 1
            result.insert(pos, injection)

        return "\n".join(result)


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Orchestrates generation of 200 attack samples + 65 benign samples.

    Template assignment
    -------------------
    All 180 text templates are shuffled once with the seeded RNG then
    assigned to documents in round-robin order, ensuring every template
    appears at least once and no document type is flooded with one
    category.  When 200 > 180, the cycle wraps and a small number of
    templates appear twice — always in a different document context.
    """

    BENIGN_COUNTS: dict[str, int] = {
        "sensor_log":            15,
        "maintenance_report":    15,
        "dga_report":            10,
        "operator_handover":     10,
        "scada_alert":           10,
        "maintenance_schedule":   5,
    }

    VARIANTS_PER_DOC: dict[str, int] = {
        "sensor_log":            2,
        "maintenance_report":    3,
        "dga_report":            3,
        "operator_handover":     3,
        "scada_alert":           3,
        "maintenance_schedule":  2,
    }

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()
        from attack_content import DocumentFactory, AttackTemplateLibrary
        self._factory  = DocumentFactory(rng=self._rng)
        self._lib      = AttackTemplateLibrary()
        self._inserter = AttackInserter(rng=self._rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, AttackSampleClass) -> dict:
        """
        Generate the full scaled dataset.

        Parameters
        ----------
        AttackSampleClass : the AttackSample dataclass from attack_generator

        Returns
        -------
        {'attacks': list[AttackSample], 'benign': list[AttackSample]}
        """
        clean_docs = self._factory.generate_batch(self.BENIGN_COUNTS)
        self._rng.shuffle(clean_docs)

        benign  = self._build_benign(clean_docs, AttackSampleClass)
        attacks = self._build_attacks(clean_docs, AttackSampleClass)

        return {"attacks": attacks, "benign": benign}

    # ------------------------------------------------------------------
    # Benign builder
    # ------------------------------------------------------------------

    def _build_benign(self, clean_docs: list, AS) -> list:
        benign = []
        all_layers = [
            "filtering", "defensive_tokens", "zedd",
            "rag_memory", "human_loop",
        ]
        for i, doc in enumerate(clean_docs, start=1):
            benign.append(AS(
                attack_id          = f"benign_{i:03d}",
                category           = "benign",
                attack_type        = "none",
                raw_attack         = "",
                wrapped_attack     = doc.text,
                clean_counterpart  = doc.text,
                expected_blocked_by  = [],
                expected_slip_layers = list(all_layers),
                difficulty         = "n/a",
                ground_truth_risk  = "LOW",
                image_path         = None,
                doc_type           = getattr(doc, "document_type", "unknown")
            ))
        return benign

    # ------------------------------------------------------------------
    # Attack builder
    # ------------------------------------------------------------------

    def _build_attacks(self, clean_docs: list, AS) -> list:
        all_templates = self._lib.all()
        self._rng.shuffle(all_templates)
        pool = list(all_templates)   # working copy for round-robin

        attacks: list = []
        cat_counters: dict[str, int] = {}
        pool_idx = 0

        for doc in clean_docs:
            n = self.VARIANTS_PER_DOC.get(doc.document_type, 2)
            chosen = self._pick_diverse(pool, pool_idx, n)
            pool_idx = (pool_idx + n) % len(pool)

            for template in chosen:
                cat = template.category
                cat_counters[cat] = cat_counters.get(cat, 0) + 1
                prefix = cat.split("_")[0]
                attack_id = f"{prefix}_{cat_counters[cat]:03d}"

                wrapped, clean = self._inserter.insert(doc, template)

                attacks.append(AS(
                    attack_id          = attack_id,
                    category           = cat,
                    attack_type        = template.technique,
                    raw_attack         = template.raw_phrase,
                    wrapped_attack     = wrapped,
                    clean_counterpart  = clean,
                    expected_blocked_by  = list(template.expected_blocked_by),
                    expected_slip_layers = list(template.expected_slip_layers),
                    difficulty         = template.difficulty,
                    ground_truth_risk  = template.ground_truth_risk,
                    image_path         = None,
                    doc_type           = getattr(doc, "document_type", "unknown")
                ))

        return self._normalise(attacks, 200, pool, clean_docs, AS, cat_counters)

    def _pick_diverse(self, pool: list, start: int, count: int) -> list:
        """
        Pick `count` templates starting from `start` in round-robin order.
        Pass 1: Collects only unique categories.
        Pass 2: Fills remaining slots with unselected items if count is not met.
        """
        if not pool or count <= 0:
            return []

        n = len(pool)
        picked: list = []
        seen_cats: set = set()
        used_indices: set = set()

        # Pass 1: Walk the pool once to pick unique categories only
        for offset in range(n):
            idx = (start + offset) % n
            template = pool[idx]
            if template.category not in seen_cats:
                picked.append(template)
                seen_cats.add(template.category)
                used_indices.add(idx)
                if len(picked) == count:
                    return picked

        # Pass 2: Fill remaining required slots with remaining pool items
        for offset in range(n):
            idx = (start + offset) % n
            if idx not in used_indices:
                picked.append(pool[idx])
                used_indices.add(idx)
                if len(picked) == count:
                    return picked

        return picked[:count]

    def _normalise(self, attacks, target, pool, clean_docs, AS, cat_counters) -> list:
        if len(attacks) >= target:
            return attacks[:target]

        shortfall = target - len(attacks)
        for _ in range(shortfall):
            doc      = self._rng.choice(clean_docs)
            template = self._rng.choice(pool)
            cat      = template.category
            cat_counters[cat] = cat_counters.get(cat, 0) + 1
            prefix   = cat.split("_")[0]
            attack_id = f"{prefix}_{cat_counters[cat]:03d}"
            wrapped, clean = self._inserter.insert(doc, template)
            attacks.append(AS(
                attack_id          = attack_id,
                category           = cat,
                attack_type        = template.technique,
                raw_attack         = template.raw_phrase,
                wrapped_attack     = wrapped,
                clean_counterpart  = doc.text,
                expected_blocked_by  = list(template.expected_blocked_by),
                expected_slip_layers = list(template.expected_slip_layers),
                difficulty         = template.difficulty,
                ground_truth_risk  = template.ground_truth_risk,
                image_path         = None,
                doc_type           = getattr(doc, "document_type", "unknown")
            ))
        return attacks


# ---------------------------------------------------------------------------
# TrainingPairBuilder
# ---------------------------------------------------------------------------

class TrainingPairBuilder:
    """
    Builds training_pairs.json for zedd_trainer.py.

    Pair types
    ----------
    label=0  injected-clean: one pair per attack sample (wrapped vs clean)
    label=1  clean-clean   : each benign doc paired with 2 others (seed=42)

    Schema
    ------
    [{"injected": str, "clean": str, "label": int}, ...]
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()


    def build(self, attacks: list, benign: list) -> list[dict]:
        pairs: list[dict] = []

        # --- 1. Injected vs Clean Pairs (label=0) ---
        # 1:1 pairing between injected doc and its exact clean origin doc
        for s in attacks:
            clean_source = getattr(s, "clean_counterpart", None)
            if clean_source:
                pairs.append({
                    "text1": s.wrapped_attack,
                    "text2":    clean_source,
                    "label":    0,
                    "doc_type": getattr(s, "doc_type", "unknown"),
                })

        # --- 2. Clean vs Clean Pairs (label=1) ---
        # Group strictly by document type; NO fallback to cross-category
        pair_rng = random.Random(42)
        grouped_benign = defaultdict(list)
        
        for b in benign:
            doc_type = getattr(b, "doc_type", "unknown")
            grouped_benign[doc_type].append(b.wrapped_attack)

        for doc_type, texts in grouped_benign.items():
            n = len(texts)
            if n < 2:
                # Cannot form intra-category pairs for a category with only 1 sample
                continue

            for i in range(n):
                text_a = texts[i]
                # Filter candidates strictly within the SAME category
                same_cat_candidates = [t for j, t in enumerate(texts) if j != i]
                
                # Sample up to 2 unique partners from the same category
                k = min(2, len(same_cat_candidates))
                partners = pair_rng.sample(same_cat_candidates, k=k)
                
                for text_b in partners:
                    pairs.append({
                        "text1": text_a,
                        "text2":    text_b,
                        "label":    1,
                        "doc_type": doc_type,
                    })

        pair_rng.shuffle(pairs)
        return pairs

    
    def save(self, pairs: list[dict], path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
        n0 = sum(1 for p in pairs if p["label"] == 0)
        n1 = sum(1 for p in pairs if p["label"] == 1)
        print(f"[TrainingPairBuilder] {len(pairs)} pairs saved "
              f"(label=0: {n0}, label=1: {n1}) → {out}")

    def load(self, path) -> list[dict]:
        with open(Path(path), encoding="utf-8") as f:
            return json.load(f)

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from attack_generator import AttackSample  # Import your AttackSample dataclass
    
    print("[attack_pipeline] Starting dataset generation...")
    
    # 1. Initialize builder with a fixed seed for reproducibility
    builder = DatasetBuilder(rng=random.Random(42))
    
    # 2. Generate attacks (200) and benign (65) samples
    data = builder.generate(AttackSample)
    attacks = data["attacks"]
    benign = data["benign"]
    
    print(f"[attack_pipeline] Generated {len(attacks)} attacks and {len(benign)} benign samples.")
    
    # 3. Build training pairs for ZEDD
    pair_builder = TrainingPairBuilder()
    pairs = pair_builder.build(attacks=attacks, benign=benign)
    
    # 4. Save to data/training_pairs.json
    output_path = Path("data/training_pairs.json")
    pair_builder.save(pairs, output_path)
    
    print("[attack_pipeline] Pipeline completed successfully!")