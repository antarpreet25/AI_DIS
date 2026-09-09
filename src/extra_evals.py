"""
extra_evals.py   (ADDITIVE — read-only pipeline use; nothing existing changed)

Runs two zero-cost supplementary evaluations through the UNCHANGED pipeline
(agent stubbed):

  E  grid hard negatives  (data/attacks/grid_hard_negatives.json)
     Legitimate but alarming-sounding grid notes. Correct = deliver all.
     Reports over-flagging rate (domain-specific false positives).

  F  cross-domain control  (data/attacks/cross_domain_probe.json)
     Non-grid text. `benign` should be delivered; `injection` should still
     be blocked even though ZEDD's register is grid language. Probes the
     boundary of the domain conditioning.

USAGE
    python src/extra_evals.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OUT = _ROOT / "data" / "results" / "extra_evals.json"


def run(pipe, readings, items):
    rows = []
    for it in items:
        res = pipe.run(it["text"], readings, n_window=10)
        rows.append({"id": it["id"], "blocked": res.blocked,
                     "blocked_by": res.blocked_by if res.blocked else None})
    return rows


def main():
    from main import Pipeline
    from sensor_generator import build_scenarios, generate_readings

    readings = generate_readings(build_scenarios()["normal"], duration_min=90, interval_min=5, seed=7)
    pipe = Pipeline(zedd_mode="sentence")

    hn = json.loads((_ROOT / "data" / "attacks" / "grid_hard_negatives.json").read_text(encoding="utf-8"))["negatives"]
    xd = json.loads((_ROOT / "data" / "attacks" / "cross_domain_probe.json").read_text(encoding="utf-8"))

    hn_rows = run(pipe, readings, hn)
    xb_rows = run(pipe, readings, xd["benign"])
    xi_rows = run(pipe, readings, xd["injection"])

    def rate(rows):
        b = sum(r["blocked"] for r in rows)
        return b, len(rows), (b / len(rows) if rows else 0.0)

    hn_b, hn_n, hn_r = rate(hn_rows)
    xb_b, xb_n, xb_r = rate(xb_rows)
    xi_b, xi_n, xi_r = rate(xi_rows)

    print("\n" + "=" * 60)
    print("E — GRID HARD NEGATIVES  (alarming but legitimate; correct = deliver)")
    print("=" * 60)
    print(f"  over-flagged (false block): {hn_b}/{hn_n}  ({hn_r:.1%})")
    if hn_b:
        print("  layers:", dict(Counter(r["blocked_by"] for r in hn_rows if r["blocked"])))
        for r in hn_rows:
            if r["blocked"]:
                print(f"    [{r['blocked_by']}] {r['id']}")

    print("\n" + "=" * 60)
    print("F — CROSS-DOMAIN CONTROL  (non-grid text through the grid pipeline)")
    print("=" * 60)
    print(f"  non-grid BENIGN  blocked (false positive): {xb_b}/{xb_n}  ({xb_r:.1%})")
    print(f"  non-grid INJECTION blocked (still caught) : {xi_b}/{xi_n}  ({xi_r:.1%})")
    print(f"    injection layers: {dict(Counter(r['blocked_by'] for r in xi_rows if r['blocked']))}")
    if xi_n - xi_b:
        print("    injections that slipped:", [r["id"] for r in xi_rows if not r["blocked"]])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "grid_hard_negatives": {"blocked": hn_b, "n": hn_n, "over_flag_rate": round(hn_r, 4), "rows": hn_rows},
        "cross_domain_benign": {"blocked": xb_b, "n": xb_n, "fpr": round(xb_r, 4), "rows": xb_rows},
        "cross_domain_injection": {"blocked": xi_b, "n": xi_n, "catch_rate": round(xi_r, 4), "rows": xi_rows},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
