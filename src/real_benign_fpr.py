"""
real_benign_fpr.py   (ADDITIVE — imports the pipeline read-only; changes
nothing existing)

Item A. Measures the defence pipeline's FALSE-POSITIVE RATE on REAL,
publicly-sourced benign power-grid text (data/real_benign/grid_text_corpus.json)
rather than on synthetically-generated benign documents only.

Each passage is verbatim domain text with no injected instruction; a
correct pipeline delivers all of them to the agent. Anything blocked is a
false positive. Reports overall FPR, a per-source breakdown, and which
layer produced each false block, with the agent stubbed (£0).

USAGE
    python src/real_benign_fpr.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

CORPUS = _ROOT / "data" / "real_benign" / "grid_text_corpus.json"
OUT = _ROOT / "data" / "results" / "real_benign_fpr.json"


def main():
    from main import Pipeline
    from sensor_generator import build_scenarios, generate_readings

    passages = json.loads(CORPUS.read_text(encoding="utf-8"))["passages"]
    readings = generate_readings(build_scenarios()["normal"], duration_min=90, interval_min=5, seed=7)
    pipe = Pipeline(zedd_mode="sentence")

    rows, by_source = [], defaultdict(lambda: [0, 0])   # source -> [false_blocks, total]
    by_layer = defaultdict(int)
    for p in passages:
        res = pipe.run(p["text"], readings, n_window=10)
        fp = res.blocked
        rows.append({"source": p["source"], "text": p["text"][:90],
                     "blocked": fp, "blocked_by": res.blocked_by if fp else None})
        by_source[p["source"]][1] += 1
        if fp:
            by_source[p["source"]][0] += 1
            by_layer[res.blocked_by] += 1

    n = len(passages)
    fp_n = sum(r["blocked"] for r in rows)
    print("\n" + "=" * 62)
    print(f"REAL BENIGN GRID TEXT — false-positive rate  (n={n}, agent stubbed, £0)")
    print("=" * 62)
    print(f"  false positives : {fp_n}/{n}  ({fp_n/n:.1%})")
    print(f"  delivered clean : {n-fp_n}/{n}  ({(n-fp_n)/n:.1%})")
    print("  by source (false blocks / total):")
    for s, (b, t) in sorted(by_source.items()):
        print(f"    {s:16} {b}/{t}")
    if by_layer:
        print("  false blocks by layer:", dict(by_layer))
        print("\n  false-positive passages:")
        for r in rows:
            if r["blocked"]:
                print(f"    [{r['blocked_by']}] {r['text']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n": n, "false_positives": fp_n, "fpr": round(fp_n / n, 4),
        "by_source": {s: {"false_blocks": b, "total": t} for s, (b, t) in by_source.items()},
        "by_layer": dict(by_layer), "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
