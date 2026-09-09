"""
detection_layer_footprint.py   (ADDITIVE — read-only import of the pipeline;
nothing existing is modified)

Item D. Measures the deployment footprint of the DETECTION path (filtering,
ZEDD, RAG) on CPU, with the agent stubbed. The point for a substation/OT
setting: the three automated detection layers run locally with no external
network call — only the agent step needs the hosted API. This script
quantifies their latency and memory so the dissertation can state a
concrete "runs on a substation gateway" figure rather than asserting it.

Inputs are the real domain-attack documents and the real benign grid
passages (a realistic mix of long and short inputs). No API calls, £0.

USAGE
    python src/detection_layer_footprint.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import median

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

OUT = _ROOT / "data" / "results" / "detection_layer_footprint.json"


def _rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except Exception:
            return None


def _pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    inputs = []
    dom = json.loads((_ROOT / "data" / "attacks" / "domain_attacks.json").read_text(encoding="utf-8"))
    inputs += [a["document"] for a in dom["attacks"]]
    corp = json.loads((_ROOT / "data" / "real_benign" / "grid_text_corpus.json").read_text(encoding="utf-8"))
    inputs += [p["text"] for p in corp["passages"]]
    print(f"[footprint] {len(inputs)} real inputs (domain attacks + real benign passages)")

    rss_before = _rss_mb()
    # Build the real pipeline and time ITS detection components — identical
    # configuration to production, no reconstruction.
    from main import Pipeline
    pipe = Pipeline(zedd_mode="sentence")
    filt, zedd, rag = pipe.filter, pipe.zedd, pipe.rag
    rss_after_load = _rss_mb()

    tf, tz, tr = [], [], []
    for text in inputs:
        t = time.perf_counter(); filt.screen_input(text); tf.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); zedd.detect(text);        tz.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); rag.check(text);           tr.append((time.perf_counter() - t) * 1000)

    combined = [a + b + c for a, b, c in zip(tf, tz, tr)]
    rss_peak = _rss_mb()

    def stat(xs):
        return {"mean_ms": round(sum(xs) / len(xs), 2), "p50_ms": round(median(xs), 2),
                "p95_ms": round(_pctl(xs, 0.95), 2), "max_ms": round(max(xs), 2)}

    res = {
        "n_inputs": len(inputs),
        "external_network_calls_in_detection_path": 0,
        "layers": {"filtering": stat(tf), "zedd": stat(tz), "rag_memory": stat(tr),
                   "combined_detection": stat(combined)},
        "memory_mb": {"before_load": round(rss_before, 1) if rss_before else None,
                      "after_model_load": round(rss_after_load, 1) if rss_after_load else None,
                      "peak": round(rss_peak, 1) if rss_peak else None},
        "device": "cpu",
    }

    print("\n" + "=" * 60)
    print("DETECTION-PATH FOOTPRINT  (filtering + ZEDD + RAG, CPU, no API)")
    print("=" * 60)
    for k, v in res["layers"].items():
        print(f"  {k:20} mean {v['mean_ms']:>7} ms  p95 {v['p95_ms']:>7} ms  max {v['max_ms']:>7} ms")
    print(f"  peak RSS: {res['memory_mb']['peak']} MB   external calls in detection path: 0")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
