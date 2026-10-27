"""Compares measured throughput per GPU between two InferenceX systems at equal interactivity.

Usage: python3 measured_compare.py tables.json [hw_a] [hw_b] [max_age_days]
For each (model, precision, isl, osl, spec_method) it builds the measured frontier
(best output tokens/s per GPU at >= X tokens/s per user) of each system across
all frameworks and configurations, and prints hw_a / hw_b at several interactivity levels.
"""
import json, sys, statistics as st
from collections import defaultdict
from datetime import date

t = json.load(open(sys.argv[1]))
A, B = (sys.argv[2:4] if len(sys.argv) > 3 else ("gb300", "b300"))
MAX_AGE = int(sys.argv[4]) if len(sys.argv) > 4 else 120
cfg = {c["id"]: c for c in t["configs"]}
runs = {r["id"]: r for r in t["workflow_runs"]}
latest = max(date.fromisoformat(r["date"]) for r in t["benchmark_results"])

pts = defaultdict(list)  # (model, prec, isl, osl, spec, hw) -> [(interactivity, tput/gpu, disagg, fw)]
for r in t["benchmark_results"]:
    c, run, m = cfg[r["config_id"]], runs[r["workflow_run_id"]], r["metrics"] or {}
    if (r["benchmark_type"] != "single_turn" or r["error"] or r["offload_mode"] != "off"
            or run.get("conclusion") != "success" or c["hardware"] not in (A, B)
            or (latest - date.fromisoformat(r["date"])).days > MAX_AGE):
        continue
    x, y = m.get("median_intvty") or m.get("mean_intvty"), m.get("output_tput_per_gpu")
    if x and y:
        pts[(c["model"], c["precision"], r["isl"], r["osl"], c["spec_method"], c["hardware"])].append(
            (x, y, c["disagg"], c["framework"]))

def best(p, xmin):
    # Only when the measured curve spans xmin (some point at or below it), so a
    # system is not penalized for not having been measured at high concurrency.
    if min(x for x, *_ in p) > xmin:
        return None
    v = [y for x, y, *_ in p if x >= xmin]
    return max(v) if v else None

LEVELS = (20, 50, 100, 200)
ratios = defaultdict(list)
print(f"Measured {A}/{B}, output tok/s per GPU at >= X tok/s per user (data through {latest}, last {MAX_AGE} days)")
for key in sorted({k[:5] for k in pts}):
    pa, pb = pts.get(key + (A,)), pts.get(key + (B,))
    if not pa or not pb:
        continue
    row = []
    for lv in LEVELS:
        a, b = best(pa, lv), best(pb, lv)
        row.append(f"{a / b:5.2f}" if a and b else "    -")
        if a and b:
            ratios[lv].append(a / b)
    print(" ".join(map(str, key)).ljust(40), f"n={len(pa)}/{len(pb)}", *row)
for lv in LEVELS:
    r = ratios[lv]
    if r:
        print(f">= {lv} tok/s/user: {len(r)} combos, median {st.median(r):.2f}, min {min(r):.2f}, max {max(r):.2f}")
