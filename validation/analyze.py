"""Summarizes run_validation.py results: coverage and prediction error per system.

Usage: python3 analyze.py results.jsonl
Error = predicted / measured - 1. "MAPE" is the mean absolute error, "median" the median signed error
(negative = the simulator predicts less than measured), "±25%" the share of points within 25%.
"""
import json, sys, statistics as st
from collections import Counter, defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]


def stats(rs, metric):
    e = [r[f"pred_{metric}"] / r[f"meas_{metric}"] - 1 for r in rs
         if r.get(f"pred_{metric}") and r.get(f"meas_{metric}")]
    if not e:
        return "-"
    return (f"MAPE {100 * st.mean(map(abs, e)):4.0f}%  median {100 * st.median(e):+4.0f}%  "
            f"±25% {100 * sum(abs(x) <= .25 for x in e) / len(e):3.0f}%")


def group(name, key):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    print(f"\n== {name}")
    for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])):
        ok = [r for r in rs if r["status"] == "ok"]
        print(f"{str(k):34} points {len(rs):4}  simulated {len(ok):4} ({100 * len(ok) / len(rs):3.0f}%)")
        for m, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"), ("out_per_gpu", "tok/s/GPU")):
            if ok:
                print(f"    {label:10} {stats(ok, m)}")


def wide(r):
    tp, ep, n, g = r["decode"]
    return "decode EP>=16" if (ep or 1) >= 16 else "decode EP<16"


group("by system and serving mode", lambda r: (r["hardware"], "disagg" if r["disagg"] else "agg"))
group("by system and model", lambda r: (r["hardware"], r["model"], r["precision"]))
group("GB300 coverage by decode EP", lambda r: (r["hardware"], wide(r)))
print("\n== reasons for points that could not be simulated")
for (hw, s), n in sorted(Counter((r["hardware"], r["status"] + ": " + r.get("reason", "")[:70])
                                 for r in rows if r["status"] != "ok").items(), key=lambda x: -x[1])[:12]:
    print(f"{n:4}  {hw:6} {s}")
