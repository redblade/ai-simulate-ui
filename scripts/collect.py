"""Collects the pareto.csv files of the AISimulate runs into one compact JSON for the UI."""
import csv, json, re, sys
from pathlib import Path

runs_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
TAG = re.compile(r"(.+)_(gb300|b300_sxm|rtx_pro_6000_server)_i(\d+)_o(\d+)_p(\d+)$")
FIELDS = ["tokens/s/gpu", "tokens/s/user", "ttft", "tpot", "seq/s/gpu"]

def f(x):
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None

def points(path, mode):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            vals = [f(r.get(k)) for k in FIELDS]
            if None in vals:
                continue
            if mode == "agg":
                cfg = f"{r.get('parallel')} bs{r.get('bs')} x{r.get('num_total_gpus')}gpu"
            else:
                cfg = (f"P {r.get('(p)workers')}x{r.get('(p)parallel')} / "
                       f"D {r.get('(d)workers')}x{r.get('(d)parallel')} bs{r.get('(d)bs')}")
            rows.append(vals + [cfg])
    return rows

results, failures = {}, []
for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
    m = TAG.match(d.name)
    if not m or not (d / ".done").exists():
        continue
    model, system, isl, osl, pct = m.groups()
    key = f"{system}|{isl}|{osl}|{pct}"
    entry = {}
    for mode in ("agg", "disagg"):
        found = list(d.glob(f"**/{mode}/pareto.csv"))
        if found:
            pts = points(found[0], mode)
            if pts:
                entry[mode] = pts
    if entry:
        results.setdefault(model, {})[key] = entry
    else:
        log = (d / "log.txt").read_text(errors="ignore") if (d / "log.txt").exists() else ""
        err = [l for l in log.splitlines() if "[E]" in l]
        failures.append({"model": model, "key": key, "error": (err[-1][-300:] if err else "no results")})

json.dump({"fields": FIELDS + ["config"], "gpus": 72, "results": results, "failures": failures},
          open(out, "w"), separators=(",", ":"))
print(f"models={len(results)} combos={sum(len(v) for v in results.values())} failures={len(failures)}")
