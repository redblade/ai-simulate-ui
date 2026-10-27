"""Converts the pg_restore COPY output of the InferenceX dump into tables.json.

Usage: python3 parse_dump.py tables.copy tables.json
Keeps configs, benchmark_results and workflow_runs (the tables used for validation).
"""
import json, re, sys

TABLES = {"configs", "benchmark_results", "workflow_runs"}
INTS = {"id", "workflow_run_id", "config_id", "isl", "osl", "conc", "run_attempt", "github_run_id",
        "prefill_tp", "prefill_ep", "prefill_num_workers", "decode_tp", "decode_ep",
        "decode_num_workers", "num_prefill_gpu", "num_decode_gpu"}
BOOLS = {"disagg", "is_multinode", "prefill_dp_attention", "decode_dp_attention"}
ESC = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}


def value(v):
    if v == r"\N":
        return None
    return re.sub(r"\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)",
                  lambda m: chr(int(m[1][1:], 16)) if m[1][0] == "x"
                  else chr(int(m[1], 8)) if m[1][0].isdigit() else ESC.get(m[1], m[1]), v)


tables, current, cols = {}, None, []
with open(sys.argv[1]) as fh:
    for line in fh:
        if current is None:
            m = re.fullmatch(r"COPY public\.([a-z_]+) \(([^)]+)\) FROM stdin;\n?", line)
            if m and m[1] in TABLES:
                current, cols = m[1], [c.strip('"') for c in m[2].split(", ")]
                tables[current] = []
            continue
        if line.rstrip("\n") == r"\.":
            current = None
            continue
        row = dict(zip(cols, map(value, line.rstrip("\n").split("\t")), strict=True))
        for k in INTS & row.keys():
            if row[k] is not None:
                row[k] = int(row[k])
        for k in BOOLS & row.keys():
            if row[k] is not None:
                row[k] = row[k] == "t"
        if row.get("metrics"):
            row["metrics"] = json.loads(row["metrics"])
        tables[current].append(row)

assert set(tables) == TABLES, tables.keys()
json.dump(tables, open(sys.argv[2], "w"))
print({k: len(v) for k, v in tables.items()})
