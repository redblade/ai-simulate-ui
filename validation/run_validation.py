"""Runs the AISimulate estimator (the same one Infra Planner uses) on InferenceX measured points.

Usage: python3 run_validation.py tables.json results.jsonl [hardware ...] [--workers N] [--max-age DAYS]

For each measured point (model, system, framework, parallelism, ISL/OSL, concurrency) it asks
NVIDIA's InferenceX adapter, shipped in the aisimulate wheel, to translate the configuration,
runs `cli_estimate` and stores predicted and measured TTFT, TPOT and output tokens/s per GPU.
Points that cannot be simulated keep their reason (adapter rejection, missing perf data...).
Speculative decoding (MTP) points are skipped: their acceptance rate is not in the dump.
Finished points are skipped on restart.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

ap = argparse.ArgumentParser()
ap.add_argument("tables")
ap.add_argument("out")
ap.add_argument("hardware", nargs="*", default=["gb300", "b300"])
ap.add_argument("--workers", type=int, default=os.cpu_count())
ap.add_argument("--max-age", type=int, default=120)
args = ap.parse_args()

# InferenceX hardware names the adapter does not map itself.
SYSTEM_OVERRIDE = {"rtx6000pro": "rtx_pro_6000_server"}


def select(tables):
    cfg = {c["id"]: c for c in tables["configs"]}
    runs = {r["id"]: r for r in tables["workflow_runs"]}
    latest = max(date.fromisoformat(r["date"]) for r in tables["benchmark_results"])
    best = {}
    for r in tables["benchmark_results"]:
        c, run = cfg[r["config_id"]], runs[r["workflow_run_id"]]
        m = r["metrics"] or {}
        if (c["hardware"] not in args.hardware or r["benchmark_type"] != "single_turn" or r["error"]
                or r["offload_mode"] != "off" or c["spec_method"] != "none"
                or run.get("conclusion") != "success" or not m.get("mean_ttft") or not m.get("mean_tpot")
                or (latest - date.fromisoformat(r["date"])).days > args.max_age):
            continue
        # One measurement per configuration and workload: the most recent.
        key = (r["config_id"], r["isl"], r["osl"], r["conc"], r.get("recipe_fingerprint"))
        if key not in best or r["date"] > best[key][1]["date"]:
            best[key] = (c, r)
    return list(best.values())


def predict(item):
    import io, contextlib, logging
    logging.disable(logging.WARNING)
    from aisimulate.legacy_cli.api import cli_estimate
    import aisimulate.sdk.config_adapter as ad

    c, r = item
    m = r["metrics"]
    gpus = (c["num_decode_gpu"] or 0) + ((c["num_prefill_gpu"] or 0) if c["disagg"] else 0)
    row = {
        "benchmark_id": r["id"], "date": r["date"], "hardware": c["hardware"], "model": c["model"],
        "precision": c["precision"], "framework": c["framework"], "disagg": c["disagg"],
        "multinode": c["is_multinode"], "isl": r["isl"], "osl": r["osl"], "conc": r["conc"], "gpus": gpus,
        "prefill": [c["prefill_tp"], c["prefill_ep"], c["prefill_num_workers"], c["num_prefill_gpu"]],
        "decode": [c["decode_tp"], c["decode_ep"], c["decode_num_workers"], c["num_decode_gpu"]],
        "meas_ttft_ms": m["mean_ttft"] * 1000, "meas_tpot_ms": m["mean_tpot"] * 1000,
        "meas_out_per_gpu": m.get("output_tput_per_gpu"),
    }
    try:
        overrides = ad.AdapterOverrides(system_name=SYSTEM_OVERRIDE[c["hardware"]]) \
            if c["hardware"] in SYSTEM_OVERRIDE else None
        rep = ad.adapt_config(ad.InferenceXSource(config=c, benchmark=r), overrides)
        if len(rep.requests) != 1:
            diag = [d.message for o in rep.outcomes for d in (o.diagnostics or ())]
            return {**row, "status": "adapter_rejected", "reason": "; ".join(diag)[:300]}
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            est = cli_estimate(**ad.to_cli_estimate_kwargs(rep.requests[0]))
        tps = getattr(est, "tokens_per_second", None)
        return {**row, "status": "ok", "seconds": round(time.time() - t0, 1),
                "pred_ttft_ms": float(est.ttft), "pred_tpot_ms": float(est.tpot),
                "pred_out_per_gpu": float(tps) / gpus if tps and gpus else None,
                "backend_version": getattr(est, "backend_version", None)}
    except Exception as e:  # noqa: BLE001 - every failure is a recorded outcome
        return {**row, "status": type(e).__name__, "reason": str(e)[:300]}


if __name__ == "__main__":
    points = select(json.load(open(args.tables)))
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["benchmark_id"] for l in open(args.out)}
    todo = [p for p in points if p[1]["id"] not in done]
    print(f"{len(points)} points selected, {len(todo)} to run", flush=True)
    with open(args.out, "a") as out, ProcessPoolExecutor(args.workers) as pool:
        for i, fut in enumerate(as_completed(pool.submit(predict, p) for p in todo), 1):
            out.write(json.dumps(fut.result()) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
