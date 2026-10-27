"""Adds InferenceX measurements (validation/parse_dump.py output) to data.json as a second data source.

Usage: python3 build_measured.py tables.json data.json [max_age_days]

Keeps single-turn runs that succeeded, without speculative decoding or KV offload, from the last
max_age_days (default 120) of the dump, on the three systems of the page. Points use the same layout
as the simulated ones, keyed by the model card id: "system|isl|osl|0" -> {"agg": [...], "disagg": [...]}.
"""
import json, sys
from datetime import date

tables, out = json.load(open(sys.argv[1])), sys.argv[2]
MAX_AGE = int(sys.argv[3]) if len(sys.argv) > 3 else 120
data = json.load(open(out))

HW = {"gb300": "gb300", "b300": "b300_sxm", "rtx6000pro": "rtx_pro_6000_server"}
# InferenceX model alias -> (model card id, display name used when the model is not already on the page)
MODELS = {
    "qwen3.5": ("Qwen/Qwen3.5-397B-A17B", "Qwen3.5-397B-A17B"),
    "kimik2.5": ("moonshotai/Kimi-K2.5", "Kimi K2.5"),
    "dsv4": ("deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro"),
    "minimaxm3": ("MiniMaxAI/MiniMax-M3", "MiniMax M3"),
    "dsr1": ("deepseek-ai/DeepSeek-R1-0528", "DeepSeek R1 0528"),
    "minimaxm2.5": ("MiniMaxAI/MiniMax-M2.5", "MiniMax M2.5"),
    "glm5": ("zai-org/GLM-5", "GLM-5"),
    "glm5.1": ("zai-org/GLM-5.1", "GLM-5.1"),
    "gptoss120b": ("openai/gpt-oss-120b", "gpt-oss-120b"),
    "glm5.2": ("zai-org/GLM-5.2", "GLM-5.2"),
    "minimaxm2.7": ("MiniMaxAI/MiniMax-M2.7", "MiniMax M2.7"),
}

cfg = {c["id"]: c for c in tables["configs"]}
runs = {r["id"]: r for r in tables["workflow_runs"]}
latest = max(date.fromisoformat(r["date"]) for r in tables["benchmark_results"])


def topology(c):
    if c["disagg"]:
        return (f"P {c['prefill_num_workers']}x tp{c['prefill_tp']} ep{c['prefill_ep']} / "
                f"D {c['decode_num_workers']}x tp{c['decode_tp']} ep{c['decode_ep']}")
    return f"tp{c['decode_tp']} ep{c['decode_ep']}"


measured, count = {}, 0
for r in tables["benchmark_results"]:
    c, run, m = cfg[r["config_id"]], runs[r["workflow_run_id"]], r["metrics"] or {}
    if (c["hardware"] not in HW or c["model"] not in MODELS or r["benchmark_type"] != "single_turn"
            or r["error"] or r["offload_mode"] != "off" or c["spec_method"] != "none"
            or run.get("conclusion") != "success"
            or (latest - date.fromisoformat(r["date"])).days > MAX_AGE):
        continue
    tpg, ttft, tpot = m.get("output_tput_per_gpu"), m.get("mean_ttft"), m.get("mean_tpot")
    if not (tpg and ttft and tpot):
        continue
    gpus = (c["num_decode_gpu"] or 0) + ((c["num_prefill_gpu"] or 0) if c["disagg"] else 0)
    label = (f"{c['framework']} {c['precision'].upper()} · {gpus} GPUs · {topology(c)} · conc {r['conc']}"
             f" · measured {r['date']}")
    point = [round(tpg, 3), round(1 / tpot, 3), round(ttft * 1000, 3), round(tpot * 1000, 3),
             round(tpg / r["osl"], 5), label, run.get("html_url")]
    key = f"{HW[c['hardware']]}|{r['isl']}|{r['osl']}|0"
    mode = "disagg" if c["disagg"] else "agg"
    measured.setdefault(MODELS[c["model"]][0], {}).setdefault(key, {}).setdefault(mode, []).append(point)
    count += 1

# Models measured by InferenceX but not yet listed on the page: listed without accuracy scores.
known = {m["bench"] for m in data["models"]}
for bench_id, name in MODELS.values():
    if bench_id in measured and bench_id not in known:
        data["models"].append({"sim": bench_id, "bench": bench_id, "name": name, "scores": {}})

data["measured"] = measured
data["measuredInfo"] = {
    "source": "SemiAnalysis InferenceX public database dump",
    "release": "db-dump/2026-09-14",
    "through": latest.isoformat(),
    "maxAgeDays": MAX_AGE,
    "points": count,
}
json.dump(data, open(out, "w"), separators=(",", ":"))
print(f"measured points {count}, models {len(measured)}, combos {sum(len(v) for v in measured.values())}")
