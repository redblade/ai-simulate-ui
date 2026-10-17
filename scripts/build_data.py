"""Merges simulations (sim.json from collect.py) and benchmarks into data.json for the UI."""
import json, sys, datetime
sim = json.load(open(sys.argv[1]))
bench = json.load(open(sys.argv[2]))
out = sys.argv[3]

# simulated id (quantized checkpoint with AISimulate data) -> benchmark model card id
SIM_TO_BENCH = {
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "nvidia/Qwen3.5-397B-A17B-NVFP4": "Qwen/Qwen3.5-397B-A17B",
    "nvidia/Qwen3.5-122B-A10B-NVFP4": "Qwen/Qwen3.5-122B-A10B",
    "nvidia/Kimi-K2.7-Code-NVFP4": "nvidia/Kimi-K2.7-Code-NVFP4",
    "nvidia/MiniMax-M2.7-NVFP4": "MiniMaxAI/MiniMax-M2.7",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "nvidia/GLM-5.2-NVFP4": "zai-org/GLM-5.2",
    "sgl-project/DeepSeek-V4-Flash-FP8": "deepseek-ai/DeepSeek-V4-Flash",
}
QUANT = {k: ("NVFP4" if "NVFP4" in k else "FP8" if "FP8" in k else "MXFP4" if "gpt-oss" in k else "BF16") for k in SIM_TO_BENCH}
TAG_TO_SIM = {k.replace("/", "_"): k for k in SIM_TO_BENCH}  # collect.py uses folder names
bench_to_sim = {v: k for k, v in SIM_TO_BENCH.items()}

models = []
for b in bench["models"]:
    scores = {k: v for k, v in b["scores"].items() if isinstance(v, dict) and isinstance(v.get("value"), (int, float))}
    sim_id = bench_to_sim.get(b["id"], b["id"])
    name = b["name"] + (f" · {QUANT[sim_id]}" if sim_id in QUANT and QUANT[sim_id] not in b["name"] else "")
    models.append({"sim": sim_id, "bench": b["id"], "name": name, "total_params_b": b.get("total_params_b"),
                   "active_params_b": b.get("active_params_b"), "context_k": b.get("context_k"),
                   "license": b.get("license"), "scores": scores})

data = {
    "generated": datetime.date.today().isoformat(),
    "gpus": sim["gpus"],
    "combos": sum(len(v) for v in sim["results"].values()),
    "results": {TAG_TO_SIM[k]: v for k, v in sim["results"].items() if k in TAG_TO_SIM},
    "failures": [dict(f, model=TAG_TO_SIM.get(f["model"], f["model"])) for f in sim["failures"]],
    "models": models,
    "metricLabels": {"swe_bench_verified": "SWE-bench Verified", "swe_bench_pro": "SWE-bench Pro",
                     "terminal_bench_2_0": "Terminal-Bench 2.0", "longbench_v2": "LongBench v2",
                     "aa_lcr": "AA-LCR", "browsecomp": "BrowseComp", "mmlu_pro": "MMLU-Pro",
                     "gpqa_diamond": "GPQA Diamond"},
    "metricInfo": {
        "mmlu_pro": "Multiple-choice questions (10 options) across 14 subjects, from law and medicine to economics and engineering. Measures general knowledge and reasoning. Best for chatbots and general-purpose assistants.",
        "gpqa_diamond": "198 PhD-level questions in physics, chemistry and biology, written so they cannot be solved by searching the web. Measures expert reasoning. Best for reasoning models.",
        "longbench_v2": "503 multiple-choice questions over very long texts (documents, code, dialogues, tables), from 8 thousand to 2 million words. Measures long-context understanding. Best for summarization and document Q&A.",
        "aa_lcr": "Artificial Analysis Long Context Reasoning: questions that require cross-referencing several real documents, about 100 thousand tokens in total. Best for RAG and document analysis.",
        "browsecomp": "Hard questions whose answer must be found by browsing many web pages. Measures search and tool use. Best for research agents and agentic RAG.",
        "swe_bench_verified": "500 real issues from Python projects on GitHub, human-verified: the model must write a patch that makes the tests pass. Best for agentic coding.",
        "swe_bench_pro": "Longer, more complex issues, in several languages and also on non-public repositories, so less exposed to contamination. Best for agentic coding on enterprise codebases.",
        "terminal_bench_2_0": "Real tasks to complete in a terminal: building, configuring services, debugging, processing data. Best for agents that work in the shell and DevOps.",
    },
    # Use-case presets: input/output tokens, cache hit, SLA and starting traffic. Reasonable defaults, editable in the UI.
    "usecases": {
        "chat": {"label": "Chatbot", "hint": "Multi-turn conversation: the chat history is resent every turn, so much of the prompt is already cached.",
                 "metrics": ["mmlu_pro", "gpqa_diamond"],
                 "defaults": {"isl": 4000, "osl": 500, "pct": 50, "ttft": 1000, "minSpeed": 30, "traffic": 50}},
        "rag": {"label": "Document Q&A", "hint": "RAG: question plus retrieved chunks, short answer. System prompt and popular documents partly shared.",
                "metrics": ["longbench_v2", "aa_lcr", "browsecomp"],
                "defaults": {"isl": 16000, "osl": 500, "pct": 50, "ttft": 2000, "minSpeed": 20, "traffic": 20}},
        "summary": {"label": "Document summarization", "hint": "Summaries of long documents uploaded by users: heavy prefill, little reuse across requests.",
                    "metrics": ["longbench_v2", "aa_lcr"],
                    "defaults": {"isl": 32000, "osl": 1000, "pct": 0, "ttft": 5000, "minSpeed": 20, "traffic": 5}},
        "docgen": {"label": "Document generation", "hint": "Drafting reports, long emails, documentation: short prompt, long output, decode-bound.",
                   "metrics": ["mmlu_pro", "gpqa_diamond"],
                   "defaults": {"isl": 2000, "osl": 4000, "pct": 0, "ttft": 2000, "minSpeed": 30, "traffic": 10}},
        "reasoning": {"label": "Reasoning", "hint": "Thinking models: the reasoning chain produces thousands of tokens before the answer.",
                      "metrics": ["gpqa_diamond", "mmlu_pro"],
                      "defaults": {"isl": 1000, "osl": 8000, "pct": 0, "ttft": 2000, "minSpeed": 50, "traffic": 10}},
        "completion": {"label": "Code completion", "hint": "IDE completion: a few lines per answer, very low perceived latency, open files almost always cached.",
                       "metrics": ["swe_bench_verified", "terminal_bench_2_0"],
                       "defaults": {"isl": 2000, "osl": 100, "pct": 90, "ttft": 300, "minSpeed": 50, "traffic": 100}},
        "agentic": {"label": "Agentic coding", "hint": "Agents that reread the repository and session history every turn: long context, high KV cache reuse.",
                    "metrics": ["swe_bench_verified", "swe_bench_pro", "terminal_bench_2_0"],
                    "defaults": {"isl": 64000, "osl": 2000, "pct": 90, "ttft": 8000, "minSpeed": 40, "traffic": 5}},
    },
    "benchNote": "Scores published by vendors or by NVIDIA on the model cards (collected " + bench["updated"] +
                 "). Harnesses and reasoning settings differ across labs, so cross-family comparisons are indicative. "
                 "Models without a simulation are listed by accuracy but have no curves.",
}
json.dump(data, open(out, "w"), separators=(",", ":"))
print("models", len(models), "simulated", len(data["results"]), "combos", data["combos"])
