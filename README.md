# Infra Planner

A web page that compares GB300 NVL72, B300 HGX NVL8 and RTX PRO 6000 by use case, open-weight model and expected traffic,
based on AISimulate (NVIDIA Dynamo) simulations at equal GPU count (72).

> **What this tool does**
>
> Infra Planner answers one question: *for this use case, this model and this much traffic, which GPU system should I buy, and how many GPUs do I need?*
>
> You pick a use case (chatbot, document Q&A, summarization, document generation, reasoning, code completion, agentic coding). The use case presets input/output tokens, prefix-cache reuse, a latency SLA (max time to first token, min tokens/s per user) and a starting traffic level, all editable. You then pick an open-weight model, ranked by the accuracy benchmark that fits the use case (for example SWE-bench for coding, LongBench for long documents).
>
> For each system the page finds the most efficient serving configuration that still meets the SLA. It then shows:
> - the recommended system at the selected traffic, with the GPUs and purchase units (racks or nodes) it needs;
> - the traffic thresholds where the recommendation changes (for example "NVL8 up to 250 req/s, then NVL72");
> - the throughput/interactivity frontier of each system and how GPU count scales with traffic;
> - a matrix of all filtered models for the same scenario.
>
> **The simulations behind it**
>
> All results are **simulated on synthetic traffic**, not measured. No real GPU and no real request ran. Every request in a scenario has the same input and output length. Cache reuse is a fixed share of the prompt rather than real multi-turn or agentic sessions. The numbers show how the systems compare under controlled conditions; they are not what a production deployment with real traffic will see.
>
> The page does not compute performance itself: every number comes from a simulation run ahead of time. The grid covers:
> - **3 systems** at 72 GPUs each: GB300 NVL72 (one NVLink rack), B300 HGX NVL8 (9 nodes of 8 over InfiniBand), RTX PRO 6000 server (9 nodes of 8, PCIe);
> - **9 open-weight models**: gpt-oss-120b, Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, Kimi K2.7 Code, MiniMax M2.7, Nemotron 3 Super, Qwen3-Coder-480B, GLM-5.2 and DeepSeek V4 Flash, mostly in the NVFP4/FP8 checkpoints that have performance data;
> - **18 grid scenarios** (input 4k/16k/64k tokens × output 500/2k × prefix cache 0/50/90%) plus the use-case presets (1k/8k, 2k/4k, 2k/100, 32k/1k), each also with cache 0 to model "Dynamo off".
>
> That is 621 runs in total (9 models × 3 systems × 23 scenarios). Some model/system pairs cannot be simulated (for example GLM-5.2 on RTX PRO 6000) because AISimulate has no performance data for them; the page shows them as "cannot simulate".
>
> **How it relies on AISimulate**
>
> [AISimulate](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/simulation/ai-simulate/overview) is NVIDIA's experimental simulator for Dynamo deployments. It needs no GPUs: it estimates each forward pass from kernel performance data that NVIDIA measured on each system (GEMM, attention, MoE, communication) and ships inside the Python package.
>
> For each model × system × scenario, `scripts/run_grid.sh` calls `aiconfigurator cli default` (the AISimulate configuration search) with a 72-GPU budget. AISimulate explores tensor/expert/data parallelism, batch sizes, and aggregated versus disaggregated (prefill/decode split) serving. For each mode it saves the Pareto frontier of tokens/s per GPU against tokens/s per user, with TTFT and TPOT for every point. The backend is TensorRT-LLM, or vLLM for DeepSeek V4 Flash, which has no TensorRT-LLM data. `collect.py` gathers those frontiers and `build_data.py` merges them with the benchmark scores into `data.json`.
>
> The page then does three things on top of the simulated frontiers:
> - it filters the points by the SLA;
> - it picks the point with the highest requests/s per GPU;
> - it scales that point linearly to the requested traffic.
>
> "Dynamo on" means the best of aggregated and disaggregated serving, with the selected cache hit assumed to be realized by Dynamo's KV-aware router. "Dynamo off" means aggregated serving with no cache reuse across replicas. The router itself is not simulated request by request.
>
> AISimulate is experimental, and NVIDIA positions it for evaluation, not production capacity planning. Treat the results as a first-order comparison between systems, to be validated against measured benchmarks.

## Files
- `index.html`, `data.json`: the page and the data it reads.
- `data/benchmarks.json`, `data/benchmarks_notes.md`: published accuracy scores, with sources and caveats.
- `scripts/run_grid.sh`: runs the simulation grid (`aiconfigurator cli default`, agg vs disagg).
- `scripts/collect.py`: collects each run's `pareto.csv` into one compact JSON.
- `scripts/build_data.py`: merges simulations and benchmarks into `data.json`. It holds the simulated model → benchmark card mapping, the use-case presets and the benchmark descriptions.

## Regenerating the data
```bash
python3 -m venv venv && ./venv/bin/pip install --pre --only-binary=:all: aisimulate   # Python 3.11-3.13
AIC=./venv/bin/aiconfigurator OUT=./grid bash scripts/run_grid.sh 4
python3 scripts/collect.py ./grid sim.json
python3 scripts/build_data.py sim.json data/benchmarks.json data.json
```
Each run takes 20 s to 4 min on one core. The full set of 621 runs takes a few hours with 4 processes; finished runs are skipped, so the script can be stopped and restarted.

To view the page locally, serve the folder (`python3 -m http.server`) and open `index.html`: the page loads `data.json` with `fetch`, which does not work from `file://`.

## Modelling assumptions
- Dynamo on = best of aggregated and disaggregated serving, with the selected cache hit. Dynamo off = aggregated only, cache hit 0.
- Loose simulation SLAs (TTFT 10 s, TPOT 200 ms); the UI then filters by TTFT and per-user speed.
- GPUs needed = traffic / (req/s per GPU of the best point at SLA), rounded to the purchase unit up to 288 GPUs.
- When the selected input/output combination was not simulated, the page uses the closest simulated one and says so.

## Validation and roadmap (v2)

Planned next steps to check the simulated numbers against real measurements. None of this is implemented yet.

**Fixed-length scenarios against InferenceX.** [InferenceX](https://inferencex.semianalysis.com) (SemiAnalysis) measures real hardware, including GB300 NVL72 and B300, on the same axes this page uses: tokens/s per GPU against tokens/s per user. Its fixed-length scenarios (1k/1k chat, 1k/8k reasoning, 8k/1k summarization) use random prompts with prefix caching disabled, so they are a no-reuse, worst case for the cache. The plan is to simulate the same scenarios at 0% cache for the models both projects cover, then report the gap in throughput per GPU at equal interactivity. Real traffic almost always reuses some prefix (system prompts, chat history, popular documents), so real deployments should match or beat these numbers; our 50% and 90% cache scenarios are assumptions that InferenceX does not measure.

**Agentic coding calibrated on real traces.** Today agentic coding is a synthetic scenario with a fixed 90% cache hit. In agentic traffic, though, the cache hit is an outcome: it depends on session structure and on whether the working set fits in memory, so prefixes get evicted under load. The v2 approach:
1. **Calibrate a synthetic preset on real traces.** From InferenceX's public AgentX coding-agent traces (`semianalysisai/cc-traces-weka-062126` on Hugging Face), derive the effective cache hit, context growth per turn, turns per session and inter-turn pauses. Feed them into AISimulate's multi-turn synthetic sessions. The preset stays controllable and comparable with the other use cases, but its cache reuse reflects real sessions.
2. **Cross-check three ways.** Replay the real AgentX trace in AISimulate (it supports the `weka` trace format). Compare it with the calibrated synthetic preset and with InferenceX's measured AgentX results on GB300 and B300. If the three curves agree, the synthetic preset can be trusted for planning.
