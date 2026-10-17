# Infra Planner

Infra Planner helps choose GPU infrastructure for LLM inference. It answers one question:

> *For this use case, this model and this much traffic, which GPU system should I buy, and how many GPUs do I need?*

It compares **GB300 NVL72**, **B300 HGX NVL8** and **RTX PRO 6000** at equal GPU count (72), using performance simulated with NVIDIA's [AISimulate](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/simulation/ai-simulate/overview).

> [!IMPORTANT]
> All results are **simulated on synthetic traffic**, not measured. No real GPU and no real request ran. Use the numbers to compare systems under controlled conditions, not to predict production performance. See [Limitations](#limitations).

## Contents

1. [Quick start](#quick-start)
2. [What the page does](#what-the-page-does)
3. [How it works](#how-it-works)
4. [The simulation grid](#the-simulation-grid)
5. [Modelling assumptions](#modelling-assumptions)
6. [Limitations](#limitations)
7. [Repository layout](#repository-layout)
8. [Regenerating the data](#regenerating-the-data)
9. [Roadmap: validation (v2)](#roadmap-validation-v2)

## Quick start

The simulation results are already in `data.json`, so viewing the page needs only Python 3 and a browser:

```bash
git clone https://github.com/redblade/ai-simulate-ui.git
cd ai-simulate-ui
python3 -m http.server 8000
```

Then open <http://localhost:8000/index.html>. Opening `index.html` directly from disk does not work, because the page loads `data.json` with `fetch`.

To rerun the simulations instead, see [Regenerating the data](#regenerating-the-data).

## What the page does

**1. Pick a use case.** Each preset sets input/output tokens, prefix-cache reuse, a latency SLA and a starting traffic level. You can change all of them afterwards.

| Use case | Input / output tokens | Cache hit | Max TTFT | Min speed per user |
|---|---|---|---|---|
| Chatbot | 4k / 500 | 50% | 1 s | 30 tok/s |
| Document Q&A (RAG) | 16k / 500 | 50% | 2 s | 20 tok/s |
| Document summarization | 32k / 1k | 0% | 5 s | 20 tok/s |
| Document generation | 2k / 4k | 0% | 2 s | 30 tok/s |
| Reasoning | 1k / 8k | 0% | 2 s | 50 tok/s |
| Code completion | 2k / 100 | 90% | 0.3 s | 50 tok/s |
| Agentic coding | 64k / 2k | 90% | 8 s | 40 tok/s |

**2. Pick a model.** Open-weight models are ranked by the accuracy benchmark that fits the use case. Each benchmark has a short description on the page, and every score in `data/benchmarks.json` has a source link.

| Use case | Benchmarks offered |
|---|---|
| Chatbot, document generation | MMLU-Pro, GPQA Diamond |
| Reasoning | GPQA Diamond, MMLU-Pro |
| Document Q&A, summarization | LongBench v2, AA-LCR (plus BrowseComp for Q&A) |
| Code completion, agentic coding | SWE-bench Verified, SWE-bench Pro, Terminal-Bench 2.0 |

**3. Read the recommendation.** For each system, the page finds the most efficient serving configuration that meets the SLA. It then shows:
- the recommended system at the selected traffic, with the GPUs and purchase units (racks or nodes) it needs;
- the traffic thresholds where the recommendation changes (for example "NVL8 up to 250 req/s, then NVL72");
- the throughput/interactivity frontier of each system, and how GPU count grows with traffic;
- a matrix of all filtered models for the same scenario.

## How it works

```
run_grid.sh ──> AISimulate (aiconfigurator cli default) ──> pareto.csv per run
                                                                │
collect.py  ──> sim.json  ──┐                                   │
benchmarks.json ────────────┴──> build_data.py ──> data.json ──> index.html
```

**AISimulate** is NVIDIA's experimental simulator for Dynamo deployments. It needs no GPUs. It estimates each forward pass from kernel performance data (GEMM, attention, MoE, communication) that NVIDIA measured on each system and ships inside the Python package.

**One simulation run** takes a model, a system and a scenario. It searches tensor, expert and data parallelism, batch sizes, and aggregated versus disaggregated (prefill/decode split) serving, all within a 72-GPU budget. For each serving mode it saves the Pareto frontier of tokens/s per GPU against tokens/s per user, with TTFT and TPOT for every point. The backend is TensorRT-LLM, except DeepSeek V4 Flash, which uses vLLM because AISimulate has no TensorRT-LLM data for it.

**The page** never computes performance itself. On top of the simulated frontiers it:
1. keeps the points that meet the SLA;
2. picks the point with the highest requests/s per GPU;
3. scales that point linearly to the requested traffic and rounds up to the purchase unit.

## The simulation grid

| Dimension | Values |
|---|---|
| Systems (72 GPUs each) | GB300 NVL72: one NVLink rack<br>B300 HGX NVL8: 9 nodes of 8 over InfiniBand<br>RTX PRO 6000 server: 9 nodes of 8, PCIe |
| Models (9) | gpt-oss-120b, Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, Kimi K2.7 Code, MiniMax M2.7, Nemotron 3 Super, Qwen3-Coder-480B, GLM-5.2, DeepSeek V4 Flash. Mostly the NVFP4/FP8 checkpoints that have performance data |
| Grid scenarios (18) | Input 4k / 16k / 64k × output 500 / 2k × cache 0 / 50 / 90% |
| Extra preset scenarios (5) | 1k/8k, 2k/4k, 32k/1k, and 2k/100 at 90% and 0% cache |
| Total | 9 × 3 × 23 = **621 runs**, of which 557 produced results |

The 64 runs without results are almost all on RTX PRO 6000. There AISimulate has no data for GLM-5.2 and DeepSeek V4 Flash, and a few long-context scenarios don't fit for Kimi, Qwen3-Coder and Qwen3.5. The page shows them as "cannot simulate".

## Modelling assumptions

| Topic | Assumption |
|---|---|
| Dynamo on | Best of aggregated and disaggregated serving, with the selected cache hit assumed to be realized by Dynamo's KV-aware router |
| Dynamo off | Aggregated serving only, no cache reuse across replicas (round-robin routing) |
| Simulation SLA | Loose (TTFT 10 s, TPOT 200 ms), so the full frontier is kept; the page then filters by your SLA |
| Scaling with traffic | GPUs needed = traffic / requests/s per GPU of the best point at SLA |
| Purchase units | Rounded up to racks of 72 (NVL72) or nodes of 8 (NVL8, RTX PRO 6000) up to 288 GPUs; beyond that, throughput per GPU decides |
| Missing combinations | If the selected input/output pair was not simulated, the page uses the closest simulated one and says so |

## Limitations

- **Synthetic traffic.** Every request in a scenario has the same input and output length. Cache reuse is a fixed share of the prompt rather than real multi-turn or agentic sessions. Real traffic has length distributions, bursts and cache eviction under memory pressure, which this version does not capture.
- **Simulated, not measured.** AISimulate is experimental, and NVIDIA positions it for evaluation, not production capacity planning.
- **Router not simulated.** The effect of Dynamo's KV-aware routing is an assumption (the cache hit you select), not simulated request by request.
- **Linear scaling.** Above 72 GPUs, GPU counts are extrapolated from the 72-GPU simulation.
- **Vendor benchmarks.** Accuracy scores are mostly vendor-reported, and harnesses differ between labs, so cross-family comparisons are indicative.

## Repository layout

| Path | Content |
|---|---|
| `index.html` | The page (single file, no build step) |
| `data.json` | Simulation frontiers, models, benchmarks and use-case presets read by the page |
| `data/benchmarks.json` | Published accuracy scores with source URLs |
| `data/benchmarks_notes.md` | Caveats on the benchmark data |
| `scripts/run_grid.sh` | Runs the simulation grid with AISimulate |
| `scripts/collect.py` | Collects each run's `pareto.csv` into one JSON |
| `scripts/build_data.py` | Builds `data.json`; holds the model mapping, use-case presets and benchmark descriptions |

## Regenerating the data

Requires Python 3.11–3.13.

```bash
python3 -m venv venv
./venv/bin/pip install --pre --only-binary=:all: aisimulate
AIC=./venv/bin/aiconfigurator OUT=./grid bash scripts/run_grid.sh 4   # 4 parallel runs
python3 scripts/collect.py ./grid sim.json
python3 scripts/build_data.py sim.json data/benchmarks.json data.json
```

- Each run takes 20 s to 4 min on one core; the full grid takes a few hours with 4 processes.
- Finished runs are skipped, so the script can be stopped and restarted.
- Then view the page as in [Quick start](#quick-start).

## Roadmap: validation (v2)

Planned next steps to check the simulated numbers against real measurements. None of this is implemented yet.

### Fixed-length scenarios against InferenceX

[InferenceX](https://inferencex.semianalysis.com) (SemiAnalysis) measures real hardware, including GB300 NVL72 and B300, on the same axes this page uses: tokens/s per GPU against tokens/s per user.

- Its fixed-length scenarios (1k/1k chat, 1k/8k reasoning, 8k/1k summarization) use random prompts with prefix caching disabled. They are a no-reuse, worst case for the cache.
- Plan: simulate the same scenarios at 0% cache for the models both projects cover, and report the gap in throughput per GPU at equal interactivity.
- Real traffic almost always reuses some prefix (system prompts, chat history, popular documents), so real deployments should match or beat those numbers. Our 50% and 90% cache scenarios are assumptions that InferenceX does not measure.

### Agentic coding calibrated on real traces

Today agentic coding is a synthetic scenario with a fixed 90% cache hit. In real agentic traffic the cache hit is an outcome: it depends on session structure, and on whether the working set fits in memory before prefixes get evicted.

1. **Calibrate a synthetic preset on real traces.** From InferenceX's public AgentX coding-agent traces (`semianalysisai/cc-traces-weka-062126` on Hugging Face), derive the effective cache hit, context growth per turn, turns per session and inter-turn pauses. Feed them into AISimulate's multi-turn synthetic sessions. The preset stays controllable and comparable with the other use cases, but its cache reuse reflects real sessions.
2. **Cross-check three ways.** Replay the real AgentX trace in AISimulate, which supports the `weka` trace format. Compare it with the calibrated preset and with InferenceX's measured AgentX results on GB300 and B300. If the three curves agree, the synthetic preset can be trusted for planning.
