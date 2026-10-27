# Validation against InferenceX

This folder checks Infra Planner's simulated numbers against real measurements from [InferenceX](https://inferencex.semianalysis.com) (SemiAnalysis). The measurements come from the public database dump of 14 September 2026, the same one NVIDIA uses for its own [accuracy page](https://ai-dynamo.org/aisimulate/e2e-accuracy/).

It answers two questions:

1. **How much faster is GB300 NVL72 than B300 NVL8 on real hardware?** Infra Planner's simulations say about 5%.
2. **How far off is the estimator Infra Planner uses** (`aiconfigurator cli`, shipped with AISimulate) when it is given the exact measured configuration?

## Results (dump of 14/09/2026, measurements from the last 120 days, no speculative decoding)

### 1. Measured GB300 vs B300

This compares the best measured output tokens/s per GPU at the same tokens/s per user, across all frameworks and configurations. A level is only compared when both systems were measured at or below it. Full table: [results/measured_gb300_vs_b300.txt](results/measured_gb300_vs_b300.txt).

| Tokens/s per user | Model/workload pairs | GB300 / B300, median | Range |
|---|---|---|---|
| ≥ 50 | 22 | **1.76×** | 0.64× – 17×  |
| ≥ 100 | 21 | **1.63×** | 0.51× – 7.7× |
| ≥ 200 | 5 | 1.48× | 0.55× – 3.5× |

- The gain is concentrated on large MoE models: DeepSeek R1, Kimi K2.5 and Qwen3.5-397B are 3–7× faster. MiniMax M2.5/M3 are about even.
- **Caveat.** This compares the best configurations each system was measured with. Every GB300 measurement is a disaggregated, multi-node Dynamo recipe. Most B300 measurements are single-node and aggregated. Part of the gap may therefore come from recipe tuning rather than hardware.

### 2. Estimator error on the measured configurations

Error = predicted / measured − 1. Full output: [results/estimator_vs_measured.txt](results/estimator_vs_measured.txt).

| System, mode | Points | Simulated | TTFT median | TPOT median | Tokens/s per GPU median | Tokens/s per GPU within ±25% |
|---|---|---|---|---|---|---|
| B300, aggregated | 444 | 67% | −23% | +11% | −30% | 43% |
| B300, disaggregated | 175 | 46% | −45% | +26% | −43% | 5% |
| GB300, disaggregated | 335 | **20%** | −70% | +33% | −41% | 9% |
| GB300, decode EP ≥ 16 | 93 | **2%** | −94% | +108% | −68% | 0% |

- **Throughput and TTFT are both underestimated.** The estimator predicts about 30–40% less throughput per GPU than measured on both systems, and a TTFT that is too low.
- **Wide expert parallelism on GB300 cannot be simulated.** 110 GB300 points fail with "Cross-node EP requires DeepEP A2A data": AISimulate has no communication data for experts spread across trays. Those are exactly the configurations where GB300 shows its advantage.
- **Where both can be simulated, the errors are similar.** Small-EP disaggregated configurations show −41% on GB300 and −43% on B300, so the relative comparison holds there.
- **RTX PRO 6000.** InferenceX has only 16 points (Qwen3.5 FP4 on SGLang). AISimulate has no SGLang MoE data for that model on this GPU, so none could be simulated.

### What this means for Infra Planner

The simulated NVL72 vs NVL8 gap (about 1.05×) understates the measured one (about 1.6–1.8× median, and more for large MoE models). The reason is that the simulation cannot model wide expert parallelism across the NVL72 rack. Until it can, results for large MoE models should not be used to choose between NVL72 and NVL8.

## Reproducing

Requires about 25 GB of download (streamed, not stored), `zstd`, `pg_restore` 17 or newer, and the `aisimulate` wheel used for the planner.

```bash
cd validation
PG_RESTORE=/path/to/pg_restore OUT=tables.copy ./fetch_dump.sh             # ~20 min
python3 parse_dump.py tables.copy tables.json
python3 measured_compare.py tables.json gb300 b300 120 > results/measured_gb300_vs_b300.txt
../venv/bin/python run_validation.py tables.json results/estimator_vs_measured.jsonl gb300 b300   # ~10 min, 4 cores
python3 analyze.py results/estimator_vs_measured.jsonl > results/estimator_vs_measured.txt
```

| File | Content |
|---|---|
| `fetch_dump.sh` | Streams the dump and extracts the `configs`, `benchmark_results` and `workflow_runs` tables |
| `parse_dump.py` | Converts the extracted tables to JSON |
| `measured_compare.py` | Measured throughput per GPU of two systems at equal interactivity |
| `run_validation.py` | Runs the estimator on each measured configuration, using NVIDIA's InferenceX adapter from the wheel |
| `analyze.py` | Coverage and error statistics |
| `results/` | Outputs of the run above |
