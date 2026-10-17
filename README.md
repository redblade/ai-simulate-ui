# Infra Planner

A web page that compares GB300 NVL72, B300 HGX NVL8 and RTX PRO 6000 by use case, open-weight model and expected traffic,
based on AISimulate (NVIDIA Dynamo) simulations at equal GPU count (72).

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
Each run takes 20 s to 4 min on one core. The full grid (9 models × 3 systems × 18 grid scenarios plus the use-case presets) takes a few hours with 4 processes.

To view the page locally, serve the folder (`python3 -m http.server`) and open `index.html`: the page loads `data.json` with `fetch`, which does not work from `file://`.

## Modelling assumptions
- Dynamo on = best of aggregated and disaggregated serving, with the selected cache hit. Dynamo off = aggregated only, cache hit 0.
- Loose simulation SLAs (TTFT 10 s, TPOT 200 ms); the UI then filters by TTFT and per-user speed.
- GPUs needed = traffic / (req/s per GPU of the best point at SLA), rounded to the purchase unit up to 288 GPUs.
- When the selected input/output combination was not simulated, the page uses the closest simulated one and says so.
