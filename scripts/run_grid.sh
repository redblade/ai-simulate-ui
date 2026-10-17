#!/usr/bin/env bash
# Runs the AISimulate simulation grid (aiconfigurator cli default, agg vs disagg).
# Usage: AIC=/path/to/aiconfigurator OUT=/path/runs ./run_grid.sh [parallelism]
set -u
AIC=${AIC:-aiconfigurator}
OUT=${OUT:-./runs}
P=${1:-4}
GPUS=${GPUS:-72}
# Format: model_id[:backend], default backend trtllm.
MODELS=(
  openai/gpt-oss-120b
  nvidia/Qwen3.5-397B-A17B-NVFP4
  nvidia/Qwen3.5-122B-A10B-NVFP4
  nvidia/Kimi-K2.7-Code-NVFP4
  nvidia/MiniMax-M2.7-NVFP4
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  Qwen/Qwen3-Coder-480B-A35B-Instruct
  nvidia/GLM-5.2-NVFP4
  sgl-project/DeepSeek-V4-Flash-FP8:vllm   # no TensorRT-LLM data for this model
)
SYSTEMS=(gb300 b300_sxm rtx_pro_6000_server)
ISLS=(4000 16000 64000)
OSLS=(500 2000)
PREFIX_PCT=(0 50 90)
# Use-case presets first, then the rest of the grid.
{
  # UI use-case presets: ISL OSL cache%, each also with cache 0 (Dynamo off).
  for sc in "4000 500 50" "4000 500 0" "64000 2000 90" "64000 2000 0" "16000 500 50" "16000 500 0" \
            "32000 1000 0" "2000 4000 0" "1000 8000 0" "2000 100 90" "2000 100 0"; do
    for m in "${MODELS[@]}"; do for s in "${SYSTEMS[@]}"; do echo "$m $s $sc"; done; done
  done
  for i in "${ISLS[@]}"; do for o in "${OSLS[@]}"; do for p in "${PREFIX_PCT[@]}"; do
    for m in "${MODELS[@]}"; do for s in "${SYSTEMS[@]}"; do echo "$m $s $i $o $p"; done; done
  done; done; done
} | awk '!seen[$0]++' | xargs -P "$P" -L 1 bash -c '
  m=${0%%:*}; be=trtllm; [ "$0" != "$m" ] && be=${0#*:}; s=$1; i=$2; o=$3; p=$4
  pre=$(( i * p / 100 ))
  tag=$(echo "$m" | tr "/" "_")_${s}_i${i}_o${o}_p${p}
  dir='"$OUT"'/$tag
  [ -f "$dir/.done" ] && exit 0
  mkdir -p "$dir"
  '"$AIC"' cli default --model-path "$m" --total-gpus '"$GPUS"' --system "$s" --backend "$be" \
    --isl "$i" --osl "$o" --prefix "$pre" --ttft 10000 --tpot 200 --top-n 1 --no-color \
    --save-dir "$dir" > "$dir/log.txt" 2>&1
  echo $? > "$dir/.done"
'
