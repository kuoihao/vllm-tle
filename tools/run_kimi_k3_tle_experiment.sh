#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 EXPERIMENT_DIR [TOKENS] [HEADS]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$1"
TOKENS="${2:-1024}"
HEADS="${3:-2}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
BENCH="$ROOT_DIR/benchmarks/kernels/kimi_k3_tle_kda.py"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-1}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR/ncu" "$OUT_DIR/nsys" "$OUT_DIR/metrics"
git -C "$ROOT_DIR" rev-parse HEAD > "$OUT_DIR/git-commit.txt"
git -C "$ROOT_DIR" status --short > "$OUT_DIR/git-status.txt"
{
  date -u
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -m pip show flagtree torch
  ncu --version
  nsys --version
} > "$OUT_DIR/environment.txt" 2>&1 || true

run_backend() {
  local backend="$1"
  local ncu_report="$OUT_DIR/ncu/${backend}.ncu-rep"
  local nsys_report="$OUT_DIR/nsys/${backend}.nsys-rep"

  "$PYTHON_BIN" "$BENCH" \
    --backend "$backend" \
    --tokens "$TOKENS" \
    --heads "$HEADS" \
    --check \
    --output "$OUT_DIR/metrics/${backend}.json"

  if ncu --target-processes all --set basic \
      --export "$OUT_DIR/ncu/${backend}" --force-overwrite \
      "$PYTHON_BIN" "$BENCH" \
      --backend "$backend" \
      --tokens "$TOKENS" \
      --heads "$HEADS" \
      --warmup "$PROFILE_WARMUP" \
      --iterations "$PROFILE_ITERATIONS" \
      --output "$OUT_DIR/metrics/${backend}-ncu.json"; then
    touch "$OUT_DIR/ncu/${backend}.ok"
  else
    touch "$OUT_DIR/ncu/${backend}.failed"
  fi
  if [[ -f "$ncu_report" ]]; then
    ncu --import "$ncu_report" --page raw \
      > "$OUT_DIR/ncu/${backend}.txt" 2>&1 || true
  else
    printf 'NCU report was not produced: %s\n' "$ncu_report" \
      > "$OUT_DIR/ncu/${backend}.txt"
  fi

  if nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=true --output="$OUT_DIR/nsys/${backend}" \
      "$PYTHON_BIN" "$BENCH" \
      --backend "$backend" \
      --tokens "$TOKENS" \
      --heads "$HEADS" \
      --warmup "$PROFILE_WARMUP" \
      --iterations "$PROFILE_ITERATIONS" \
      --output "$OUT_DIR/metrics/${backend}-nsys.json"; then
    touch "$OUT_DIR/nsys/${backend}.ok"
  else
    touch "$OUT_DIR/nsys/${backend}.failed"
  fi
  if [[ -f "$nsys_report" ]]; then
    nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --force-export=true \
      "$nsys_report" > "$OUT_DIR/nsys/${backend}.txt" 2>&1 || true
  else
    printf 'NSYS report was not produced: %s\n' "$nsys_report" \
      > "$OUT_DIR/nsys/${backend}.txt"
  fi
}

run_backend triton
run_backend tle
