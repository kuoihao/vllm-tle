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

  ncu --target-processes all --set basic \
    --export "$OUT_DIR/ncu/${backend}" --force-overwrite \
    "$PYTHON_BIN" "$BENCH" \
    --backend "$backend" \
    --tokens "$TOKENS" \
    --heads "$HEADS" \
    --output "$OUT_DIR/metrics/${backend}-ncu.json"
  ncu --import "$ncu_report" --page raw \
    > "$OUT_DIR/ncu/${backend}.txt" 2>&1 || true

  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=true --output="$OUT_DIR/nsys/${backend}" \
    "$PYTHON_BIN" "$BENCH" \
    --backend "$backend" \
    --tokens "$TOKENS" \
    --heads "$HEADS" \
    --output "$OUT_DIR/metrics/${backend}-nsys.json"
  nsys stats --report cudaapisum,gpukernsum --force-export=true \
    "$nsys_report" > "$OUT_DIR/nsys/${backend}.txt" 2>&1 || true
}

run_backend triton
run_backend tle
