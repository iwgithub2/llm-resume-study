#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
  echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi
case "${1:-mock}" in
  mock) uv sync --locked --extra dev ;;
  transformers) uv sync --locked --extra dev --extra transformers ;;
  vllm)
    if [[ "$(uname -s)" != Linux ]]; then
      echo "vLLM setup requires Linux + NVIDIA CUDA. Use transformers on this host." >&2
      exit 2
    fi
    uv sync --locked --extra dev --extra vllm
    ;;
  *) echo "Usage: bash scripts/setup.sh [mock|transformers|vllm]" >&2; exit 2 ;;
esac  
