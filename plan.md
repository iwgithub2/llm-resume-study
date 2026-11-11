# Plan: Local LLM resume study

Updated: 2026-11-10

## Goal

Deliver a reproducible local name-substitution research pipeline and synthetic examples.

## Status

done — initial repository published to `iwgithub2/llm-resume-study` on `main`.

## Now

- Delivered strict slot substitution, three explicit backends, locked dependencies,
  resumable factorial experiments, provenance snapshots, paired analysis and documentation.

## Next

1. Optional: validate primary vLLM backend on the intended Linux NVIDIA host.
2. Researcher: choose experimental name groups and review the study protocol.
3. Optional: add a separately pinned vLLM Metal adapter if the study requires vLLM on Mac.

## Decisions

- Python 3.12; vLLM primary, Transformers explicit fallback; no paid APIs.
- Pin model and dependencies; JSONL is the durable source, CSV is regenerated.
- Never infer demographics; synthetic group labels are researcher metadata.
- Fixed original batch replay preserves resume scheduling; invalid parses count as completed.
- Read-only verifier/adversarial reviews completed; findings fixed with regressions.
- User authorized committing and pushing to `git@github.com:iwgithub2/llm-resume-study.git`.
- Added `configs/macos.yaml` for the tested Transformers CPU baseline. Current upstream
  vLLM Metal supports Apple Silicon, but is not integrated into the pinned CUDA adapter.

## Verify

- `uv sync --locked --extra dev --extra transformers` — installed Python 3.12.12,
  torch 2.8.0 and Transformers 4.57.1; full `uv.lock` includes the vLLM extra.
- `uv run --locked --extra dev --extra transformers pytest -q` — 72 passed.
- `uv run --locked --extra dev --extra transformers ruff check src tests` — passed.
- `.venv/bin/ruff format src tests` and `ruff format --check src tests` — formatted;
  workspace-required final trailing spaces applied afterward; lint/tests passed again.
- `bash -n scripts/setup.sh` — passed.
- `uv run --locked --extra transformers python -m resume_study.cli generate` — 8 variants.
- `uv run --locked --extra transformers python -m resume_study.cli run --backend mock
  --run-id example_mock --limit 2`, then same with `--resume` replacing `--limit 2`
  — 32 unique valid records, complete; paired analysis exported.
- `uv run --locked --extra transformers python -m resume_study.cli run --backend transformers
  --device cpu --batch-size 1 --limit 2 --run-id qwen_cpu_smoke` — 2 real Qwen responses,
  both valid (score 4 / shortlist); deliberately incomplete 2-of-32 smoke schedule.
- `uv run --locked --extra transformers python -m resume_study.cli analyze --run-id
  qwen_cpu_smoke` — exported analysis; labels incomplete coverage.
- Final runs used `HF_HOME=$PWD/.cache/huggingface`, `HF_HUB_OFFLINE=1`,
  `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Cache downloaded before offline mode.
- Host: macOS 27.0.1 arm64, Apple M5, 24 GiB RAM; CPU inference.
- Sandbox-specific uv cache/interpreter overrides: `UV_CACHE_DIR=/private/tmp/llm-resume-uv-cache`,
  `UV_PYTHON_INSTALL_DIR=/private/tmp/llm-resume-python`; uv 0.9.28 run outside macOS sandbox.
- Ignore rules checked for weights/cache/venv, private data, outputs and credentials.
- `uv run --locked --extra transformers python -m resume_study.cli check --config
  configs/macos.yaml` — ready; 32 conditions. Two-case `qwen_macos_config_smoke` run
  and analysis completed with the Mac config. Tests rerun: 72 passed; Ruff lint passed.
- Initial commit `16b596a` pushed successfully with `git push -u origin main` to
  `git@github.com:iwgithub2/llm-resume-study.git`; only source, synthetic inputs,
  configuration, lockfile, tests and documentation published.

## Risks / blockers

- NVIDIA/vLLM and MPS inference not executed on this CPU validation path.
- GitHub SSH push succeeded; `main` tracks `origin/main`.
- Smoke responses alone support no empirical conclusion; the small model copied the
  prompt's example reason. Treat this as execution validation only.  
