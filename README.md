# Local LLM resume evaluation study

A local Python research pipeline measuring how Qwen evaluates synthetic resumes when
only applicant names change. Group labels are researcher-provided metadata. The code
reports descriptive differences; it does not infer identity or make hiring decisions.

## Quick start: no GPU or model download

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this folder:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m resume_study.cli check --backend mock
uv run --locked --extra dev python -m resume_study.cli generate
uv run --locked --extra dev python -m resume_study.cli run --backend mock --run-id example_mock
uv run --locked --extra dev python -m resume_study.cli analyze --run-id example_mock
uv run --locked --extra dev pytest -q
```

The example has 2 base resumes × 4 names × 2 jobs × 1 prompt × 2 repetitions = 32
conditions. Mock scores are fixed synthetic test values, **not Qwen research results**.
Run IDs cannot overwrite existing directories. Use a fresh ID for a new experiment.

## Real local inference

Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The model uses its official tokenizer chat
template. [Pinned model files](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).

Linux + supported NVIDIA GPU (primary backend):

```bash
uv sync --locked --extra dev --extra vllm
uv run --locked --extra vllm python -m resume_study.cli check
uv run --locked --extra vllm python -m resume_study.cli run --run-id qwen_gpu
```

Mac baseline (Transformers CPU; also works on Linux):

```bash
uv sync --locked --extra dev --extra transformers
uv run --locked --extra transformers python -m resume_study.cli check --config configs/macos.yaml
uv run --locked --extra transformers python -m resume_study.cli run --config configs/macos.yaml --limit 2 --run-id qwen_smoke
```

vLLM **can run on Apple Silicon** through the community-maintained
[vLLM Metal plugin](https://docs.vllm.ai/projects/vllm-metal/en/stable/), which uses MLX.
Upstream also documents [experimental macOS CPU support](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html?device=apple).
This repository's pinned `vllm` adapter implements Linux/CUDA; it does not yet integrate
vLLM Metal or the macOS CPU build. For this small-model study, start with the tested
Transformers CPU baseline above. GPU execution with Transformers `--device mps` is an
explicit option requiring separate validation. Adding vLLM Metal would require its own
locked environment, adapter and reproducibility checks; installing it alone does not
make the existing CUDA adapter work on a Mac.

Initial use downloads about 1 GB of model files into the Hugging Face cache. Inference
always runs locally. Backends never switch automatically. Recommended planning
budgets: 8 GB system RAM for CPU; Linux NVIDIA Ampere or newer with 8 GB VRAM for
vLLM and these short inputs. Actual memory use depends on configuration. MPS is an
explicit Transformers option (`--device mps`); it requires separate validation.

## Outputs and resume

`results/RUN_ID/` contains JSONL, CSV, a manifest, the complete condition schedule,
and hashed input/source snapshots. Invalid responses remain in the data with null
scores. Analysis writes a fresh directory under `analysis/`.

```bash
# Finish the same smoke run with identical settings and original batching:
uv run --locked --extra transformers python -m resume_study.cli run --config configs/macos.yaml --run-id qwen_smoke --resume
```

Changing inputs, code, configuration, packages, hardware, or recorded execution
settings rejects resume. Partial original batches replay; existing records remain
unchanged. `--limit` limits newly saved records, not necessarily computed requests;
use `--batch-size 1` for exactly two computed cases.

Edit `configs/experiment.yaml`, `configs/name_groups.yaml`, synthetic inputs under
`data/`, and prompt files under `templates/`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the full protocol, output dictionary, cache/offline instructions, limitations,
and GitHub publication commands. [plan.md](plan.md) records local verification status.  

## Pilot on the supplied resumes

For the user-added `resumes-and-shi/` dataset, use the separate 0–100 manifest pilot:

```bash
uv run --locked --extra transformers python -m resume_study.pilot --config configs/pilot_qwen4b_eager.yaml --run-id audit --audit-only
uv run --locked --extra transformers python -u -m resume_study.pilot --config configs/pilot_qwen4b_eager.yaml --run-id qwen4b_eager_pilot
```

This uses pinned Qwen3 4B on Mac MPS and an 80-call diagnostic subset. Download/cache
instructions, exact selection, new output schema, analysis and resume commands are in
[docs/PILOT.md](docs/PILOT.md). The supplied files and original example stay intact.
The frozen SDPA pilot and supplementary eager checks are documented separately in
[the pilot report](docs/pilot-2026-11-12.md); their outputs must not be pooled.

## Literature review and published findings

Start with the [literature-review ideas and synthesis outline](docs/literature-review/README.md),
the [13-source annotated reading list](docs/literature-review/sources.md), and
[BibTeX references](docs/literature-review/references.bib). The recommended focus is
distinguishing controlled identity-signal effects from evaluator validity and
reproducibility problems. This is a narrative-review seed set, not a completed
systematic review or a claim that the pilot demonstrates demographic bias.

The [published evidence archive](docs/pilot-artifacts/README.md) contains selected
frozen raw outputs, exact source/input snapshots, checksums and analysis tables.
Model weights, caches, environments and unrelated runs remain excluded.
  
