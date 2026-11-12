# Reproduction protocol

The user-added `resumes-and-shi/` 0–100 study has a separate manifest pilot; see
[docs/PILOT.md](docs/PILOT.md). The protocol below documents the original 1–5 example.
Transformers attention can be selected explicitly with `model.attention_implementation`
(`eager` by default, or `sdpa`); keep it fixed within a run. The backend explicitly
disables checkpoint generation-default merging with `use_model_defaults=False` so
recent model defaults cannot silently re-enable sampling in a greedy condition.

## Scope and research design

Pipeline: validate YAML and synthetic inputs → fill explicit name slots → verify
canonical qualification text → build factorial conditions → shuffle once with a fixed
seed → format Qwen chats → infer locally → validate JSON → persist raw and parsed
results → describe groups and paired name conditions.

This is research infrastructure for a literature review or controlled pilot study.
It contains no empirical finding about demographic groups. Synthetic group A/B labels
carry no demographic meaning. Name-based proxies do not establish a person's race,
ethnicity, nationality, gender, or other identity. If later research assigns perceived
associations, document its external basis, uncertainty, cultural context, sampling,
and ethics review. Do not infer group labels from the model or the name in this code.

Hold qualifications, job, prompt, backend, model revision, dtype, sampling and execution
settings constant across names. Compare within each base-resume/job/prompt stratum,
matching repetition indexes. Prespecify names, groups, exclusions and comparisons.
Group aggregates can confound job/base mix if a run is incomplete. More names and
resumes improve coverage; repeated greedy outputs are not independent observations.
This implementation supplies descriptive tables, without significance tests or causal
or discrimination claims. The system prompt explicitly tells the evaluator to ignore
names: that instruction is part of the experimental treatment and may affect outcomes.

## Layout

```text
configs/                 experiment settings; researcher-provided group dictionary
data/resumes/base/       synthetic input text with explicit name slots
data/resumes/generated/  generated variants and relationship index (ignored)
data/jobs/               plain-text synthetic job descriptions
data/names/names.csv     synthetic name IDs, text and group labels
templates/              system, evaluation and starter resume templates
src/resume_study/       config, generation, conditions, inference, runner, results, analysis, CLI
tests/                  offline unit, regression and end-to-end tests
scripts/setup.sh         repository-specific environment setup
results/RUN_ID/          immutable-design run outputs and snapshots (ignored)
analysis/RUN_ID_SUFFIX/  derived descriptive tables (ignored)
uv.lock                 full dependency resolution for core and optional backends
```

## System and installation

Python 3.12 only; tested interpreter and OS are recorded in `plan.md`. CPU and mock
workflows target Linux and Apple Silicon macOS. vLLM here targets Linux with NVIDIA
CUDA; native Windows/vLLM and Intel macOS are not validated targets. Use Linux or WSL2
with a supported GPU/driver if applicable. Check the pinned
[vLLM GPU installation requirements](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html).

For a Mac, use `configs/macos.yaml`: it keeps the same pinned Qwen checkpoint and
experimental design, selects Transformers CPU explicitly, and uses batch size 1.

```bash
uv sync --locked --extra dev --extra transformers
uv run --locked --extra transformers python -m resume_study.cli check --config configs/macos.yaml
uv run --locked --extra transformers python -m resume_study.cli run --config configs/macos.yaml --limit 2 --run-id qwen_mac_smoke
uv run --locked --extra transformers python -m resume_study.cli analyze --config configs/macos.yaml --run-id qwen_mac_smoke
```

For the full experiment omit `--limit 2`, using a new run ID; to finish that same smoke
run use its ID plus `--resume` and the same config. Transformers MPS can be selected
explicitly with `--device mps`, but its kernels must be validated independently.

vLLM is not categorically unavailable on macOS. The current
[vLLM Metal plugin](https://docs.vllm.ai/projects/vllm-metal/en/stable/) supports Apple
Silicon GPU execution using MLX; upstream also documents
[experimental macOS CPU builds](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html?device=apple).
Neither is integrated into this repository's pinned vLLM 0.11.0 CUDA adapter. Use a
separately pinned environment and a new explicit backend adapter before collecting
Metal-based study results. Record that backend as a separate experimental condition;
do not merge its outputs with CUDA or Transformers runs under the same run ID.
The tested Transformers CPU path is the recommended baseline for this repository on Mac.

Plan for at least 8 GB host RAM and 2 GB cache/disk space for Transformers CPU, with
additional space for Python packages and results. An 8 GB NVIDIA Ampere-or-newer GPU
is a practical recommendation for these short prompts, not a measured minimum.
float32 is the shared default for portability; float16/bfloat16 are configurable GPU
conditions that must use new run IDs. MPS float32/float16 are supported explicitly by
the wrapper but require testing on the intended hardware. No silent CPU/MPS/CUDA fallback.

From a fresh clone, choose ONE environment:

```bash
# Lightweight, fully offline-testable pipeline:
uv sync --locked --extra dev
# Transformers CPU/MPS/CUDA:
uv sync --locked --extra dev --extra transformers
# Primary Linux NVIDIA backend:
uv sync --locked --extra dev --extra vllm
```

Alternatively `bash scripts/setup.sh mock|transformers|vllm` runs the selected setup.
`uv.lock` is the authoritative transitive lock; do not regenerate it during replication.
Direct pins: Jinja2 3.1.6, PyYAML 6.0.3, pydantic 2.12.3, torch 2.8.0,
Transformers 4.57.1, huggingface-hub 0.36.0, vLLM 0.11.0, pytest 8.4.2, Ruff 0.14.3.
The pinned [vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)
require torch 2.8.0. `requirements.txt` provides core/test direct pins for inspection;
prefer `uv sync --locked` over independently resolving that file.

## Model cache and revision

Exact model: `Qwen/Qwen2.5-0.5B-Instruct` at commit
`7ae557604adf67be50417f59c2c2f167def9a775`. Both model and tokenizer load from a local
snapshot of that revision, using safetensors, with remote code disabled. Model license
and usage information are in the [official model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
No inference API is used. First download needs internet access; subsequent runs reuse
the Hugging Face cache. Avoid sharing a cache containing private credentials.

```bash
export HF_HOME="$PWD/.cache/huggingface"
uv run --locked --extra transformers python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775", allow_patterns=["*.json", "*.txt", "*.safetensors"])'
# After downloading, disable all Hub network access:
export HF_HUB_OFFLINE=1
```

Also set `model.local_files_only: true` in your YAML **before starting** an offline
study. vLLM is an offline engine even during online-enabled model acquisition. A cache
miss gives an actionable error; it never downloads a different model or revision.
The model is approximately 988 MB of weights plus tokenizer/config files at the pinned
[revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775).

## Configuration and inputs

`--config PATH` is accepted after every subcommand; default `configs/experiment.yaml`.
`root` is relative to the YAML's directory; all input globs and output paths are
relative to that root and must remain inside it. Run from the repository root for
default paths. Every unknown YAML field, invalid enum/range and missing input is rejected.
Input discovery sorts project-relative paths. File paths and contents contribute to
stable IDs, so renaming a source file creates a different condition.

Important settings:

| Setting | Meaning |
| --- | --- |
| `inputs.resumes`, `inputs.jobs` | Globs; every matched base crosses every matched job |
| `inputs.names`, `inputs.name_groups` | CSV and YAML metadata paths |
| `prompts.system`, `prompts.evaluations` | Shared system text and list of evaluation templates |
| `model.name`, `model.revision` | Model ID and immutable 40-character revision hash |
| `model.backend` | `vllm`, `transformers`, or explicitly synthetic `mock` |
| `model.device`, `model.dtype` | Explicit hardware and arithmetic precision |
| `model.temperature`, `model.top_p`, `model.max_new_tokens` | Sampling controls; default 0.0, 1.0, 256 |
| `model.seed` | Base inference seed, default 42 |
| `model.max_model_len` | Prompt plus response token budget; no silent truncation |
| `model.gpu_memory_utilization` | vLLM memory fraction, default 0.7 |
| `model.local_files_only` | Require a cached snapshot |
| `experiment.repetitions` | Repeats per full base/name/job/prompt combination |
| `experiment.order_seed` | Deterministic shuffle seed, default 42 |
| `experiment.batch_size` | Fixed original batch grouping, default 4 |
| `outputs.*` | Generated text, run, and analysis directories |

Add synthetic resumes as `.txt` under `data/resumes/base/`. The only variable field
allowed is literal `{{ applicant_name }}`; it may occur multiple times. No Jinja filters,
calls, control flow, comments or whitespace-control syntax. Missing/unknown slots fail.
The renderer verifies literal segments and reconstructs a canonical placeholder from
known slot positions. Names appearing in unrelated literal text remain unchanged.
UTF-8 text is used; Python normalizes text-file newline sequences when reading.
Each variant records its base ID and canonical hash. Copy `templates/resume_template.txt`
to start a new base, then replace bracketed synthetic qualification text.

Add job `.txt` files under `data/jobs/`; empty files fail. Change globs to choose a
subset. Add names with unique `name_id`, unique nonempty single-line `applicant_name`,
and an existing `group_label` in `data/names/names.csv`. Names are not preselected by
group during inference. Add group definitions to `configs/name_groups.yaml`; no group
label or name ID is supplied to the model. Multiple names per group are supported.

Edit prompts as experimental inputs before a new run. Evaluation templates require
exactly the two direct variables `resume` and `job_description`; all surrounding text
is fixed. Add another template path to run a separate prompt condition. System text,
rubric and generation settings stay shared across groups. The model's official chat
template is applied consistently by both backends; its hash is recorded. Prompts and
resumes are not truncated, redacted or rewritten by inference.

## Commands and exact example reproduction

```bash
uv run --locked --extra dev python -m resume_study.cli check --backend mock
uv run --locked --extra dev python -m resume_study.cli generate
uv run --locked --extra dev pytest tests/test_generation.py::test_only_slot_changes_even_if_name_occurs_in_literal -q
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check src tests
uv run --locked --extra dev ruff format src tests

# Full synthetic end-to-end example: 32 conditions, no weights needed.
uv run --locked --extra dev python -m resume_study.cli run --backend mock --run-id example_mock
uv run --locked --extra dev python -m resume_study.cli analyze --run-id example_mock

# Real two-case CPU smoke (one computed case per batch):
uv run --locked --extra transformers python -m resume_study.cli check --backend transformers --device cpu
uv run --locked --extra transformers python -m resume_study.cli run --backend transformers --device cpu --batch-size 1 --limit 2 --run-id qwen_smoke

# Full primary GPU experiment:
uv run --locked --extra vllm python -m resume_study.cli check
uv run --locked --extra vllm python -m resume_study.cli run --run-id qwen_gpu
uv run --locked --extra vllm python -m resume_study.cli analyze --run-id qwen_gpu
```

Check validates inputs, Python packages and device availability, without loading weights
or reserving GPU memory. Its successful result cannot guarantee a later model load.
Exit status 2 indicates a configuration/dependency/device/actionable runtime problem.
Generate writes 8 example variants plus `variants.json`; rerunning identical inputs is
idempotent. Changed generated files are never overwritten; choose a new generated
directory for changed input sets. Run generates variants in memory and records them in
condition snapshots, so a separate `generate` command is optional.

Run IDs default to UTC timestamp plus random UUID suffix. A supplied existing ID is
an error unless `--resume` is supplied. Repeat independent experiments with new IDs;
to change order or inference seed, edit YAML before starting the new run. CLI backend,
device and batch overrides are validated and included in the manifest's effective config.

```bash
# Finish qwen_smoke without changing its design:
uv run --locked --extra transformers python -m resume_study.cli run --backend transformers --device cpu --batch-size 1 --run-id qwen_smoke --resume
uv run --locked --extra transformers python -m resume_study.cli analyze --run-id qwen_smoke
```

`--limit N` caps newly saved conditions per invocation. The full design stays in the
manifest. Partially completed original batches are computed again, preserving original
request order and batching context; only previously absent condition IDs are appended.
With batch size 4, `--limit 2` may compute 4 requests and save 2. Invalid parse records
count as completed and are never automatically retried. Conditions are not duplicated.
An OS file lock prevents simultaneous writers and releases after process termination.
JSONL records are flushed and fsynced individually; CSV/manifest are atomic replacements.
If killed between writes, JSONL is authoritative and CSV is regenerated on resume.
A malformed/truncated JSONL line is an explicit stop requiring inspection: preserve
the original file before any manual recovery; the program never discards corrupt data.

Resume rejects changed effective config, raw config file, input/source/lock hashes,
recorded packages, hardware or execution environment. Original batch grouping and engine
metadata must also match. This intentionally rejects even some harmless edits. Start
a new run after changing code or dependencies. Runtime upgrade or moving machines is
not a supported resume path. Complete runs resume as a no-op for inference.

## Run files and data dictionary

| File | Content |
| --- | --- |
| `manifest.json` | Schema/run ID, UTC creation, status, effective config, source-config hash, file hashes, runtime, Git state, fingerprint, model/backend, full order, counts, engine metadata, attempt history |
| `conditions.json` | Full shuffled design, exact message contents, stable IDs and paired seeds |
| `config_source.yaml` | Exact original YAML; effective CLI overrides are in manifest config |
| `inputs/` | Exact hashed copies of selected inputs, source modules and dependency definitions |
| `results.jsonl` | Durable source: one JSON record per completed condition, including invalid parses |
| `results.csv` | Derived equivalent table; nested sampling parameters are JSON strings |
| `.run.lock` | OS lock handle; existence alone does not mean a process is running |

Every result field:

| Field | Definition |
| --- | --- |
| `run_id` | Run directory identifier |
| `condition_id` | Hash of condition metadata, inputs, prompt and model settings |
| `base_resume_id` | Base path/content hash ID |
| `generated_resume_id` | Base/name/metadata/rendered-text hash ID |
| `base_path` | Project-relative source resume path |
| `name_id` | Researcher-assigned CSV ID |
| `applicant_name` | Exact name substituted into explicit slots |
| `name_group` | Researcher-provided label, never inferred |
| `job_id`, `job_hash` | Job path/content ID and content SHA-256 |
| `prompt_template_id` | Evaluation path/content plus system prompt hash ID |
| `repetition_index` | Zero-based repetition number |
| `random_seed` | Actual inference seed for this condition (not order seed) |
| `input_hash` | SHA-256 of generated resume text |
| `canonical_hash` | SHA-256 of resume with only known slots canonicalized |
| `prompt_hash` | SHA-256 of canonical JSON-serialized system/user messages |
| `formatted_prompt_hash` | SHA-256 of actual chat-template-formatted model input; mock uses canonical message JSON |
| `model_name`, `model_revision` | Requested model ID and immutable revision (mock never loads it) |
| `backend`, `is_mock` | Actual engine; explicit synthetic marker |
| `sampling_parameters` | Temperature, top-p, token budget, disabled top-k, repetition penalty 1.0 |
| `score` | Strict integer 1–5, or null on any parse/schema error |
| `recommendation` | `shortlist`, `maybe`, `reject`, or null |
| `explanation` | Nonempty reason string, or null |
| `raw_response` | Entire unmodified decoded response, even when invalid |
| `parsing_status`, `parsing_error` | `valid`/`invalid` and validation detail or null |

JSON must contain exactly score/recommendation/reason. Markdown fences, commentary,
duplicate keys, booleans, floating-point scores, unknown labels and extra fields are
invalid. No JSON repair, score invention, extraction from surrounding prose or silent
exclusion. CSV nulls are empty cells; JSONL preserves typed nulls. Treat raw model text
as untrusted data when opening CSV in a spreadsheet; CSV is a literal archival export.

Git commit SHA is null until the repository has a commit; manifest Git status records
dirty/untracked state, with a tracked diff hash. Selected input/source snapshots and
SHA-256 hashes remain available independently of Git. Installed distribution versions,
compute-related environment flags (including attention controls), CPU/GPU details and
NVIDIA driver when available are recorded. No credentials or full environment dump is
included. Keep all run artifacts private if using private data later.

## Analysis outputs

Every analysis invocation creates a fresh output directory; it does not modify raw data.
`groups.csv` reports total/valid/failure counts, mean scores and shortlist/maybe/reject
percentages with valid-response denominators. `within_strata.csv` reports the same for
each name within base/job/prompt. `paired_names.csv` reports each name pair's mean
left-minus-right score difference using only matched, valid repetition indexes, plus
matched/valid/failed/unmatched counts. Different strata never pair with each other.
`analysis.json` records overall counts, expected vs observed coverage, run status,
mock marker, source JSONL/schedule hashes and interpretation limits. Analysis validates
stored condition/model metadata and reparses raw responses before using scores.
Empty valid sets yield null means/percentages.
Incomplete runs can be analyzed but must be labeled as incomplete and unbalanced.
No optional plotting dependency is needed; use exported CSV for later figures.

## Randomness and limits

The complete input order is sorted, then shuffled once using a private Python RNG and
`experiment.order_seed` before engine loading. Resume uses this original order. Each
base/job/prompt/repetition receives a deterministic 32-bit seed derived from the model
seed; names share that seed within the stratum (common random numbers), and distinct
repetitions can have distinct seeds. A shared seed does not imply identical token draws
after different prompts. Seeds are in every record and condition snapshot.

vLLM runs offline with V1 multiprocessing disabled, an explicit engine seed and
per-request sampling seeds. Eager mode, no prefix caching, one GPU, explicit dtype,
generation defaults and fixed batches reduce uncontrolled variability. See the official
[vLLM reproducibility guidance](https://docs.vllm.ai/en/v0.11.0/usage/reproducibility/).
Transformers holds the model loaded and evaluates one request at a time within each
batch, resetting its seed per condition. It uses eager attention and deterministic
PyTorch algorithms. Both disable top-k/repetition penalties and use the official chat
template. Greedy temperature 0 ignores top-p; Transformers records its effective
generation config. No output constraints force valid JSON, so parse failures measure
model behavior as well as prompt suitability.

There is **no promise of bit-for-bit agreement** across GPUs, CPU/MPS, backends,
package versions, attention kernels, batch sizes, OS/driver changes or different
tokenization. Deterministic algorithms can raise unsupported-kernel errors; move to an
explicitly configured new CPU run rather than weakening settings inside a run. Different
context lengths and memory pressure may require smaller batches or token budgets;
those are new experimental configurations. The small 0.5B model is a pilot instrument
with limited capabilities; do not generalize its results to all LLMs or actual hiring.

## GitHub publication

This folder is initialized locally. No initial commit or remote push is automatic when
GitHub CLI is unavailable. Ignore rules exclude caches/weights, outputs, credentials,
private data directories, and unrecognized resume/job/name input filenames. Only the
explicit synthetic examples are allowlisted. This is not a privacy guarantee: inspect
all staged changes, especially if editing an allowlisted synthetic file.

After reviewing the files, run these commands yourself (replace OWNER):

```bash
git status --short
git add .
git diff --cached --stat
git diff --cached
git commit -m "Add reproducible local resume evaluation study"
gh auth login
gh repo create OWNER/llm-resume-study --private --source=. --remote=origin
git push -u origin HEAD
```

Without `gh`, create an empty private repository in GitHub's UI, then:

```bash
git remote add origin git@github.com:OWNER/llm-resume-study.git
git push -u origin HEAD
```

No real applicant information belongs in the published repository. Review privacy and
institutional research requirements before collecting data beyond these synthetic examples.  
