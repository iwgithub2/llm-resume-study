# Supplied-materials Qwen 4B pilot

The user-added `resumes-and-shi/` materials at Git commit `cadd21f` define a different
task from the original example: score 0–100, yes/no interview at threshold 70, exactly
two justifications. Use the separate manifest runner below. The original `resume-study
run` command retains its 1–5 example schema and Cartesian resume/job design; it must
not be used directly on this supplied manifest.

## Frozen diagnostic design

Model: `Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Backend: Transformers 4.57.1 / torch 2.8.0,
Mac MPS, float16, SDPA attention, deterministic algorithms, one request at a time,
temperature 0, top-p 1, maximum 160 output tokens, original per-row seeds, shuffle
seed 20261112. All fields are explicit in `configs/pilot_qwen4b.yaml`.

The `pilot80` selection contains 80 calls / 64 distinct versions, 20 calls per family:

- All 16 reference resumes at repetitions 1 and 2: 32 calls.
- First named variant (`_01`) from each of the four supplied name codes, for A HIGH
  and A LOW in each of four families: 32 calls.
- GEO_01 and GEO_04 on A HIGH in every family: 8 calls.
- Lexicographically first and last university condition IDs on A HIGH in each family:
  8 calls. These are diagnostic alternatives, **not prestige extremes**. In the
  administrative family this selects Arizona and UMass, not Penn and Stanford.

Only one name per code appears in this diagnostic pilot; it cannot estimate general
demographic group effects. HIGH/LOW are intended relative manipulations, not ground
truth decisions. The full design contains 400 unique texts repeated three times, for
1,200 calls per model. Do not multiply the manifest by another three repetitions or
cross it with all four jobs. Each row uses only its matching job family.

Every supplied file is checksum-verified and checked against its baseline with only
the explicit NAME, LOCATION, UNIVERSITY placeholders expanded. Original materials are
never modified. Pilot prompts follow the supplied rubric, contain no scored example,
and request at most 20 whitespace-delimited words per justification. The neutral pilot
system prompt differs from the original example's explicit instruction to ignore names;
results from those prompt conditions should not be pooled.

## Commands

Run from the repository root:

```bash
uv sync --locked --extra dev --extra transformers
export HF_HOME="$PWD/.cache/huggingface"
uv run --locked --extra transformers python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554", allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja"])'
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

uv run --locked --extra transformers python -m resume_study.pilot --run-id audit --audit-only
uv run --locked --extra transformers python -u -m resume_study.pilot --run-id qwen4b_pilot_20261112
uv run --locked --extra transformers python -m resume_study.pilot_analysis --run-dir results/qwen4b_pilot_20261112 --output analysis/qwen4b_pilot_20261112
```

Choose a fresh run ID/output directory when reproducing. To continue a genuinely
interrupted pilot, use the same config and ID with `--resume`; source/settings/input
hashes and the environment must match. `--limit N` stops after N newly saved calls.
No automatic device fallback or response repair occurs. MPS failure requires a new
explicit CPU/float32 configuration and a new run ID. For a full study, copy the config,
change `selection` to `all`, and freeze that new condition before collecting data.

The first exploratory two-case device checks used longer reasons and are excluded
from the frozen pilot. One also exposed Transformers' automatic restoration of
model-specific sampling defaults. The adapter now passes `use_model_defaults=False`,
records that setting, and has a regression test with a tiny local Qwen3 model proving
that a requested greedy run remains greedy even when the checkpoint defaults sample.

## Outputs and interpretation

`results/RUN_ID/` contains raw JSONL/CSV, exact condition messages, configuration,
source/input snapshots, package/hardware/settings hashes and a run manifest. Each row
retains the original evaluation/version/baseline IDs, treatment code, seed and repetition,
plus decoded text, strict parse result, threshold-consistency flag and measured inference
seconds. `condition_id` is the unique evaluation ID; the original reused condition code
is preserved as `treatment_id`. MPS live and driver allocation are measured after each
response; these are observations, not guaranteed peak process memory. The response-token
count is obtained by re-tokenizing decoded text, not an exact generation token trace.

The parser accepts exactly integer score 0–100, interview string yes/no and two nonempty
justification strings. Score/decision disagreement is retained and flagged separately.
Reason word counts and factual accuracy require separate inspection; syntax validity
does not establish correct evaluation. No invalid responses are silently retried.

Analysis outputs: `summary.json`, eight matched HIGH/LOW reference comparisons,
16 repeated-reference comparisons, 48 within-baseline treatment/reference contrasts,
and family/qualification/block summaries. Valid-pair denominators accompany comparison
counts. Only one repetition of the reference enters HIGH/LOW comparisons; repeated
copies are not independent applicants. Contrasts are descriptive diagnostics, without
significance tests, demographic interpretations or causal claims.

The dated pilot report records observed results and study-design recommendations.

## Follow-up after the attention comparison

The frozen SDPA run is retained for reproduction. Paired checks showed material output
changes with eager attention; use the explicit follow-up config for the next pilot:

```bash
uv run --locked --extra transformers python -u -m resume_study.pilot --config configs/pilot_qwen4b_eager.yaml --run-id qwen4b_eager_pilot
```

This schedules another 80-case condition; it has not been substituted for the primary
run. Supplementary eager checks cover the complete 16-reference subset and two extra
negative controls, not a full 80-case repeat. Their scripts, raw outputs, message hashes
and settings are retained under `results/qwen4b_attention_check/` and
`results/qwen4b_eager_reference_check/`. Those checks are exploratory, separate from
the primary SDPA results. See [the dated report](pilot-2026-11-12.md).
  
