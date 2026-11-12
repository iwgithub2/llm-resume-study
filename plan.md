# Plan: Local LLM resume study

Updated: 2026-11-12

## Goal

Develop a source-backed literature-review direction and publish the pilot code/findings.

## Status

done — literature-review roadmap, pilot implementation/findings and curated evidence
published to iwgithub2/llm-resume-study on main (content commit77cdaba).

## Now

- Pulled cadd21f; preserved all supplied materials. Audited 1,200 checksums and exact
  slot invariance: 400 distinct texts, three identical repetitions each.
- Added a separate 0–100 manifest adapter and matched-job diagnostic analysis.
- SDPA: 55/80 strict-valid; 16/16 raw reference repeats identical.
- Eager: 16/16 references valid; all eight HIGH/LOW pairs separate by 3–23 points.
  All supplied references still exceed70; unrelated controls scored10/no and0/no.
- Full evidence and recommendations: docs/pilot-2026-11-12.md.
- Added four literature-review directions,13 annotated primary-source records,
  BibTeX and a search/extraction protocol; seed narrative review, not systematic.
- Curated142-member evidence ZIP preserves raw bytes/checksums and source snapshots;
  weights, caches, environments and unrelated runs excluded.

## Next

1. Read the source matrix and complete full-method extraction/search updates before
   making systematic-review or novelty claims.
2. Validate partial-credit anchors with blinded human ratings, then run the eager
   80-case subset and held-out controls using configs/pilot_qwen4b_eager.yaml;
   retain it as a separate condition and include a small eager repeat audit.
3. Add independent profiles/names rather than redundant greedy repetitions.

## Decisions

- Qwen/Qwen3-4B-Instruct-2507 at cdbee75f17c01a7cc42f958dc650907174af0554.
- Mac M5/24GiB, torch2.8.0, Transformers4.57.1, MPS float16; no API/new dependencies.
- Main: SDPA, temperature0, max160tokens, shuffle20261112, original row seeds.
- Fixed checkpoint-default merging with use_model_defaults=False; real tiny-Qwen3
  regression proves recent checkpoint defaults cannot re-enable sampling.
- Eager/SDPA outputs differ materially. No silent fallback or pooling.
- Original1–5 example remains separate from supplied0–100 task. No original edits.
- User requested committing/pushing the pilot changes, findings and review ideas to
  git@github.com:iwgithub2/llm-resume-study.git; content commit77cdaba pushed successfully.

## Verify

- uv run --locked --extra dev --extra transformers pytest -q —83passed.
- uv run --locked --extra dev --extra transformers ruff check src tests —passed.
- bash -n scripts/setup.sh —passed.
- Literature package:13 unique bibliography/source keys; local Markdown links verified.
- Evidence archive:142 unique members,141 payload hashes and ZIP integrity verified;
  original primaryJSONL hash preserved. Independent scientific and publication reviews passed.
- git push origin main —published content commit77cdaba; final status update follows.
- python -m resume_study.pilot --run-id audit --audit-only —1,200audited/80selected.
- python -m resume_study.pilot --run-id qwen4b_pilot_20261112 —80complete,
  20.80minutes inference; maximum observed MPS driver allocation11.93GiB.
- python -m resume_study.pilot_analysis --run-dir results/qwen4b_pilot_20261112
  --output analysis/qwen4b_pilot_20261112 —tables independently verified.
- Supplementary scripts and raw data: results/qwen4b_attention_check/ and
  results/qwen4b_eager_reference_check/;14newreference calls plus2earlierprobes.
- Runtime: HF_HOME=$PWD/.cache/huggingface, HF_HUB_OFFLINE=1,
  OMP_NUM_THREADS=4, MKL_NUM_THREADS=4. Full commands: docs/PILOT.md.
- Sandbox UV_CACHE_DIR=/private/tmp/llm-resume-uv-cache and
  UV_PYTHON_INSTALL_DIR=/private/tmp/llm-resume-python; UV outside macOS sandbox.

## Risks / limitations

- Eager reference check is exploratory; no full eager80run or CPUfloat32 cross-check.
- HIGH/LOW are intended relative manipulations, not validated hiring labels.
- Keep outputs separate by attention implementation and inspect grounding.
- Raw data/model cache remain ignored. Proxy codes do not establish identities;
  repeated identical inputs are not independent applicants.  
