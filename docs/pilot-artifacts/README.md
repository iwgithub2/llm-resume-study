# Published pilot evidence

[Download the frozen evidence archive](pilot-2026-11-12.zip).
It contains 141 payload files plus an internal `SHA256SUMS` index. Raw files retain
their original bytes, including invalid responses; no model scores were repaired.
The ZIP is 405,253 bytes. Its SHA-256 is:

```text
434791307ce2847f7f5ddf0304d4f4b68c48f62b886772a02f12f1f6b1f875cd
```

## Contents

| Folder | Evidence |
| --- | --- |
| `primary_sdpa/` | All80 raw JSONL/CSV records, exact80-condition schedule, original manifest/config and source/input snapshots |
| `primary_sdpa/analysis/` | Summary, supplementary metrics, qualification/repeat pairs and treatment contrasts |
| `eager_attention_probe/` | Four targeted eager comparisons, two extra negative controls, provenance and diagnostic script |
| `eager_references/` | Sixteen reference records, paired HIGH/LOW table, summary/provenance and diagnostic script |

The 16 eager references include two reused rows from the four-case eager probe, with
14 additional calls. Do not count those reused records twice. The two negative
controls are out-of-dataset diagnostics. Exploratory device checks preceding the
frozen pilot are excluded. All profiles and employers in the published inputs are
synthetic, as described by the supplied dataset.

The archive excludes model weights, Hugging Face caches, virtual environments,
credentials and unrelated experiment runs. Ordinary `results/` and `analysis/`
directories remain ignored by Git; this curated archive is the explicit publication
artifact requested for the pilot findings.

Primary 80-record JSONL SHA-256, preserved exactly:

```text
f8e7653b622c09423116142e18a8e1aaa8e4e536dbef5bf287b87801db766f3b
```

## Inspect without running a model

```bash
python -m zipfile -t docs/pilot-artifacts/pilot-2026-11-12.zip
python -m zipfile -e docs/pilot-artifacts/pilot-2026-11-12.zip /tmp/resume-pilot-evidence
```

Each `SHA256SUMS` entry names an archive-relative payload file. The archive index
excludes itself. The diagnostic scripts preserve the original repository-relative
runtime paths; inspecting the frozen results requires no model or inference API.
The recorded Git baseline is `cadd21f` plus the then-uncommitted pilot implementation;
the archived source snapshots and file hashes record the actual executed code.
Use [PILOT.md](../PILOT.md) to reproduce new runs, and the
[dated report](../pilot-2026-11-12.md) for interpretation and limits. Eager and SDPA
conditions are separate, and these data establish neither population-level fairness
nor actual hiring accuracy.
  
