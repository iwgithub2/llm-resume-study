# Synthetic resume audit materials

## Contents

- `evaluation_resumes/`: all 1,200 evaluation resume files, organized by repetition and job family. Names, locations and universities are filled in.
- `evaluation_manifest.jsonl`: condition labels, repetition seeds and file checksums for each evaluation.
- `resumes/`: the 16 original baseline templates.
- `jobs/`: four job descriptions with the simple 40/35/25 rubric.

There are 400 distinct resume texts evaluated three times, producing 1,200 files. Repetitions intentionally reuse identical resume text with different seeds. The variants comprise 256 name versions, 64 geography alternatives, 64 university alternatives and 16 shared references.

These are the approved existing materials, before any proposed further weakening of lower-qualification profiles. HIGH/LOW labels indicate intended relative qualifications, not validated hiring decisions.

The screening task requests one model-reported 0–100 score, a yes/no interview recommendation and exactly two short justifications per resume. The interview threshold is 70. All applicants and employers are fictional. Model weights, environments, caches and results are excluded.
