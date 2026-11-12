# Search and extraction protocol

Version: 2026-11-12. The current 13-source list is a purposive seed set supporting
review ideas. It is not an exhaustive search, meta-analysis or PRISMA review.

## Search scope

Use ACL Anthology, ACM Digital Library, AAAI/AIES, NeurIPS/ICLR proceedings, journal
publisher pages, arXiv, NBER/AEA, and author/institutional repositories. Search engines
may locate records, but cite primary papers/author corrections rather than news summaries.
Search employment audits from 2003 onward; include earlier/later methodological work
where needed. Do not claim a complete 2026 search from these seed queries.

Example reproducible query strings:

```text
("large language model" OR LLM) AND (resume OR résumé OR hiring) AND (audit OR bias)
("resume screening" OR "job-resume matching") AND (name OR gender OR race OR education)
("correspondence audit") AND (names OR "racial perception")
("LLM judge" OR "LLM-as-a-judge") AND (validity OR reliability OR bias)
("language model") AND ("prompt formatting" OR "inference backend" OR reproducibility)
("stimuli" AND "random factor") OR ("replicating studies" AND "samples of stimuli")
```

The actual seed search on 2026-11-12 used narrower variants of these queries, then
followed original publisher/author links. Exact seed sources, read basis, access
limitations and statuses appear in [sources.md](sources.md). Record the database,
query, date, filters and export filename for each future systematic-search pass.

## Inclusion and exclusion

Include primary empirical hiring audits; primary studies of proxy validity; or
methodological papers directly relevant to evaluator validity, stimulus sampling,
prompt sensitivity or implementation reproducibility. Label peer-reviewed records
and preprints separately. Exclude news/blog summaries as scientific evidence, duplicate
preprint/published versions as separate studies, and work whose task cannot be determined.
Keep a reason log for exclusions after reading, not just keyword screening.

Check linked code repositories and corrections. In particular, the Wilson/Caliskan
author repository has a 2026 correction notice; original gender-only direction claims
must not be copied uncritically. Keep the correction with the study record.

## Extraction form

For each paper, capture:

| Field | Record |
| --- | --- |
| Bibliography/version | Authors, year, venue/status, DOI, inspected version/date, correction links |
| Task/unit | Individual scoring, ranking, retrieval, generation, or summary-mediated decision; unit sampled |
| Model/execution | Model/version, provider/backend, precision, decoding, seeds, dates and settings disclosed |
| Inputs/design | Real/synthetic source, occupations, independent profiles/names, repeats, randomized factors |
| Identity signal | Name selection/pretesting, explicit labels, pronouns, institutions, locations, interactions |
| Outcomes | Score/rank/decision, thresholds, structured-output failures/refusals, grounding measures |
| Validity checks | Human comparison, qualification manipulation, weak controls, prompt/numerical robustness |
| Analysis | Within-stimulus pairing, clustering/random effects, multiplicity, intervals and missing-data treatment |
| Finding | Direction/magnitude only for the stated task/model; uncertainty and denominator |
| Limits/transfer | What the paper does not establish for our four-family Qwen study |

Two reviewers should independently extract high-stakes direction/magnitude claims and
resolve disagreements with page/section evidence. Use the abstract for initial screening;
read full methods/results before detailed effect-size synthesis. Some seed entries are
currently abstract/metadata reads and are marked accordingly.

## Synthesis and stopping rules

Group evidence by computational task before comparing directions. Separate original
and corrected results. Do not average unlike metrics or treat repeated calls as independent
studies. Report how outcomes/thresholds and invalid-response handling differ.

Only report flow counts after deduplicating actual exports; no screened/excluded totals
are invented here. For a formal review, add export files, the exclusion log and a dated
search update before calling the review systematic. Proposed gaps are questions to test,
not established novelty claims.
  
