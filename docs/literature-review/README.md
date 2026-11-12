# Literature-review directions

Prepared 2026-11-12. This is a source-backed narrative-review plan and seed reading
list, not a completed systematic review. [Sources and limitations](sources.md),
[search protocol](search-protocol.md), and [BibTeX](references.bib) accompany it.

## Recommended central question

**When does a name-based LLM hiring audit measure differential treatment, and when
does it measure an unreliable evaluation instrument?**

Suggested title: **From Resume Audit to Evaluator Audit: Validity, Identity Signals,
and Reproducibility in LLM Hiring Evaluation.**

This fits our project better than another broad question of whether LLMs are biased.
The contribution would be a careful synthesis and reproducible local case study,
not a claim to have first discovered hiring bias or backend sensitivity.

## Four viable review angles

| Direction | Review question | Literature connection | Fit to our evidence |
| --- | --- | --- | --- |
| **1. Validity before fairness — recommended** | What checks are needed before score gaps are interpretable? | Correspondence audits [S01/S02], judge validation [S11], reproducibility [S13] | Our SDPA outputs were often invalid; eager references passed the qualification check. Human criterion validity remains untested. |
| **2. Scores versus decisions** | Can similar score differences produce different conclusions under ranking or a cutoff? | Separate assessment tasks [S03], large randomized scoring audit [S05] | All eager reference scores exceed70, despite qualification gaps of3–23. This is a ceiling/cutoff issue, not evidence of no name effects. |
| **3. What names and credentials signal** | How do studies validate demographic, educational and location proxies? | Name perception [S08], retrieval frequency/length [S04], education signals [S06] | Our name codes are labels, schools differ by family, and geography uses four Massachusetts towns. |
| **4. Explanations and downstream screening** | Can apparently reasonable explanations introduce unsupported or differently framed qualifications? | Assessment versus generation [S03], summary-mediated screening [S07] | Some SDPA explanations invented applied skills. We have not estimated a group-specific grounding-error rate. |

Directions2–4 can be sections of Direction1 rather than separate projects.

## Synthesis argument

The correspondence-study tradition supplies a useful intervention: change an identity
signal while controlling other application content. Bertrand and Mullainathan's field
experiment concerns employer callbacks, however, whereas our outcome is a model-reported
score on fictional profiles. The design analogy does not make those outcomes equivalent.
[S01](https://www.aeaweb.org/articles?id=10.1257/0002828042002561)

LLM hiring papers also study different computational tasks. Generative resume assessment,
embedding retrieval, job matching and downstream summary-based screening should be
compared as distinct mechanisms, not pooled into one universal bias direction. Occupations,
model versions, identity manipulations and outcomes must be extracted alongside results.
[S03](https://arxiv.org/abs/2405.04412), [S04](https://arxiv.org/abs/2407.20371v3),
[S05](https://doi.org/10.1093/pnasnexus/pgaf089),
[S07](https://aclanthology.org/2026.emnlp-main.1623/)

Measurement reliability and validity are separate. Repeated identical judgments establish
repeatability for selected stimuli; they do not establish correct evidence use or human
agreement. A review should therefore ask which studies verify qualifications, explanations,
format failures and external criteria before interpreting group contrasts.
[S09](https://doi.org/10.1037/a0028347),
[S10](https://doi.org/10.1177/1745691614564879),
[S11](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

Our pilot provides a motivating case rather than a demographic result: 55/80 strict-valid
SDPA outputs; 16/16 eager references valid; all eight eager HIGH/LOW pairs ordered as
intended; all 16 references above 70. These checks do not establish hiring accuracy or
demographic neutrality. [Pilot report](../pilot-2026-11-12.md)

## Proposed literature-review structure

1. **Define the decision and target construct.** Resume quality, job fit, human callback,
   similarity rank and model-reported hireability are different outcomes. State whether
   the review concerns identity-string effects, perceived identity, or deployment fairness.
2. **Correspondence audits and proxy validity.** Explain controlled substitutions and
   how name-selection/pretesting affects interpretation. Separate a named shared reference
   from a genuinely name-redacted control. Start with S01, S02 and S08.
3. **LLM hiring evidence by task.** Compare direct scoring, retrieval, matching and
   summary-mediated systems. Discuss intersectional findings without assuming the same
   sign across tasks or models. Use S03–S07; record source versions and corrections.
4. **Qualification and evidence validity.** Ask whether evaluators distinguish stronger
   from weaker profiles and ground statements in supplied evidence. Discuss how human
   annotation and explicit partial-credit anchors complement schema checks. Use S11 as
   methodological motivation, with its chatbot-to-hiring transfer limitation stated.
5. **Sensitivity and reproducibility.** Review prompts, decoding, backend/attention,
   dtype and model revisions as experimental conditions. S12 concerns prompt formatting;
   S13 concerns inference engines. Our pilot varied attention, not prompt formatting.
6. **Outcomes, missingness and generalization.** Review score distributions, thresholds,
   ranking, refusals/parse failures and units of analysis. Preserve stimulus identities;
   repeated calls do not enlarge the sampled applicant population. Use S09/S10.
7. **Synthesis and feasible follow-up.** Present the local pilot as exploratory instrument
   validation, specify what it cannot establish, and propose a held-out calibrated audit.

## Concrete research questions for a follow-up study

| Question | Proposed comparison/outcome | Current status |
| --- | --- | --- |
| RQ1: Does the evaluator respond to qualifications? | Matched HIGH–LOW score differences, blinded human anchors, negative controls | Eager 8/8 positive reference gaps; no human validation yet |
| RQ2: Does changing only a supplied name alter scores? | Within-base/job/qualification name contrasts; continuous score primary | Full eager name study not collected; no group-effect conclusion from pilot |
| RQ3: Are unusable responses condition-dependent? | Strict-schema failure, length compliance and threshold consistency by matched block | Observed SDPA missingness; needs fixed eager protocol and held-out checks |
| RQ4: Do conclusions persist across implementations? | Same messages/settings, fixed attention implementations, matched valid/failure outcomes | Four targeted comparisons plus16 reference follow-ups; exploratory, not universal |
| RQ5: Does a name-irrelevance instruction change behavior? | Preregistered neutral versus explicit-ignore prompt, retaining qualification controls | Proposed only; not tested by this pilot |

Do not add all interactions at once. First validate the instrument; then run the
name comparison with a fixed condition. Treat geography/university and mitigation
prompts as separate secondary questions unless sample size and design justify more.

## Suggested next study design

- Retain the supplied materials unchanged as the versioned pilot dataset. Obtain
  blinded ratings of job-relevant evidence and qualification strength before revising it.
- Freeze eager attention, dtype, checkpoint, prompt and output schema. Check CPU/float32
  on a small calibration subset when practical; numerical agreement is not yet established.
- Use continuous score as the primary outcome; keep the preset70 decision as a secondary
  thresholded outcome. Add independently validated borderline/weak profiles rather than
  moving the cutoff to manufacture interview differences.
- Use one greedy pass over 400 unique variants plus a small repeat audit. More independent
  profiles/names are preferable to treating the 1,200 repeated calls as new applicants.
- Keep name/profile/job identifiers in analysis; justify any crossed/clustered uncertainty
  and population claims. Do not fit a complicated random-effects model merely because
  it is available with only two A/B profiles per family.
- Annotate explanation grounding and report failures, including observations that cannot
  yield a score. Do not silently repair responses or selectively retry until valid.

## Claims to avoid

The pilot does not prove discrimination, absence of bias, actual hiring accuracy, or
universal superiority of eager attention. A name string is not verified demographic
identity. Similar greedy responses are not independent replications. Published evidence
from a different task/model is not a direct estimate for this Qwen condition. Any
claimed literature gap remains provisional until the search protocol is completed.
  
