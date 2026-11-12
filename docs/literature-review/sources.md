# Annotated primary-source reading list

Checked 2026-11-12. Thirteen seed studies; read basis is stated below. Publisher
records establish venue metadata where available. An abstract/metadata read is not
a substitute for full-method extraction. See [search-protocol.md](search-protocol.md).
No paper PDF is redistributed. Bibliographic records: [references.bib](references.bib).

## S01

**Marianne Bertrand and Sendhil Mullainathan (2004). Are Emily and Greg More Employable
Than Lakisha and Jamal? A Field Experiment on Labor Market Discrimination.**
*American Economic Review*,94(4),991–1013. **Peer-reviewed.**
[Publisher/abstract](https://www.aeaweb.org/articles?id=10.1257/0002828042002561).
DOI10.1257/0002828042002561. Key:`bertrand2004names`.

Controlled fictitious applications provide the foundational correspondence-audit
comparison. The paper also examines qualification-related callback differences.
Our model-score outcome is not a human employer callback, so this is a design
foundation rather than a transferable effect-size estimate. Read basis: publisher
metadata and abstract; attempted NBER access was blocked.

## S02

**Johann D. Gaebler, Sharad Goel, Aziz Huq and Prasanna Tambe (2024). Auditing the Use
of Language Models to Guide Hiring Decisions.** **Preprint record used; a peer-reviewed
venue was not verified in this search.** [arXiv2404.03086v1](https://arxiv.org/abs/2404.03086v1).
Key:`gaebler2024auditing`.

Applies correspondence-style manipulation to LLM candidate assessments using teaching
applications and discusses audit limitations. It directly motivates controlled
identity-signal interventions, but its application corpus and models differ from our
synthetic entry-level profiles. Read basis: original arXiv abstract/metadata.

## S03

**Lena Armstrong, Abbey Liu, Stephen MacNeil and Danaë Metaxa (2024). The Silicon
Ceiling: Auditing GPT's Race and Gender Biases in Hiring.** *EAAMO2024*.
**Peer-reviewed conference record.** [Author preprint](https://arxiv.org/abs/2405.04412v3),
[publisher DOI](https://doi.org/10.1145/3689904.3694699). Key:`armstrong2024ceiling`.

Separately studies resume assessment and resume generation, including named and
anonymous assessment conditions. It motivates distinguishing outcomes and considering
an anonymous control. Generation findings must not be treated as evidence that our
fixed qualifications changed. Read basis: author-preprint abstract and publisher record.

## S04 — correction-sensitive source

**Kyra Wilson and Aylin Caliskan (2024). Gender, Race, and Intersectional Bias in
Resume Screening via Language Model Retrieval.** *AIES*,7,1578–1590.
**Peer-reviewed conference paper; later author erratum.**
[Revised author paper](https://arxiv.org/html/2407.20371v3),
[DOI](https://doi.org/10.1609/aies.v7i1.31748),
[author correction notice](https://github.com/kyrawilson/Resume-Screening-Bias).
Key:`wilson2024retrieval`.

Studies embedding retrieval rather than generative rubric scoring, and investigates
name frequency and document length. The author repository's August29,2026 notice
reports that published gender-only comparison directions were inverted; it says
race-only/intersectional results were replicated and the repository code was left
uncorrected for traceability. We do not reuse original gender-direction percentages
or copy correction code without independent checks. Read basis: primary full text,
revision metadata and author README. This is a reason to track corrections, not to
generalize its retrieval findings to all LLM hiring pipelines.

## S05

**Jiafu An, Difang Huang, Chen Lin and Mingzhu Tai (2025). Measuring gender and racial
biases in large language models: Intersectional evidence from automated resume
evaluation.** *PNAS Nexus*,4(3),pgaf089. **Peer-reviewed.**
[Publisher DOI](https://doi.org/10.1093/pnasnexus/pgaf089),
[institutional record/abstract](https://hub.hku.hk/handle/10722/358864).
Key:`an2025intersectional`.

Large randomized resume-scoring experiments report heterogeneous gender/race patterns
and consider thresholded implications. This motivates separating score differences
from decision differences and avoiding one universal bias direction. It does not
validate our cutoff, small model or proxy names. Read basis: institutional primary
record/abstract; publisher full-text access failed during this search.

## S06

**Hayate Iso, Pouya Pezeshkpour, Nikita Bhutani and Estevam Hruschka (2025). Evaluating
Bias in LLMs for Job-Resume Matching: Gender, Race, and Education.** *NAACL Industry
Track*,672–683. **Peer-reviewed.**
[ACL record](https://aclanthology.org/2025.naacl-industry.55/).
DOI10.18653/v1/2025.naacl-industry.55. Key:`iso2025matching`.

Examines demographic and educational signals in job-resume matching; educational
background remains a concern in the evaluated conditions. It supports including
credentials in the review without treating school-string effects as identical to
protected-identity effects. No universal ranking of our eleven universities follows.
Read basis: official proceedings metadata/abstract.

## S07

**Huy Nghiem, Phuong-Anh Nguyen-Le, Sy-Tuyen Ho and Hal Daumé III (2026). Bias in the
Tails: How Name-conditioned Evaluative Framing in Resume Summaries Destabilizes
LLM-based Hiring.** *EMNLP2026*,36889–36933. **Peer-reviewed proceedings record.**
[ACL record](https://aclanthology.org/2026.emnlp-main.1623/).
DOI10.18653/2026.emnlp-main.1623. Key:`nghiem2026tails`.

Distinguishes factual content from evaluative framing in generated summaries and
studies a downstream LLM-judged hiring simulation. It motivates auditing explanations
and pipeline stages, not just mean scores. Our direct scorer has no summary-to-judge
stage, so the mechanism is related rather than identical. Read basis: official
proceedings metadata/abstract; detailed method extraction remains pending.

## S08

**S. Michael Gaddis (2017). How Black Are Lakisha and Jamal? Racial Perceptions from
Names Used in Correspondence Audit Studies.** *Sociological Science*,4,469–489.
**Peer-reviewed.** [Publisher](https://sociologicalscience.com/articles-v4-19-469/).
DOI10.15195/v4.a19. Key:`gaddis2017names`.

Perceived racial associations vary across names, surname combinations and naming
patterns associated with maternal education. Pretest full names rather than assuming
equivalent signals from a category label. The human survey does not establish how
Qwen perceives our particular names. Read basis: original publisher full paper.

## S09

**Charles M. Judd, Jacob Westfall and David A. Kenny (2012). Treating Stimuli as a
Random Factor in Social Psychology: A New and Comprehensive Solution to a Pervasive
but Largely Ignored Problem.** *Journal of Personality and Social Psychology*,103(1),
54–69. **Peer-reviewed.** [DOI](https://doi.org/10.1037/a0028347).
Key:`judd2012stimuli`.

Shows why stimulus variation must be accounted for when generalizing beyond selected
items and develops crossed analyses. It motivates retaining name/profile identities
and distinguishing stimuli from calls. Human participants in that paper are not
analogous to our random seeds; no automatic sample-size adequacy follows. Read basis:
original paper, accessed through an academic course-hosted copy.

## S10

**Jacob Westfall, Charles M. Judd and David A. Kenny (2015). Replicating Studies in
Which Samples of Participants Respond to Samples of Stimuli.** *Perspectives on
Psychological Science*,10(3),390–399. **Peer-reviewed.**
[Publisher](https://journals.sagepub.com/doi/10.1177/1745691614564879).
Key:`westfall2015replicating`.

Distinguishes repetition with the same stimuli from broader replication with new
stimulus samples. This motivates held-out names/profiles after exact computational
reproduction. It supplies neither an LLM-specific effect size nor a universal minimum
number of resumes. Read basis: publisher original-paper record/text.

## S11

**Lianmin Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.**
*NeurIPS36, Datasets and Benchmarks Track*. **Peer-reviewed.**
[Official proceedings](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html).
Key:`zheng2023judging`.

Examines LLM judge agreement and biases including position/verbosity effects in
chatbot evaluation. It motivates human agreement checks separate from consistency.
Reported chatbot preference agreement does not validate Qwen hiring scores, and the
paper does not establish universal self-enhancement bias. Read basis: original paper,
abstract and relevant full-text sections.

## S12

**Melanie Sclar, Yejin Choi, Yulia Tsvetkov and Alane Suhr (2024). Quantifying Language
Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start
worrying about prompt formatting.** *ICLR2024*. **Peer-reviewed.**
[Official paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/6c0e99d736da621403018ca7b32b1a4d-Paper-Conference.pdf).
Key:`sclar2024formatting`.

Meaning-preserving formatting changes affect evaluated few-shot task accuracy and
can alter comparisons. It motivates preregistered formatting robustness checks;
our pilot did not vary prompt formatting. Results on classification/multiple-choice
tasks do not imply the same effect magnitude in hiring. Read basis: original paper,
including formatting-method limitations.

## S13

**David Pape, Jonathan Evertz and Lea Schönherr (2026). The Silent Hyperparameter:
Quantifying the Impact of Inference Backends on LLM Reproducibility.**
**Preprint; no peer-reviewed venue verified.**
[arXiv2605.19537v3](https://arxiv.org/html/2605.19537v3), revised2026-11-06.
Key:`pape2026backends`.

Controlled inference-engine comparisons show that within-engine repeatability need
not imply cross-engine equivalence. Our same-Mac attention comparison is an
application-specific, smaller check—not the first demonstration of this issue.
The paper does not establish a universally correct backend. Read basis: original
full text, including scope and within-/cross-engine distinctions.
  
