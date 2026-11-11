from resume_study.analysis import compare_names, summary


def row(name, repetition, score=3, status="valid", base="base"):
    return dict(
        name_id=name,
        applicant_name=name,
        name_group=name,
        base_resume_id=base,
        job_id="job",
        prompt_template_id="prompt",
        repetition_index=repetition,
        score=score,
        parsing_status=status,
        recommendation="maybe",
    )


def test_paired_comparisons_match_repetitions_and_strata():
    rows = [
        row("a", 0, 5),
        row("a", 1, 1),
        row("a", 2, 3),
        row("b", 0, 2),
        row("b", 1, None, "invalid"),
        row("b", 3, 4),
        row("a", 0, 1, base="other"),
    ]
    within, pairs = compare_names(rows)
    assert len(within) == 3 and len(pairs) == 1
    assert pairs[0]["n_matched_repetitions"] == 2
    assert pairs[0]["n_valid_pairs"] == 1
    assert pairs[0]["n_failed_pairs"] == 1
    assert pairs[0]["n_unmatched_repetitions"] == 2
    assert pairs[0]["mean_score_difference_left_minus_right"] == 3


def test_failures_excluded_from_percent_denominator():
    stats = summary([row("a", 0), row("b", 0, None, "invalid")])
    assert stats["n_total"] == 2 and stats["n_valid"] == 1
    assert stats["n_parsing_failures"] == 1
    assert stats["maybe_pct_valid"] == 100
    assert summary([row("a", 0, None, "invalid")])["mean_score"] is None  
