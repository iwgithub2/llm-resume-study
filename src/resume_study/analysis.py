"""Descriptive, within-stratum comparisons. Never causal conclusions."""

import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .config import Config
from .experiment import file_hash, validate_records
from .results import atomic_json, read_records, write_csv


def summary(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["parsing_status"] == "valid"]
    return {
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_parsing_failures": len(rows) - len(valid),
        "mean_score": statistics.mean(r["score"] for r in valid) if valid else None,
        **{
            f"{label}_pct_valid": 100
            * sum(r["recommendation"] == label for r in valid)
            / len(valid)
            if valid
            else None
            for label in ("shortlist", "maybe", "reject")
        },
    }


def summarize_groups(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["name_group"]].append(row)
    return [{"name_group": key, **summary(value)} for key, value in sorted(groups.items())]


def compare_names(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    strata = defaultdict(list)
    for row in rows:
        key = (row["base_resume_id"], row["job_id"], row["prompt_template_id"])
        strata[key].append(row)
    within, pairs = [], []
    for stratum, members in sorted(strata.items()):
        base = dict(zip(("base_resume_id", "job_id", "prompt_template_id"), stratum, strict=True))
        names = defaultdict(list)
        for row in members:
            names[row["name_id"]].append(row)
        for name_id, values in sorted(names.items()):
            within.append(
                {
                    **base,
                    "name_id": name_id,
                    "applicant_name": values[0]["applicant_name"],
                    "name_group": values[0]["name_group"],
                    **summary(values),
                }
            )
        for left_id, right_id in itertools.combinations(sorted(names), 2):
            left = {r["repetition_index"]: r for r in names[left_id]}
            right = {r["repetition_index"]: r for r in names[right_id]}
            common = sorted(left.keys() & right.keys())
            valid = [
                i
                for i in common
                if left[i]["parsing_status"] == right[i]["parsing_status"] == "valid"
            ]
            pairs.append(
                {
                    **base,
                    "left_name_id": left_id,
                    "right_name_id": right_id,
                    "left_group": names[left_id][0]["name_group"],
                    "right_group": names[right_id][0]["name_group"],
                    "n_matched_repetitions": len(common),
                    "n_valid_pairs": len(valid),
                    "n_failed_pairs": len(common) - len(valid),
                    "n_unmatched_repetitions": len(left.keys() ^ right.keys()),
                    "mean_score_difference_left_minus_right": statistics.mean(
                        left[i]["score"] - right[i]["score"] for i in valid
                    )
                    if valid
                    else None,
                }
            )
    return within, pairs


def analyze(directory: Path, output: Path) -> Path:
    rows = read_records(directory / "results.jsonl")
    if not rows:
        raise ValueError("No experimental results to analyze")
    manifest = json.loads((directory / "manifest.json").read_text())
    conditions = json.loads((directory / "conditions.json").read_text())
    validate_records(
        rows, conditions, manifest["run_id"], Config.model_validate(manifest["config"]).model
    )
    groups = summarize_groups(rows)
    within, paired = compare_names(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "groups.csv", groups)
    write_csv(output / "within_strata.csv", within)
    write_csv(output / "paired_names.csv", paired)
    atomic_json(
        output / "analysis.json",
        {
            "run_id": manifest["run_id"],
            "is_mock": manifest["is_mock"],
            "results_sha256": file_hash(directory / "results.jsonl"),
            "conditions_sha256": file_hash(directory / "conditions.json"),
            "run_status": manifest["status"],
            "observed_conditions": len(rows),
            "expected_conditions": manifest["expected_conditions"],
            "overall": summary(rows),
            "interpretation": "Descriptive observations only. Group aggregates may be unbalanced; "
            "use within-stratum paired comparisons. Percent denominators exclude parse failures. "
            "Names do not establish demographic identity; no causal or discrimination claim.",
        },
    )
    return output  
