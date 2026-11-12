"""Descriptive diagnostics for the manifest pilot; no demographic significance claims."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .experiment import file_hash, run_lock
from .pilot import validate_pilot_records
from .results import atomic_json, read_records, write_csv


def describe(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["parsing_status"] == "valid"]
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "mean_score": statistics.mean(r["score"] for r in valid) if valid else None,
        "min_score": min((r["score"] for r in valid), default=None),
        "max_score": max((r["score"] for r in valid), default=None),
        "n_interview_yes": sum(r["interview"] == "yes" for r in valid),
        "n_threshold_inconsistent": sum(not r["threshold_consistent"] for r in valid),
    }


def diagnostics(rows: list[dict]) -> tuple[dict, list[dict], list[dict], list[dict]]:
    by_eval = {r["evaluation_id"]: r for r in rows}
    references = [r for r in rows if r["block"] == "shared_reference" and r["repetition"] == 1]
    by_baseline = {r["baseline_id"]: r for r in references}
    qualification = []
    for high in references:
        if high["qualification"] != "HIGH":
            continue
        low = by_baseline.get(high["baseline_id"].removesuffix("HIGH") + "LOW")
        if low:
            valid = high["parsing_status"] == low["parsing_status"] == "valid"
            qualification.append(
                {
                    "job_family": high["job_family"],
                    "profile": high["profile"],
                    "high_score": high["score"],
                    "low_score": low["score"],
                    "difference_high_minus_low": high["score"] - low["score"] if valid else None,
                    "high_interview": high["interview"],
                    "low_interview": low["interview"],
                }
            )
    repeats = []
    for first in references:
        other = by_eval.get(first["version_id"] + "__R2")
        if other:
            valid = first["parsing_status"] == other["parsing_status"] == "valid"
            repeats.append(
                {
                    "baseline_id": first["baseline_id"],
                    "both_valid": valid,
                    "same_raw_response": first["raw_response"] == other["raw_response"],
                    "same_score": first["score"] == other["score"] if valid else None,
                    "same_interview": first["interview"] == other["interview"] if valid else None,
                }
            )
    contrasts = []
    for row in rows:
        if row["repetition"] != 1 or row["block"] == "shared_reference":
            continue
        reference = by_baseline.get(row["baseline_id"])
        if reference:
            valid = row["parsing_status"] == reference["parsing_status"] == "valid"
            contrasts.append(
                {
                    "baseline_id": row["baseline_id"],
                    "job_family": row["job_family"],
                    "block": row["block"],
                    "treatment_id": row["treatment_id"],
                    "score": row["score"],
                    "reference_score": reference["score"],
                    "difference_from_reference": row["score"] - reference["score"]
                    if valid
                    else None,
                    "interview_changed": row["interview"] != reference["interview"]
                    if valid
                    else None,
                }
            )
    seconds = [r["inference_seconds"] for r in rows]
    summary = {
        "overall": describe(rows),
        "reference_r1": describe(references),
        "median_inference_seconds": statistics.median(seconds) if seconds else None,
        "total_inference_seconds": sum(seconds),
        "qualification_pairs": len(qualification),
        "valid_qualification_pairs": sum(
            r["difference_high_minus_low"] is not None for r in qualification
        ),
        "high_scored_above_low": sum(
            r["difference_high_minus_low"] > 0
            for r in qualification
            if r["difference_high_minus_low"] is not None
        ),
        "repeat_pairs": len(repeats),
        "valid_repeat_pairs": sum(r["both_valid"] for r in repeats),
        "repeat_score_agreements": sum(r["same_score"] is True for r in repeats),
        "repeat_raw_agreements": sum(r["same_raw_response"] for r in repeats),
    }
    return summary, qualification, repeats, contrasts


def analyze_pilot(directory: Path, output: Path) -> Path:
    with run_lock(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        conditions = json.loads((directory / "conditions.json").read_text())
        rows = read_records(directory / "results.jsonl")
        if not rows:
            raise ValueError("No pilot records to analyze")
        validate_pilot_records(rows, conditions, manifest)
        summary, qualification, repeats, contrasts = diagnostics(rows)
        summary.update(
            run_id=manifest["run_id"],
            config=manifest["config"],
            run_status=manifest["status"],
            expected_conditions=manifest["expected_conditions"],
            dataset_audit=manifest["dataset_audit"],
            results_sha256=file_hash(directory / "results.jsonl"),
            notes="Feasibility/calibration pilot; HIGH/LOW are intended manipulations, not validated "
            "hiring labels. Selected names/schools/towns do not establish demographic or causal effects.",
        )
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "summary.json", summary)
    write_csv(output / "qualification_pairs.csv", qualification)
    write_csv(output / "repeat_pairs.csv", repeats)
    write_csv(output / "contrasts.csv", contrasts)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["job_family"], row["qualification"], row["block"])].append(row)
    write_csv(
        output / "by_family_qualification_block.csv",
        [
            {"job_family": family, "qualification": level, "block": block, **describe(values)}
            for (family, level, block), values in sorted(groups.items())
        ],
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_pilot(args.run_dir, args.output))  
