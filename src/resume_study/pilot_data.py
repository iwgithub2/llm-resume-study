"""Validate supplied manifest materials and select a fixed diagnostic pilot."""

import json
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from .config import Model, StrictModel, digest, local_path
from .experiment import file_hash
from .generate_resumes import ENV
from .results import unique_object


class PilotConfig(StrictModel):
    root: str = ".."
    dataset: str = "resumes-and-shi"
    selection: Literal["pilot80", "all"] = "pilot80"
    order_seed: int = 20261112
    system_prompt: str = "templates/pilot_system.txt"
    evaluation_prompt: str = "templates/pilot_evaluation.j2"
    model: Model


class SuppliedRow(StrictModel):
    evaluation_id: str
    version_id: str
    baseline_id: str
    job_family: str
    condition_id: str
    block: Literal["names", "geography", "university", "shared_reference"]
    name: str
    location: str
    university: str
    repetition: int = Field(ge=1)
    suggested_seed: int = Field(ge=0, le=2**32 - 1)
    resume_file: str
    resume_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_pilot_config(path: Path) -> tuple[PilotConfig, Path]:
    cfg = PilotConfig.model_validate(yaml.safe_load(path.read_text()))
    if cfg.model.max_new_tokens >= cfg.model.max_model_len:
        raise ValueError("max_new_tokens must be less than max_model_len")
    return cfg, (path.resolve().parent / cfg.root).resolve()


def audit_dataset(dataset: Path) -> tuple[list[dict], dict]:
    rows = []
    ids = set()
    version_texts = {}
    for index, line in enumerate(
        (dataset / "evaluation_manifest.jsonl").read_text().split("\n"), 1
    ):
        if not line.strip():
            continue
        try:
            row = SuppliedRow.model_validate(
                json.loads(line, object_pairs_hook=unique_object)
            ).model_dump()
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"Invalid input manifest line {index}: {exc}") from exc
        if row["evaluation_id"] in ids:
            raise ValueError(f"Duplicate evaluation_id: {row['evaluation_id']}")
        ids.add(row["evaluation_id"])
        path = local_path(dataset, row["resume_file"])
        if file_hash(path) != row["resume_sha256"]:
            raise ValueError(f"Resume checksum mismatch: {row['evaluation_id']}")
        source = local_path(dataset, f"resumes/{row['baseline_id']}.txt").read_text()
        for slot, value in (
            ("NAME", row["name"]),
            ("LOCATION", row["location"]),
            ("UNIVERSITY", row["university"]),
        ):
            source = source.replace("{{" + slot + "}}", value)
        if source != path.read_text():
            raise ValueError(f"Non-slot content differs from baseline: {row['evaluation_id']}")
        previous = version_texts.setdefault(row["version_id"], row["resume_sha256"])
        if previous != row["resume_sha256"]:
            raise ValueError(f"Repetitions changed resume text: {row['version_id']}")
        rows.append(row)
    if not rows:
        raise ValueError("Empty supplied evaluation manifest")
    return rows, {
        "evaluations": len(rows),
        "versions": len(version_texts),
        "distinct_texts": len(set(version_texts.values())),
        "baselines": len({r["baseline_id"] for r in rows}),
        "job_families": sorted({r["job_family"] for r in rows}),
        "checksums_and_slot_invariance": "passed",
    }


def select_pilot(rows: list[dict], mode: str) -> list[dict]:
    if mode == "all":
        return sorted(rows, key=lambda r: r["evaluation_id"])
    selected = []
    # 16 references x 2 repetitions; 32 name cases on A HIGH/LOW across four jobs;
    # 8 geography + 8 university cases on A HIGH. Total 80, 64 distinct versions.
    for row in rows:
        base = row["baseline_id"]
        if row["block"] == "shared_reference" and row["repetition"] in (1, 2):
            selected.append(row)
        elif row["repetition"] == 1 and "_A_" in base:
            if row["block"] == "names" and row["condition_id"].endswith("_01"):
                selected.append(row)
            elif base.endswith("_HIGH"):
                if row["condition_id"] in {"GEO_01", "GEO_04"}:
                    selected.append(row)
                elif row["block"] == "university":
                    choices = sorted(
                        {
                            r["condition_id"]
                            for r in rows
                            if r["baseline_id"] == base and r["block"] == "university"
                        }
                    )
                    if row["condition_id"] in {choices[0], choices[-1]}:
                        selected.append(row)
    selected.sort(key=lambda r: r["evaluation_id"])
    if len(selected) != 80 or len({r["version_id"] for r in selected}) != 64:
        raise ValueError("pilot80 requires the expected 16-base/four-family supplied design")
    return selected


def build_pilot(cfg: PilotConfig, root: Path) -> tuple[list[dict], list[Path], dict]:
    dataset = local_path(root, cfg.dataset)
    rows, audit = audit_dataset(dataset)
    rows = select_pilot(rows, cfg.selection)
    system_path = local_path(root, cfg.system_prompt)
    template_path = local_path(root, cfg.evaluation_prompt)
    template = ENV.from_string(template_path.read_text())
    system = system_path.read_text()
    inputs = {dataset / "evaluation_manifest.jsonl", system_path, template_path}
    conditions = []
    for row in rows:
        resume_path = local_path(dataset, row["resume_file"])
        job_path = local_path(dataset, f"jobs/{row['job_family']}.txt")
        baseline_path = local_path(dataset, f"resumes/{row['baseline_id']}.txt")
        inputs.update((resume_path, job_path, baseline_path))
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": template.render(
                    resume=resume_path.read_text(), job_description=job_path.read_text()
                ),
            },
        ]
        fields = {
            **row,
            "treatment_id": row["condition_id"],
            "condition_id": row["evaluation_id"],
            "qualification": row["baseline_id"].rsplit("_", 1)[1],
            "profile": row["baseline_id"].split("_")[1],
            "job_hash": file_hash(job_path),
            "prompt_hash": digest(messages),
            "messages": messages,
        }
        conditions.append(fields)
    random.Random(cfg.order_seed).shuffle(conditions)
    audit.update(
        selected_evaluations=len(conditions), selected_versions=len({r["version_id"] for r in rows})
    )
    return conditions, sorted(inputs), audit  
