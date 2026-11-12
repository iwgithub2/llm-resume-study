import json
import shutil
from pathlib import Path

import pytest
import yaml

from resume_study.inference import MockBackend
from resume_study.pilot import parse_pilot, run_pilot, validate_pilot_records
from resume_study.pilot_data import audit_dataset, build_pilot, load_pilot_config
from resume_study.results import read_records

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def pilot_study(tmp_path):
    shutil.copytree(REPO / "resumes-and-shi", tmp_path / "resumes-and-shi")
    shutil.copytree(REPO / "templates", tmp_path / "templates")
    (tmp_path / "configs").mkdir()
    data = yaml.safe_load((REPO / "configs/pilot_qwen4b.yaml").read_text())
    data["model"].update(backend="mock", device="cpu")
    path = tmp_path / "configs/pilot.yaml"
    path.write_text(yaml.safe_dump(data))
    return path, tmp_path


class PilotMock(MockBackend):
    def generate(self, messages, seeds):
        return [
            json.dumps(
                {
                    "score": 75,
                    "interview": "yes",
                    "justifications": ["Evidence one.", "Evidence two."],
                }
            )
            for _ in messages
        ]


def test_supplied_dataset_and_predefined_selection():
    cfg, root = load_pilot_config(REPO / "configs/pilot_qwen4b.yaml")
    conditions, inputs, audit = build_pilot(cfg, root)
    assert audit["evaluations"] == 1200 and audit["distinct_texts"] == 400
    assert len(conditions) == 80 and len({c["version_id"] for c in conditions}) == 64
    assert len({c["condition_id"] for c in conditions}) == 80
    assert len([c for c in conditions if c["block"] == "shared_reference"]) == 32
    assert all(
        sum(c["job_family"] == family for c in conditions) == 20 for family in audit["job_families"]
    )
    assert conditions == build_pilot(cfg, root)[0]
    assert inputs


@pytest.mark.parametrize(
    "payload",
    [
        {"score": True, "interview": "yes", "justifications": ["a", "b"]},
        {"score": 101, "interview": "yes", "justifications": ["a", "b"]},
        {"score": 75, "interview": "shortlist", "justifications": ["a", "b"]},
        {"score": 75, "interview": "yes", "justifications": ["a"]},
        {"score": 75, "interview": "yes", "justifications": ["a", " "]},
    ],
)
def test_pilot_parser_does_not_repair_invalid_output(payload):
    result = parse_pilot(json.dumps(payload))
    assert result["parsing_status"] == "invalid" and result["score"] is None


def test_threshold_inconsistency_is_recorded_without_repair():
    result = parse_pilot('{"score":69,"interview":"yes","justifications":["a","b"]}')
    assert result["parsing_status"] == "valid"
    assert result["score"] == 69 and result["interview"] == "yes"
    assert result["threshold_consistent"] is False


def test_manifest_checksum_corruption_rejected(pilot_study):
    _, root = pilot_study
    dataset = root / "resumes-and-shi"
    first = json.loads((dataset / "evaluation_manifest.jsonl").read_text().split("\n")[0])
    (dataset / first["resume_file"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        audit_dataset(dataset)


def test_pilot_mock_resume_and_metadata_integrity(pilot_study):
    path, root = pilot_study
    directory = run_pilot(path, run_id="mock_pilot", limit=2, engine=PilotMock())
    original = read_records(directory / "results.jsonl")
    assert len(original) == 2
    run_pilot(path, run_id="mock_pilot", resume=True, engine=PilotMock())
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 80 and rows[:2] == original
    manifest = json.loads((directory / "manifest.json").read_text())
    conditions = json.loads((directory / "conditions.json").read_text())
    assert manifest["status"] == "complete"
    rows[0]["score"] = 99
    with pytest.raises(ValueError, match="disagree"):
        validate_pilot_records(rows, conditions, manifest)
    assert (root / "results/mock_pilot/results.csv").is_file()


def test_diagnostic_valid_pair_denominators():
    from resume_study.pilot_analysis import diagnostics

    cfg, root = load_pilot_config(REPO / "configs/pilot_qwen4b.yaml")
    conditions, _, _ = build_pilot(cfg, root)
    rows = [
        {
            **c,
            "parsing_status": "valid",
            "score": 80 if c["qualification"] == "HIGH" else 40,
            "interview": "yes" if c["qualification"] == "HIGH" else "no",
            "threshold_consistent": True,
            "raw_response": "constant",
            "inference_seconds": 1.0,
        }
        for c in conditions
    ]
    failed = next(r for r in rows if r["evaluation_id"] == "SWE_A_HIGH__REF__R1")
    failed.update(parsing_status="invalid", score=None, interview=None, threshold_consistent=None)
    summary, qualification, repeats, _ = diagnostics(rows)
    assert len(qualification) == 8 and summary["valid_qualification_pairs"] == 7
    assert summary["high_scored_above_low"] == 7
    assert len(repeats) == 16 and summary["valid_repeat_pairs"] == 15
    assert summary["repeat_score_agreements"] == 15  
