import json

import pytest

from resume_study.analysis import analyze
from resume_study.experiment import execute, run_lock
from resume_study.inference import MockBackend
from resume_study.results import read_records


class RecordingMock(MockBackend):
    def __init__(self):
        self.batches = []

    def generate(self, messages, seeds):
        self.batches.append((messages, seeds))
        return super().generate(messages, seeds)


class InvalidMock(MockBackend):
    def generate(self, messages, seeds):
        return ["broken raw response\nwith newline" for _ in messages]


def test_resume_full_pipeline_and_analysis(study):
    cfg, root, path = study
    engine = RecordingMock()
    directory = execute(cfg, root, path, run_id="smoke", limit=2, backend=engine)
    original = read_records(directory / "results.jsonl")
    assert len(original) == 2
    assert len(engine.batches[0][0]) == 4
    assert json.loads((directory / "manifest.json").read_text())["status"] == "incomplete"
    resumed = RecordingMock()
    execute(cfg, root, path, run_id="smoke", resume=True, backend=resumed)
    rows = read_records(directory / "results.jsonl")
    assert len(rows) == 32 and rows[:2] == original
    assert resumed.batches[0] == engine.batches[0]  # Original batch replay on partial completion.
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_conditions"] == 32
    assert manifest["is_mock"]
    assert manifest["model_revision"] == cfg.model.revision
    assert (directory / "inputs/templates/system_prompt.txt").is_file()
    idle = RecordingMock()
    execute(cfg, root, path, run_id="smoke", resume=True, backend=idle)
    assert not idle.batches
    output = analyze(directory, root / "analysis/summary")
    assert (output / "groups.csv").is_file()
    assert (output / "paired_names.csv").is_file()
    assert json.loads((output / "analysis.json").read_text())["overall"]["n_valid"] == 32


def test_failed_responses_persist_and_count_as_completed(study):
    cfg, root, path = study
    directory = execute(cfg, root, path, run_id="invalid", limit=1, backend=InvalidMock())
    row = read_records(directory / "results.jsonl")[0]
    assert row["raw_response"] == "broken raw response\nwith newline"
    assert row["score"] is None and row["parsing_status"] == "invalid"
    execute(cfg, root, path, run_id="invalid", resume=True, backend=MockBackend())
    assert read_records(directory / "results.jsonl")[0] == row
    output = analyze(directory, root / "summary")
    stats = json.loads((output / "analysis.json").read_text())["overall"]
    assert stats["n_parsing_failures"] == 1 and stats["n_total"] == 32


@pytest.mark.parametrize(
    "file",
    [
        "templates/system_prompt.txt",
        "data/jobs/example_job.txt",
        "configs/experiment.yaml",
        "data/names/names.csv",
    ],
)
def test_changed_inputs_rejected_on_resume(study, file):
    cfg, root, path = study
    execute(cfg, root, path, run_id="changed", limit=1)
    target = root / file
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="Resume rejected"):
        execute(cfg, root, path, run_id="changed", resume=True)


def test_changed_snapshot_and_record_rejected(study):
    cfg, root, path = study
    directory = execute(cfg, root, path, run_id="snapshot", limit=1)
    (directory / "inputs/templates/system_prompt.txt").write_text("modified snapshot")
    with pytest.raises(ValueError, match="snapshot changed"):
        execute(cfg, root, path, run_id="snapshot", resume=True)


def test_no_overwrite_or_duplicate_writers(study):
    cfg, root, path = study
    directory = execute(cfg, root, path, run_id="existing", limit=1)
    with pytest.raises(FileExistsError):
        execute(cfg, root, path, run_id="existing")
    with run_lock(directory):
        with pytest.raises(ValueError, match="Another process"):
            execute(cfg, root, path, run_id="existing", resume=True)


def test_failure_can_be_resumed(study):
    class CrashingMock(MockBackend):
        def generate(self, messages, seeds):
            raise RuntimeError("simulated failure")

    cfg, root, path = study
    with pytest.raises(RuntimeError, match="simulated"):
        execute(cfg, root, path, run_id="crash", backend=CrashingMock())
    directory = root / "results/crash"
    assert json.loads((directory / "manifest.json").read_text())["status"] == "failed"
    execute(cfg, root, path, run_id="crash", resume=True)
    assert len(read_records(directory / "results.jsonl")) == 32


@pytest.mark.parametrize("key", ["VLLM_ATTENTION_BACKEND", "CUBLAS_WORKSPACE_CONFIG"])
def test_resume_rejects_changed_compute_environment(study, monkeypatch, key):
    cfg, root, path = study
    monkeypatch.setenv(key, "original")
    execute(cfg, root, path, run_id="environment", limit=1)
    monkeypatch.setenv(key, "different")
    with pytest.raises(ValueError, match="Resume rejected"):
        execute(cfg, root, path, run_id="environment", resume=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", 5),
        ("name_group", "modified"),
        ("backend", "transformers"),
        ("model_revision", "0" * 40),
        ("is_mock", False),
        ("sampling_parameters", {}),
    ],
)
def test_resume_and_analysis_reject_changed_result_metadata(study, field, value):
    cfg, root, path = study
    directory = execute(cfg, root, path, run_id="modified", limit=1)
    records = read_records(directory / "results.jsonl")
    records[0][field] = value
    (directory / "results.jsonl").write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="changed|disagree"):
        execute(cfg, root, path, run_id="modified", resume=True)
    with pytest.raises(ValueError, match="changed|disagree"):
        analyze(directory, root / "analysis/modified")  
