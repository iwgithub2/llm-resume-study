import json

from resume_study.cli import main


def test_mock_cli_workflow(study, capsys):
    _, _, path = study
    assert main(["check", "--config", str(path), "--backend", "mock"]) == 0
    assert json.loads(capsys.readouterr().out)["conditions"] == 32
    assert main(["generate", "--config", str(path)]) == 0
    assert (
        main(["run", "--config", str(path), "--backend", "mock", "--run-id", "cli", "--limit", "2"])
        == 0
    )
    assert (
        main(["run", "--config", str(path), "--backend", "mock", "--run-id", "cli", "--resume"])
        == 0
    )
    assert main(["analyze", "--config", str(path), "--run-id", "cli"]) == 0


def test_cli_actionable_errors(study, capsys):
    _, _, path = study
    assert main(["run", "--config", str(path), "--backend", "mock", "--limit", "0"]) == 2
    assert "limit must be positive" in capsys.readouterr().err
    assert main(["analyze", "--config", str(path), "--run-id", "../bad"]) == 2


def test_generation_oom_has_actionable_cli_error(study, capsys, monkeypatch):
    from resume_study.inference import MockBackend

    class OOMMock(MockBackend):
        def generate(self, messages, seeds):
            raise RuntimeError("CUDA out of memory")

    _, _, path = study
    monkeypatch.setattr("resume_study.experiment.load_backend", lambda cfg: OOMMock())
    assert main(["run", "--config", str(path), "--backend", "mock", "--run-id", "oom"]) == 2
    error = capsys.readouterr().err
    assert "CUDA out of memory" in error and "NEW run" in error


def test_malformed_templates_and_yaml_are_reported(study, capsys):
    cfg, root, path = study
    (root / cfg.prompts.evaluations[0]).write_text("{{ broken")
    assert main(["check", "--config", str(path), "--backend", "mock"]) == 2
    assert json.loads(capsys.readouterr().out)["ready"] is False
    path.write_text("bad: [unterminated")
    assert main(["check", "--config", str(path)]) == 2
    assert "Error:" in capsys.readouterr().err  
