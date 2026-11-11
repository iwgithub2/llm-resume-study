import shutil
from pathlib import Path

import pytest

from resume_study.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def study(tmp_path):
    for folder in ("configs", "templates", "data"):
        shutil.copytree(
            REPO / folder, tmp_path / folder, ignore=shutil.ignore_patterns("generated")
        )
    path = tmp_path / "configs/experiment.yaml"
    cfg, root = load_config(path)
    data = cfg.model_dump()
    data["model"]["backend"] = "mock"
    data["model"]["device"] = "cpu"
    return Config.model_validate(data), root, path  
