"""Validated configuration. Paths are relative to YAML's explicit root."""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


def digest(value: object) -> str:
    payload = (
        value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Model(StrictModel):
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backend: Literal["vllm", "transformers", "mock"] = "vllm"
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    attention_implementation: Literal["eager", "sdpa"] = "eager"
    temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    max_new_tokens: int = Field(default=256, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_model_len: int = Field(default=4096, gt=0)
    gpu_memory_utilization: float = Field(default=0.7, gt=0, lt=1, allow_inf_nan=False)
    local_files_only: bool = False


class Inputs(StrictModel):
    resumes: str
    jobs: str
    names: str
    name_groups: str


class Prompts(StrictModel):
    system: str
    evaluations: list[str] = Field(min_length=1)


class Experiment(StrictModel):
    repetitions: int = Field(default=2, gt=0)
    order_seed: int = Field(default=42, ge=0)
    batch_size: int = Field(default=4, gt=0)


class Outputs(StrictModel):
    results: str = "results"
    generated: str = "data/resumes/generated"
    analysis: str = "analysis"


class Config(StrictModel):
    root: str = ".."
    inputs: Inputs
    prompts: Prompts
    model: Model
    experiment: Experiment = Experiment()
    outputs: Outputs = Outputs()


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
        raise ValueError("IDs must be 1-100 letters, digits, underscores, dots or hyphens")
    return value


def load_config(path: Path) -> tuple[Config, Path]:
    path = path.resolve()
    cfg = Config.model_validate(yaml.safe_load(path.read_text()))
    root = (path.parent / cfg.root).resolve()
    if cfg.model.max_new_tokens >= cfg.model.max_model_len:
        raise ValueError("max_new_tokens must be smaller than max_model_len")
    return cfg, root


def local_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path must remain inside project root: {path}")
    return resolved


def discover(root: Path, pattern: str) -> list[Path]:
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError(f"Expected a project-relative input glob: {pattern}")
    files = sorted(
        local_path(root, str(p.relative_to(root))) for p in root.glob(pattern) if p.is_file()
    )
    if not files:
        raise ValueError(f"No input files matched {pattern}")
    return files  
