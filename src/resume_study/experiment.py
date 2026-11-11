"""Persistent, locked runs; reject changed designs/environments on resume."""

import fcntl
import hashlib
import json
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .conditions import make_conditions
from .config import Config, Model, digest, local_path, safe_id
from .inference import (
    Backend,
    InferenceError,
    execution_environment,
    hardware,
    load_backend,
    prepare_environment,
    versions,
)
from .results import append_record, atomic_json, parse_response, read_records, write_csv


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
        "diff_hash": digest(git("diff", "HEAD") or ""),
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(cfg: Config, root: Path, inputs: list[Path], config_path: Path) -> dict:
    sources = sorted((root / "src").rglob("*.py"))
    sources += [
        root / p
        for p in ("pyproject.toml", "uv.lock", "requirements.txt", ".python-version")
        if (root / p).is_file()
    ]
    all_files = sorted(set(inputs + sources))
    return {
        "config": cfg.model_dump(),
        "config_source_hash": file_hash(config_path),
        "file_hashes": {p.relative_to(root).as_posix(): file_hash(p) for p in all_files},
        "runtime": {
            "versions": versions(),
            "hardware": hardware(),
            "environment": execution_environment(),
        },
    }


@contextmanager
def run_lock(directory: Path):
    # OS releases the lock after termination; lock-file existence is not lock ownership.
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is writing this run; wait for it to finish") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def model_record_fields(model: Model) -> dict:
    return {
        "model_name": model.name,
        "model_revision": model.revision,
        "backend": model.backend,
        "is_mock": model.backend == "mock",
        "sampling_parameters": {
            "temperature": model.temperature,
            "top_p": model.top_p,
            "max_new_tokens": model.max_new_tokens,
            "top_k": "disabled",
            "repetition_penalty": 1.0,
        },
    }


def validate_records(rows: list[dict], conditions: list[dict], run_id: str, model: Model) -> None:
    expected = {row["condition_id"]: row for row in conditions}
    for row in rows:
        condition = expected.get(row["condition_id"])
        if condition is None or row.get("run_id") != run_id:
            raise ValueError("Result does not belong to this run/design")
        for key, value in condition.items():
            if key != "messages" and row.get(key) != value:
                raise ValueError(f"Result metadata changed: {row['condition_id']} / {key}")
        for key, value in model_record_fields(model).items():
            if row.get(key) != value:
                raise ValueError(f"Stored model metadata changed: {key}")
        if not isinstance(row.get("raw_response"), str):
            raise ValueError("Raw model response must be a string")
        parsed = parse_response(row["raw_response"])
        if digest(parsed) != digest({key: row.get(key) for key in parsed}):
            raise ValueError("Stored parsing fields disagree with raw response")


def execute(
    cfg: Config,
    root: Path,
    config_path: Path,
    *,
    run_id: str | None = None,
    resume: bool = False,
    limit: int | None = None,
    backend: Backend | None = None,
) -> Path:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if resume and not run_id:
        raise ValueError("Resume requires --run-id")
    prepare_environment(cfg.model)
    conditions, inputs = make_conditions(cfg, root)
    identity = provenance(cfg, root, inputs, config_path)
    fingerprint = digest(identity)
    run_id = safe_id(run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    directory = local_path(root, str(Path(cfg.outputs.results) / run_id))
    if resume:
        if not directory.is_dir():
            raise ValueError(f"Run does not exist: {run_id}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        order = [row["condition_id"] for row in conditions]
        if resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != fingerprint:
                raise ValueError(
                    "Resume rejected: configuration, inputs, source, packages or hardware changed. "
                    "Restore the original environment or start a NEW run."
                )
            if json.loads((directory / "conditions.json").read_text()) != conditions:
                raise ValueError("Stored condition schedule changed; cannot resume")
            for relative, expected_hash in manifest["file_hashes"].items():
                snapshot = directory / "inputs" / relative
                if not snapshot.is_file() or file_hash(snapshot) != expected_hash:
                    raise ValueError(f"Stored input snapshot changed: {relative}")
            if file_hash(directory / "config_source.yaml") != manifest["config_source_hash"]:
                raise ValueError("Stored configuration snapshot changed")
        else:
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
                "status": "initialized",
                "fingerprint": fingerprint,
                **identity,
                "git": git_state(root),
                "condition_order": order,
                "expected_conditions": len(conditions),
                "completed_conditions": 0,
                "is_mock": cfg.model.backend == "mock",
                "engine": None,
                "model_name": cfg.model.name,
                "model_revision": cfg.model.revision,
                "backend": cfg.model.backend,
                "attempts": [],
            }
            for relative in identity["file_hashes"]:
                destination = directory / "inputs" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, destination)
            shutil.copyfile(config_path, directory / "config_source.yaml")
            atomic_json(directory / "conditions.json", conditions)
            atomic_json(manifest_path, manifest)
        rows = read_records(directory / "results.jsonl")
        validate_records(rows, conditions, run_id, cfg.model)
        done = {row["condition_id"] for row in rows}
        attempt = {"started_at": utc_now(), "limit": limit, "previously_completed": len(done)}
        manifest["attempts"].append(attempt)
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        added = 0
        try:
            if len(done) < len(conditions):
                engine = backend or load_backend(cfg.model)
                if engine.name != cfg.model.backend:
                    raise ValueError("Backend instance does not match configured backend")
                if manifest["engine"] is not None and manifest["engine"] != engine.metadata:
                    raise ValueError("Engine settings changed; start a new run")
                manifest["engine"] = engine.metadata
                atomic_json(manifest_path, manifest)
                size = cfg.experiment.batch_size
                for offset in range(0, len(conditions), size):
                    batch = conditions[offset : offset + size]
                    if all(row["condition_id"] in done for row in batch):
                        continue
                    # Replay the ORIGINAL batch, including completed requests. Only new rows persist.
                    # This holds scheduler/batch context fixed across interruptions and --limit.
                    try:
                        texts = engine.generate(
                            [row["messages"] for row in batch],
                            [row["random_seed"] for row in batch],
                        )
                    except InferenceError:
                        raise
                    except Exception as exc:
                        raise InferenceError(
                            f"{engine.name} decoding failed at original batch {offset // size}: {exc}. "
                            "For memory errors reduce batch_size/max_model_len/max_new_tokens in a NEW run. "
                            "For unsupported deterministic kernels select an explicit CPU run. "
                            "Existing records are preserved; resume requires the original settings."
                        ) from exc
                    if len(texts) != len(batch) or any(not isinstance(t, str) for t in texts):
                        raise InferenceError("Backend must return exactly one text per request")
                    for condition, raw in zip(batch, texts, strict=True):
                        if condition["condition_id"] in done:
                            continue
                        fields = {k: v for k, v in condition.items() if k != "messages"}
                        row = {
                            "run_id": run_id,
                            **fields,
                            **model_record_fields(cfg.model),
                            "formatted_prompt_hash": digest(engine.format(condition["messages"])),
                            "raw_response": raw,
                            **parse_response(raw),
                        }
                        append_record(directory / "results.jsonl", row)
                        rows.append(row)
                        done.add(condition["condition_id"])
                        added += 1
                        if limit is not None and added >= limit:
                            break
                    if limit is not None and added >= limit:
                        break
            manifest["status"] = "complete" if len(done) == len(conditions) else "incomplete"
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["ended_at"] = utc_now()
            attempt["added_conditions"] = added
            manifest["completed_conditions"] = len(done)
            atomic_json(manifest_path, manifest)
            write_csv(directory / "results.csv", rows)
    return directory  
