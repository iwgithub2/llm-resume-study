"""Run the supplied 0–100 study manifest without changing the original 1–5 example."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from .config import StrictModel, digest, local_path, safe_id
from .experiment import file_hash, git_state, model_record_fields, provenance, run_lock, utc_now
from .inference import Backend, InferenceError, load_backend, prepare_environment
from .pilot_data import PilotConfig, build_pilot, load_pilot_config
from .results import append_record, atomic_json, read_records, unique_object, write_csv


class PilotEvaluation(StrictModel):
    score: int = Field(ge=0, le=100)
    interview: Literal["yes", "no"]
    justifications: list[str] = Field(min_length=2, max_length=2)


def parse_pilot(raw: str) -> dict:
    try:
        value = PilotEvaluation.model_validate(json.loads(raw, object_pairs_hook=unique_object))
        if any(not reason.strip() for reason in value.justifications):
            raise ValueError("Both justifications must contain text")
        return {
            **value.model_dump(),
            "parsing_status": "valid",
            "parsing_error": None,
            "threshold_consistent": (value.score >= 70) == (value.interview == "yes"),
        }
    except (ValueError, ValidationError) as exc:
        return {
            "score": None,
            "interview": None,
            "justifications": None,
            "parsing_status": "invalid",
            "parsing_error": str(exc),
            "threshold_consistent": None,
        }


def validate_pilot_records(rows: list[dict], conditions: list[dict], manifest: dict) -> None:
    design = {c["condition_id"]: c for c in conditions}
    model = PilotConfig.model_validate(manifest["config"]).model
    for row in rows:
        condition = design.get(row["condition_id"])
        if condition is None or row.get("run_id") != manifest["run_id"]:
            raise ValueError("Result is outside the saved pilot design")
        for key, value in {**condition, **model_record_fields(model)}.items():
            if key != "messages" and row.get(key) != value:
                raise ValueError(f"Stored pilot metadata changed: {key}")
        if not isinstance(row.get("raw_response"), str):
            raise ValueError("Pilot raw response must be a string")
        parsed = parse_pilot(row["raw_response"])
        if digest(parsed) != digest({k: row.get(k) for k in parsed}):
            raise ValueError("Pilot parsed fields disagree with raw response")


def run_pilot(
    config_path: Path,
    *,
    run_id: str,
    resume: bool = False,
    limit: int | None = None,
    engine: Backend | None = None,
) -> Path:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    cfg, root = load_pilot_config(config_path)
    prepare_environment(cfg.model)
    conditions, inputs, audit = build_pilot(cfg, root)
    identity = provenance(cfg, root, inputs, config_path)
    directory = local_path(root, f"results/{safe_id(run_id)}")
    if resume:
        if not directory.is_dir():
            raise ValueError("Pilot run does not exist")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        manifest_path = directory / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest["fingerprint"] != digest(identity):
                raise ValueError(
                    "Pilot resume rejected: inputs, source, settings or environment changed"
                )
            if json.loads((directory / "conditions.json").read_text()) != conditions:
                raise ValueError("Saved pilot condition order changed")
            for relative, sha in identity["file_hashes"].items():
                if file_hash(directory / "inputs" / relative) != sha:
                    raise ValueError(f"Pilot snapshot changed: {relative}")
            if file_hash(directory / "config_source.yaml") != identity["config_source_hash"]:
                raise ValueError("Pilot config snapshot changed")
        else:
            manifest = {
                "schema_version": 1,
                "run_kind": "supplied_manifest_pilot",
                "run_id": run_id,
                "created_at": utc_now(),
                "status": "initialized",
                **identity,
                "fingerprint": digest(identity),
                "git": git_state(root),
                "dataset_audit": audit,
                "expected_conditions": len(conditions),
                "completed_conditions": 0,
                "engine": None,
                "attempts": [],
            }
            for relative in identity["file_hashes"]:
                target = directory / "inputs" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, target)
            shutil.copyfile(config_path, directory / "config_source.yaml")
            atomic_json(directory / "conditions.json", conditions)
            atomic_json(manifest_path, manifest)
        rows = read_records(directory / "results.jsonl")
        validate_pilot_records(rows, conditions, manifest)
        done = {r["condition_id"] for r in rows}
        attempt = {"started_at": utc_now(), "previously_completed": len(done), "limit": limit}
        manifest["attempts"].append(attempt)
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)
        added = 0
        try:
            if len(done) < len(conditions):
                start = time.perf_counter()
                engine = engine or load_backend(cfg.model)
                attempt["engine_load_seconds"] = time.perf_counter() - start
                if engine.name != cfg.model.backend:
                    raise ValueError("Pilot backend differs from configuration")
                if manifest["engine"] is not None and manifest["engine"] != engine.metadata:
                    raise ValueError("Pilot engine settings changed")
                manifest["engine"] = engine.metadata
                atomic_json(manifest_path, manifest)
                for condition in conditions:
                    if condition["condition_id"] in done:
                        continue
                    start = time.perf_counter()
                    try:
                        outputs = engine.generate(
                            [condition["messages"]], [condition["suggested_seed"]]
                        )
                    except Exception as exc:
                        raise InferenceError(
                            f"Pilot inference failed at {condition['evaluation_id']}: {exc}. "
                            "No automatic device fallback; preserve this run and use a new "
                            "explicit CPU configuration if MPS is unsupported."
                        ) from exc
                    elapsed = time.perf_counter() - start
                    if len(outputs) != 1 or not isinstance(outputs[0], str):
                        raise InferenceError("Pilot requires one raw response per evaluation")
                    raw = outputs[0]
                    fields = {k: v for k, v in condition.items() if k != "messages"}
                    row = {
                        "run_id": run_id,
                        **fields,
                        **model_record_fields(cfg.model),
                        "formatted_prompt_hash": digest(engine.format(condition["messages"])),
                        "raw_response": raw,
                        **parse_pilot(raw),
                        "inference_seconds": elapsed,
                    }
                    if hasattr(engine, "tokenizer"):
                        row["decoded_response_tokens"] = len(
                            engine.tokenizer.encode(raw, add_special_tokens=False)
                        )
                    if cfg.model.backend == "transformers" and cfg.model.device == "mps":
                        import torch

                        row["mps_live_allocated_bytes"] = torch.mps.current_allocated_memory()
                        row["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
                    append_record(directory / "results.jsonl", row)
                    rows.append(row)
                    done.add(condition["condition_id"])
                    added += 1
                    print(
                        json.dumps(
                            {
                                "completed": len(done),
                                "expected": len(conditions),
                                "id": condition["evaluation_id"],
                                "score": row["score"],
                                "status": row["parsing_status"],
                                "seconds": round(elapsed, 2),
                            }
                        ),
                        flush=True,
                    )
                    if limit is not None and added >= limit:
                        break
            manifest["status"] = "complete" if len(done) == len(conditions) else "incomplete"
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt.update(ended_at=utc_now(), added_conditions=added)
            manifest["completed_conditions"] = len(done)
            atomic_json(manifest_path, manifest)
            write_csv(directory / "results.csv", rows)
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_qwen4b.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.audit_only:
            cfg, root = load_pilot_config(args.config)
            conditions, _, audit = build_pilot(cfg, root)
            print(json.dumps({**audit, "planned_calls": len(conditions)}, indent=2))
        else:
            print(
                f"Pilot saved: {run_pilot(args.config, run_id=args.run_id, resume=args.resume, limit=args.limit)}"
            )
        return 0
    except (ValueError, OSError, InferenceError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; resume with unchanged config and --resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())  
