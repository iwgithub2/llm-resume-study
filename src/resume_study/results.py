"""Strict parsing and durable JSONL, with CSV as a derived view."""

import csv
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from .config import StrictModel


class Evaluation(StrictModel):
    score: int = Field(ge=1, le=5)
    recommendation: Literal["shortlist", "maybe", "reject"]
    reason: str = Field(min_length=1)


def unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_response(raw: str) -> dict:
    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
        value = Evaluation.model_validate(data)
        if not value.reason.strip():
            raise ValueError("Reason must contain non-whitespace text")
        return {
            "score": value.score,
            "recommendation": value.recommendation,
            "explanation": value.reason,
            "parsing_status": "valid",
            "parsing_error": None,
        }
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return {
            "score": None,
            "recommendation": None,
            "explanation": None,
            "parsing_status": "invalid",
            "parsing_error": str(exc),
        }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    seen = set()
    # JSONL separates physical LF lines, not Unicode separators inside JSON strings.
    lines = path.read_text().split("\n")
    if lines[-1] == "":
        lines.pop()
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line, object_pairs_hook=unique_object)
            condition_id = row["condition_id"]
            if not isinstance(condition_id, str):
                raise ValueError("condition_id must be a string")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Corrupt JSONL at {path}:{index}; preserve original and inspect before resume"
            ) from exc
        if condition_id in seen:
            raise ValueError(f"Duplicate condition in results: {condition_id}")
        records.append(row)
        seen.add(condition_id)
    return records


def append_record(path: Path, row: dict) -> None:
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab+") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = fieldnames or (list(rows[0]) if rows else [])
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)  
