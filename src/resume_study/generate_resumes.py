"""Literal Jinja name slots; no evaluation of arbitrary template expressions."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, nodes

from .config import Config, digest, discover, local_path, safe_id

ENV = Environment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False)
CANONICAL = "<APPLICANT_NAME>"


@dataclass(frozen=True)
class Name:
    name_id: str
    applicant_name: str
    group_label: str


@dataclass(frozen=True)
class Variant:
    base_resume_id: str
    generated_resume_id: str
    name_id: str
    applicant_name: str
    name_group: str
    text: str
    input_hash: str
    canonical_hash: str
    base_path: str


def load_names(cfg: Config, root: Path) -> list[Name]:
    groups = yaml.safe_load(local_path(root, cfg.inputs.name_groups).read_text())
    if not isinstance(groups, dict) or not isinstance(groups.get("groups"), dict):
        raise ValueError("name_groups YAML must contain a groups mapping")
    with local_path(root, cfg.inputs.names).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"name_id", "applicant_name", "group_label"}:
            raise ValueError("Names CSV needs exactly name_id,applicant_name,group_label")
        names = []
        for row in reader:
            if all(not (v or "").strip() for v in row.values() if isinstance(v, str)):
                continue
            if None in row or any(not isinstance(v, str) or not v.strip() for v in row.values()):
                raise ValueError("Incomplete or extra fields in names CSV")
            name = Name(**{k: v.strip() for k, v in row.items()})
            safe_id(name.name_id)
            if any(ord(c) < 32 or ord(c) == 127 for c in name.applicant_name):
                raise ValueError("Applicant name must be a single line without control characters")
            if name.group_label not in groups["groups"]:
                raise ValueError(f"Unknown researcher-provided group: {name.group_label}")
            names.append(name)
    if not names or len({n.name_id for n in names}) != len(names):
        raise ValueError("Names must be nonempty with unique name_id values")
    if len({n.applicant_name for n in names}) != len(names):
        raise ValueError("Duplicate applicant names would create ambiguous name conditions")
    return sorted(names, key=lambda n: n.name_id)


def name_parts(source: str) -> list[str | None]:
    ast = ENV.parse(source)
    parts: list[str | None] = []
    for statement in ast.body:
        if not isinstance(statement, nodes.Output):
            raise ValueError("Resume permits only literal text and {{ applicant_name }}")
        for item in statement.nodes:
            if isinstance(item, nodes.TemplateData):
                parts.append(item.data)
            elif isinstance(item, nodes.Name) and item.name == "applicant_name":
                parts.append(None)
            else:
                raise ValueError("Resume permits only literal text and {{ applicant_name }}")
    # Whitespace-control and comment syntax could silently change literal text.
    if any(token in source for token in ("{#", "{%", "{{-", "-}}")) or None not in parts:
        raise ValueError("Use plain {{ applicant_name }} slots; no comments or whitespace control")
    return parts


def render_name(source: str, name: str) -> tuple[str, str]:
    parts = name_parts(source)
    rendered = ENV.from_string(source).render(applicant_name=name)
    expected = "".join(name if part is None else part for part in parts)
    if rendered != expected:
        raise ValueError("Rendered resume changed text outside name slots")
    # Reconstruct canonical text from known slot boundaries, never global replacement.
    canonical = "".join(CANONICAL if part is None else part for part in parts)
    return rendered, canonical


def generate(cfg: Config, root: Path) -> list[Variant]:
    names = load_names(cfg, root)
    variants = []
    for path in discover(root, cfg.inputs.resumes):
        source = path.read_text()
        relative = path.relative_to(root).as_posix()
        base_id = "base_" + digest([relative, source])[:24]
        canonical_hash = None
        for name in names:
            text, canonical = render_name(source, name.applicant_name)
            current = digest(canonical)
            if canonical_hash is not None and current != canonical_hash:
                raise ValueError(f"Non-name content differs for {relative}")
            canonical_hash = current
            variant_id = "resume_" + digest([base_id, asdict(name), text])[:24]
            variants.append(
                Variant(
                    base_id,
                    variant_id,
                    name.name_id,
                    name.applicant_name,
                    name.group_label,
                    text,
                    digest(text),
                    current,
                    relative,
                )
            )
    return variants


def save_variants(variants: list[Variant], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = [(directory / f"{v.generated_resume_id}.txt", v.text) for v in variants]
    entries.append(
        (directory / "variants.json", json.dumps([asdict(v) for v in variants], indent=2))
    )
    for path, content in entries:
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to overwrite changed generated input: {path}")
    for path, content in entries:
        if not path.exists():
            with path.open("x") as stream:
                stream.write(content)  
