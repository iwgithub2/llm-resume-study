"""Small argparse CLI; run from repository root or pass --config."""

import argparse
import json
import sys
import uuid
from pathlib import Path

import yaml
from jinja2 import TemplateError

from .analysis import analyze
from .conditions import make_conditions
from .config import Config, load_config, local_path, safe_id
from .experiment import execute
from .generate_resumes import generate, save_variants
from .inference import InferenceError, hardware, preflight, versions


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    for command in ("check", "generate", "run", "analyze"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
        if command in {"check", "run"}:
            sub.add_argument("--backend", choices=("vllm", "transformers", "mock"))
            sub.add_argument("--device", choices=("cpu", "cuda", "mps"))
        if command == "run":
            sub.add_argument("--run-id")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--batch-size", type=int)
        if command == "analyze":
            sub.add_argument("--run-id", required=True)
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cfg, root = load_config(args.config)
        data = cfg.model_dump()
        for field in ("backend", "device"):
            if getattr(args, field, None) is not None:
                data["model"][field] = getattr(args, field)
        if getattr(args, "batch_size", None) is not None:
            data["experiment"]["batch_size"] = args.batch_size
        cfg = Config.model_validate(data)
        if args.command == "check":
            report = {"hardware": hardware(), "versions": versions(), "backend": cfg.model.backend}
            try:
                conditions, _ = make_conditions(cfg, root)
                report["conditions"] = len(conditions)
                preflight(cfg.model)
                report["ready"] = True
            except (ValueError, InferenceError, OSError, TemplateError, yaml.YAMLError) as exc:
                report.update(ready=False, action=str(exc))
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 2
        if args.command == "generate":
            variants = generate(cfg, root)
            target = local_path(root, cfg.outputs.generated)
            save_variants(variants, target)
            print(f"Generated {len(variants)} variants: {target}")
        elif args.command == "run":
            target = execute(
                cfg, root, args.config, run_id=args.run_id, resume=args.resume, limit=args.limit
            )
            print(f"Run saved: {target}")
        else:
            run_id = safe_id(args.run_id)
            directory = local_path(root, cfg.outputs.results) / run_id
            target = local_path(root, cfg.outputs.analysis) / f"{run_id}_{uuid.uuid4().hex[:8]}"
            print(f"Analysis saved: {analyze(directory, target)}")
        return 0
    except (ValueError, OSError, InferenceError, KeyError, TemplateError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted. Resume with the same configuration and --run-id ID --resume.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())  
