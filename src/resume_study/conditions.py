"""Build a fixed factorial design before loading the inference engine."""

import random
from dataclasses import asdict
from itertools import product
from pathlib import Path

from jinja2 import meta

from .config import Config, digest, discover, local_path
from .generate_resumes import ENV, generate


def text_input(path: Path) -> str:
    text = path.read_text()
    if not text.strip():
        raise ValueError(f"Empty input: {path}")
    return text


def make_conditions(cfg: Config, root: Path) -> tuple[list[dict], list[Path]]:
    variants = generate(cfg, root)
    jobs = discover(root, cfg.inputs.jobs)
    bases = discover(root, cfg.inputs.resumes)
    system_path = local_path(root, cfg.prompts.system)
    system = text_input(system_path)
    templates = sorted({local_path(root, p) for p in cfg.prompts.evaluations})
    if len(templates) != len(cfg.prompts.evaluations):
        raise ValueError("Duplicate prompt templates")
    loaded = []
    for path in templates:
        source = text_input(path)
        if meta.find_undeclared_variables(ENV.parse(source)) != {"resume", "job_description"}:
            raise ValueError("Evaluation template must use exactly resume and job_description")
        # Only literal blocks and direct interpolation preserve qualification comparisons.
        from jinja2 import nodes

        for statement in ENV.parse(source).body:
            if not isinstance(statement, nodes.Output) or any(
                not isinstance(item, (nodes.TemplateData, nodes.Name))
                or isinstance(item, nodes.Name)
                and item.name not in {"resume", "job_description"}
                for item in statement.nodes
            ):
                raise ValueError("Evaluation template permits literal text and direct slots only")
        loaded.append((path, source, ENV.from_string(source)))
    conditions = []
    for variant, job, (template_path, source, template), repetition in product(
        variants, jobs, loaded, range(cfg.experiment.repetitions)
    ):
        job_text = text_input(job)
        job_id = "job_" + digest([job.relative_to(root).as_posix(), job_text])[:24]
        prompt_id = (
            "prompt_" + digest([template_path.relative_to(root).as_posix(), system, source])[:24]
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": template.render(resume=variant.text, job_description=job_text),
            },
        ]
        fields = asdict(variant)
        fields.pop("text")
        fields.update(
            job_id=job_id,
            job_hash=digest(job_text),
            prompt_template_id=prompt_id,
            repetition_index=repetition,
            prompt_hash=digest(messages),
            messages=messages,
        )
        # Common random numbers within the paired stratum; never seed by group/name.
        fields["random_seed"] = int(
            digest([cfg.model.seed, variant.base_resume_id, job_id, prompt_id, repetition])[:8], 16
        )
        identity = {k: v for k, v in fields.items() if k not in {"messages", "base_path"}}
        fields["condition_id"] = "condition_" + digest([identity, cfg.model.model_dump()])[:32]
        conditions.append(fields)
    random.Random(cfg.experiment.order_seed).shuffle(conditions)
    inputs = [
        *bases,
        *jobs,
        system_path,
        *templates,
        local_path(root, cfg.inputs.names),
        local_path(root, cfg.inputs.name_groups),
    ]
    return conditions, sorted(set(inputs))  
