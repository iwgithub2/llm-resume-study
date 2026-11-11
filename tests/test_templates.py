import pytest
from pydantic import ValidationError

from resume_study.conditions import make_conditions
from resume_study.config import Config, discover, load_config, local_path


def test_config_jobs_prompts_and_factorial_design(study):
    cfg, root, path = study
    loaded, loaded_root = load_config(path)
    assert loaded_root == root
    assert loaded.model.backend == "vllm"
    assert len(discover(root, cfg.inputs.jobs)) == 2
    conditions, files = make_conditions(cfg, root)
    assert len(conditions) == 2 * 4 * 2 * 1 * 2 == 32
    assert len({c["condition_id"] for c in conditions}) == 32
    assert conditions == make_conditions(cfg, root)[0]
    assert len(files) == 8
    for condition in conditions:
        prompt = condition["messages"][1]["content"]
        assert "<job>" in prompt and "<resume>" in prompt
        assert condition["applicant_name"] in prompt
        assert "synthetic_a" not in prompt and "synthetic_b" not in prompt
    strata = {}
    for row in conditions:
        key = (
            row["base_resume_id"],
            row["job_id"],
            row["prompt_template_id"],
            row["repetition_index"],
        )
        strata.setdefault(key, set()).add(row["random_seed"])
    assert all(len(seeds) == 1 for seeds in strata.values())


def test_prompt_variants_only_change_name(study):
    cfg, root, _ = study
    conditions, _ = make_conditions(cfg, root)
    first = conditions[0]
    paired = [
        c
        for c in conditions
        if all(
            c[key] == first[key]
            for key in ("base_resume_id", "job_id", "prompt_template_id", "repetition_index")
        )
    ]
    # Synthetic names do not occur in literal text; slot-aware behavior is tested separately.
    assert (
        len({c["messages"][1]["content"].replace(c["applicant_name"], "<NAME>") for c in paired})
        == 1
    )
    assert len({c["messages"][0]["content"] for c in conditions}) == 1


def test_missing_jobs_and_undefined_prompt(study):
    cfg, root, _ = study
    with pytest.raises(ValueError, match="No input"):
        discover(root, "missing/*.txt")
    (root / cfg.prompts.evaluations[0]).write_text("{{ missing }}")
    with pytest.raises(ValueError, match="exactly"):
        make_conditions(cfg, root)


@pytest.mark.parametrize(
    "field,value",
    [("revision", "main"), ("backend", "auto"), ("temperature", -1.0), ("seed", True)],
)
def test_bad_model_settings(study, field, value):
    cfg, _, _ = study
    data = cfg.model_dump()
    data["model"][field] = value
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_path_escape_rejected(study):
    _, root, _ = study
    with pytest.raises(ValueError, match="inside"):
        local_path(root, "../private.txt")


@pytest.mark.parametrize("global_name", ["lipsum", "cycler", "namespace", "range"])
def test_jinja_globals_are_not_allowed_in_prompt(study, global_name):
    cfg, root, _ = study
    (root / cfg.prompts.evaluations[0]).write_text(
        "{{ resume }} {{ job_description }} {{ " + global_name + " }}"
    )
    with pytest.raises(ValueError, match="direct slots"):
        make_conditions(cfg, root)  
