import csv

import pytest

from resume_study.generate_resumes import generate, load_names, render_name, save_variants


def test_only_slot_changes_even_if_name_occurs_in_literal():
    source = "Name: {{ applicant_name }}\nProject: Avery Ellis\nSkills: SQL\n"
    text, canonical = render_name(source, "Avery Ellis")
    other, canonical_other = render_name(source, "Morgan Lane")
    assert text == "Name: Avery Ellis\nProject: Avery Ellis\nSkills: SQL\n"
    assert other == "Name: Morgan Lane\nProject: Avery Ellis\nSkills: SQL\n"
    assert (
        canonical
        == canonical_other
        == "Name: <APPLICANT_NAME>\nProject: Avery Ellis\nSkills: SQL\n"
    )


@pytest.mark.parametrize(
    "source",
    [
        "No placeholder",
        "{{ email }}",
        "{{ applicant_name | upper }}",
        "{% if applicant_name %}x{% endif %}",
        "{{- applicant_name }}",
        "{# remove #}{{ applicant_name }}",
        "{{ applicant_name.upper() }}",
    ],
)
def test_invalid_template_is_rejected(source):
    with pytest.raises(ValueError):
        render_name(source, "Synthetic Name")


def test_multiple_slots_and_literal_template_text_in_name():
    text, canonical = render_name("{{ applicant_name }} / {{ applicant_name }}", "{{ literal }}")
    assert text == "{{ literal }} / {{ literal }}"
    assert canonical == "<APPLICANT_NAME> / <APPLICANT_NAME>"


def test_multiple_bases_names_stable_and_canonical(study):
    cfg, root, _ = study
    variants = generate(cfg, root)
    assert len(variants) == 8
    assert variants == generate(cfg, root)
    assert len({v.generated_resume_id for v in variants}) == 8
    assert len({v.base_resume_id for v in variants}) == 2
    for base in {v.base_resume_id for v in variants}:
        assert len({v.canonical_hash for v in variants if v.base_resume_id == base}) == 1
    directory = root / "generated"
    save_variants(variants, directory)
    save_variants(variants, directory)  # Idempotent identical inputs.
    path = directory / f"{variants[0].generated_resume_id}.txt"
    path.write_text("changed")
    with pytest.raises(ValueError, match="overwrite"):
        save_variants(variants, directory)


@pytest.mark.parametrize(
    "rows",
    [
        [["a", "One Name", "synthetic_a"], ["a", "Other Name", "synthetic_b"]],
        [["a", "One Name", "unknown"]],
        [["a", "", "synthetic_a"]],
        [["a", "Multi\nLine", "synthetic_a"]],
        [["a", "One Name", "synthetic_a"], ["b", "One Name", "synthetic_b"]],
    ],
)
def test_bad_name_metadata(study, rows):
    cfg, root, _ = study
    with (root / cfg.inputs.names).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name_id", "applicant_name", "group_label"])
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_names(cfg, root)  
