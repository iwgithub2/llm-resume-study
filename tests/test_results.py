import json

import pytest

from resume_study.results import append_record, parse_response, read_records, write_csv


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '```json\n{"score":3}\n```',
        '{"score":true,"recommendation":"maybe","reason":"x"}',
        '{"score":3.0,"recommendation":"maybe","reason":"x"}',
        '{"score":6,"recommendation":"maybe","reason":"x"}',
        '{"score":3,"recommendation":"hire","reason":"x"}',
        '{"score":3,"recommendation":"maybe","reason":" "}',
        '{"score":3,"recommendation":"maybe","reason":"x","extra":1}',
        '{"score":1,"score":3,"recommendation":"maybe","reason":"x"}',
        "[]",
        "null",
    ],
)
def test_invalid_responses_have_no_invented_values(raw):
    row = parse_response(raw)
    assert row["parsing_status"] == "invalid"
    assert row["score"] is row["recommendation"] is row["explanation"] is None
    assert row["parsing_error"]


def test_valid_json_whitespace():
    row = parse_response(' \n{"score":4,"recommendation":"shortlist","reason":"Test evidence."}\n')
    assert row["score"] == 4 and row["parsing_status"] == "valid"


def test_jsonl_csv_and_duplicate_detection(tmp_path):
    path = tmp_path / "results.jsonl"
    row = {"condition_id": "one", "raw_response": 'a\nb,"c"', "sampling": {"temperature": 0}}
    append_record(path, row)
    assert read_records(path) == [row]
    write_csv(tmp_path / "results.csv", [row])
    import csv

    with (tmp_path / "results.csv").open() as stream:
        saved = list(csv.DictReader(stream))[0]
    assert saved["raw_response"] == row["raw_response"]
    assert json.loads(saved["sampling"]) == row["sampling"]
    append_record(path, row)
    with pytest.raises(ValueError, match="Duplicate"):
        read_records(path)


def test_corrupt_tail_is_not_silently_discarded(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"condition_id":"one"}\n{"cond')
    with pytest.raises(ValueError, match="Corrupt JSONL"):
        read_records(path)


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_are_not_jsonl_record_boundaries(tmp_path, separator):
    path = tmp_path / "results.jsonl"
    row = {"condition_id": "unicode", "raw_response": f"first{separator}second"}
    append_record(path, row)
    assert read_records(path) == [row]


def test_append_after_complete_record_missing_final_newline(tmp_path):
    path = tmp_path / "results.jsonl"
    first = {"condition_id": "first", "raw_response": "text"}
    second = {"condition_id": "second", "raw_response": "text"}
    path.write_text(json.dumps(first))
    append_record(path, second)
    assert read_records(path) == [first, second]  
