from __future__ import annotations

import json

from toxsim.generate import main


def test_cli_writes_json_and_jsonl(tmp_path):
    destination = tmp_path / "output"
    assert main(
        [
            "--count",
            "2",
            "--destination",
            str(destination),
            "--format",
            "both",
            "--seed",
            "11",
        ]
    ) == 0

    json_patients = json.loads((destination / "patients.json").read_text(encoding="utf-8"))
    jsonl_patients = [
        json.loads(line)
        for line in (destination / "patients.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert json_patients == jsonl_patients
    assert len(json_patients) == 2


def test_cli_writes_output_file_and_infers_jsonl_format(tmp_path):
    output = tmp_path / "patients.jsonl"

    assert main(["--count", "2", "--output", str(output), "--seed", "11"]) == 0

    patients = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(patients) == 2
