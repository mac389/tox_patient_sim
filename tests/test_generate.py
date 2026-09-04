from __future__ import annotations

import json

from toxsim.generate import main


def test_cli_writes_json_and_jsonl(model_data, tmp_path):
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    # The fixture data lives in tmp_path, so use a copied fixture directory for
    # a destination that does not overlap with input.
    for path in tmp_path.glob("*.yml"):
        path.rename(model_directory / path.name)

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
            "--model-data-dir",
            str(model_directory),
        ]
    ) == 0

    json_patients = json.loads((destination / "patients.json").read_text(encoding="utf-8"))
    jsonl_patients = [
        json.loads(line)
        for line in (destination / "patients.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert json_patients == jsonl_patients
    assert len(json_patients) == 2
