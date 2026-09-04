from __future__ import annotations

import pytest

from toxsim import load_model_data


@pytest.fixture
def model_data(tmp_path):
    (tmp_path / "predictive_variables.yml").write_text(
        """
- name: intoxicant
  type: categorical
  allowed_values: [Alcohol, Street Drugs, Sedatives]
  dist:
    - value: Alcohol
      probability: 0.4
    - value: Street Drugs
      probability: 0.3
    - value: Sedatives
      probability: 0.3
- name: age
  type: continuous
  allowed_values: {min: 18, max: 100}
- name: cirrhosis
  type: categorical
  allowed_values: [Yes, No]
- name: second_diagnose
  type: categorical
  allowed_values: [Yes, No]
- name: gcs
  type: continuous
  allowed_values: {min: 3, max: 15}
- name: respiratory
  type: categorical
  allowed_values: [Yes, No]
- name: dysrhythmia
  type: categorical
  allowed_values: [Yes, No]
- name: hr
  type: continuous
  allowed_values: {min: 40, max: 250}
- name: sbp
  type: continuous
  allowed_values: {min: 60, max: 220}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    categorical = """
criteria: categorical
values:
  - {name: "Yes", score: 1}
  - {name: "No", score: 0}
"""
    (tmp_path / "intoxicant_score.yml").write_text(
        """
criteria: categorical
values:
  - {name: Alcohol, score: 1}
  - {name: Street Drugs, score: 2}
  - {name: Sedatives, score: 3}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    for name in ("cirrhosis", "second_diagnose", "respiratory", "dysrhythmia"):
        (tmp_path / f"{name}_score.yml").write_text(categorical, encoding="utf-8")
    for name, values in {
        "age": "[{min: 18, max: 100, score: 0}]",
        "gcs": "[{min: 14, max: 15, score: 0}, {min: 9, max: 13, score: 1}, {min: 7, max: 8, score: 2}, {min: 3, max: 6, score: 3}]",
        "hr": "[{min: 40, max: 250, score: 1}]",
        "sbp": "[{min: 60, max: 220, score: 1}]",
    }.items():
        (tmp_path / f"{name}_score.yml").write_text(
            f"criteria: range\nvalues: {values}\n", encoding="utf-8"
        )
    return load_model_data(tmp_path)
