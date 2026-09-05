from __future__ import annotations

import pytest

from toxsim import ModelDataError, create_patient, create_patients, load_model_data


def test_loads_packaged_canonical_model_data():
    model_data = load_model_data()
    names = {variable["name"] for variable in model_data.predictive_variables}

    assert names == set(model_data.scores)
    assert names == {
        "age",
        "cirrhosis",
        "dysrhythmia",
        "gcs",
        "hr",
        "intoxicant",
        "respiratory",
        "sbp",
        "second_diagnose",
    }


def test_missing_override_data_has_actionable_error(tmp_path):
    with pytest.raises(ModelDataError, match="model-data-dir"):
        load_model_data(tmp_path)


def test_seeded_generation_is_deterministic():
    model_data = load_model_data()
    assert create_patient(model_data, seed=42) == create_patient(model_data, seed=42)
    assert create_patients(model_data, 3, seed=42) == create_patients(model_data, 3, seed=42)
    assert create_patient(model_data, seed=42) != create_patient(model_data, seed=43)


def test_patient_scores_and_continuous_output_invariants():
    model_data = load_model_data()
    patient = create_patient(model_data, seed=7)

    assert patient["risk"] == sum(feature["score"] for feature in patient["presentation"])
    assert set(patient["extras"]) == {"severity", "hr_true", "sbp_true"}
    for feature in patient["presentation"]:
        assert {"name", "value", "score", "in_original_range"} <= set(feature)
        if feature["name"] in {"age", "hr", "sbp"}:
            assert {"true_value", "model_range"} <= set(feature)
            assert feature["in_original_range"] == (
                feature["model_range"]["min"]
                <= feature["true_value"]
                <= feature["model_range"]["max"]
            )


def test_create_patients_rejects_negative_count():
    model_data = load_model_data()
    with pytest.raises(ValueError, match="non-negative"):
        create_patients(model_data, -1)
