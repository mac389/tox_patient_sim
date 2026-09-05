from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import UUID

import numpy as np
import yaml
from scipy.stats import truncnorm


class ModelDataError(ValueError):
    """Raised when synthetic-patient model configuration cannot be loaded."""


@dataclass(frozen=True)
class ModelData:
    """Predictive variables and their corresponding scoring tables."""

    predictive_variables: tuple[dict[str, Any], ...]
    scores: Mapping[str, dict[str, Any]]


@dataclass
class _Context:
    intoxicant: str | None = None
    severity: int | None = None
    gcs: int | None = None
    respiratory: str | None = None
    dysrhythmia: str | None = None
    hr_true: int | None = None
    sbp_true: int | None = None


def load_model_data(model_data_dir: str | Path | None = None) -> ModelData:
    """Load model YAML data from a directory or packaged resources."""
    if model_data_dir is None:
        root = resources.files("toxsim").joinpath("data", "model")
        source = "packaged toxsim model data"
    else:
        root = Path(model_data_dir).expanduser()
        source = f"model-data directory {root}"

    predictive_path = root.joinpath("predictive_variables.yml")
    if not predictive_path.is_file():
        raise ModelDataError(
            f"Could not find predictive_variables.yml in {source}. "
            "Provide a directory containing predictive_variables.yml and one "
            "<variable-name>_score.yml file for every predictive variable, "
            "for example load_model_data('/path/to/model-data') or "
            "toxsim-generate --model-data-dir /path/to/model-data."
        )

    predictive_variables = _load_yaml_list(predictive_path)
    scores: dict[str, dict[str, Any]] = {}
    for variable in predictive_variables:
        name = variable.get("name")
        if not isinstance(name, str) or not name:
            raise ModelDataError(
                f"{predictive_path} contains a variable without a non-empty string name."
            )
        score_path = root.joinpath(f"{name}_score.yml")
        if not score_path.is_file():
            raise ModelDataError(
                f"Missing score file {name}_score.yml in {source}. "
                "Supply a score YAML file for every predictive variable."
            )
        score = _load_yaml_mapping(score_path)
        if "criteria" not in score or "values" not in score:
            raise ModelDataError(f"{score_path} must define 'criteria' and 'values'.")
        scores[name] = score

    return ModelData(tuple(predictive_variables), scores)


def create_patient(
    predictive_variables: ModelData | Sequence[Mapping[str, Any]] | None = None,
    scores: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    model_data: ModelData | None = None,
    model_data_dir: str | Path | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Create one context-aware synthetic patient.

    With no model arguments, bundled model data is loaded lazily. To provide
    custom data, pass either *model_data*, *model_data_dir*, or both
    *predictive_variables* and *scores*. Passing a :class:`ModelData` as the
    first positional argument is retained for compatibility.

    Supply either *seed* or *rng*, not both. A seed makes all generated fields,
    including the patient ID, deterministic.
    """
    resolved_model_data = _resolve_model_data(
        predictive_variables=predictive_variables,
        scores=scores,
        model_data=model_data,
        model_data_dir=model_data_dir,
    )
    generator = _resolve_rng(seed, rng)
    variables = _variables_by_name(resolved_model_data)
    order = _generation_order(variables)

    for _ in range(10):
        context = _Context()
        presentation = [
            _create_feature(variables[name], context, resolved_model_data.scores, generator)
            for name in order
        ]
        if _plausible(context, generator):
            return {
                "presentation": presentation,
                "patient_id": str(_random_uuid(generator)),
                "extras": {
                    "severity": context.severity,
                    "hr_true": context.hr_true,
                    "sbp_true": context.sbp_true,
                },
                "risk": int(sum(feature["score"] for feature in presentation)),
            }

    # The tenth sample remains valid even if it failed a probabilistic
    # plausibility screen; rejecting it would needlessly hide a valid patient.
    return {
        "presentation": presentation,
        "patient_id": str(_random_uuid(generator)),
        "extras": {
            "severity": context.severity,
            "hr_true": context.hr_true,
            "sbp_true": context.sbp_true,
        },
        "risk": int(sum(feature["score"] for feature in presentation)),
    }


def create_patients(
    model_data: ModelData | None = None,
    count: int = 1,
    *,
    predictive_variables: Sequence[Mapping[str, Any]] | None = None,
    scores: Mapping[str, Mapping[str, Any]] | None = None,
    model_data_dir: str | Path | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Create *count* patients using bundled or explicitly supplied model data.

    The model source arguments have the same behavior as :func:`create_patient`.
    Supplying *model_data* positionally remains compatible with earlier releases.
    """
    if count < 0:
        raise ValueError("count must be non-negative.")
    resolved_model_data = _resolve_model_data(
        predictive_variables=predictive_variables,
        scores=scores,
        model_data=model_data,
        model_data_dir=model_data_dir,
    )
    generator = _resolve_rng(seed, rng)
    return [create_patient(resolved_model_data, rng=generator) for _ in range(count)]


def score_from_value(value: Any, score_table: Mapping[str, Any]) -> int:
    """Return the score associated with a categorical or ranged value."""
    criteria = score_table["criteria"]
    values = score_table["values"]
    if criteria == "categorical":
        normalized = _normalize_bool(value)
        for item in values:
            if normalized == item["name"]:
                return int(item["score"])
        raise ValueError(f"Value {value!r} is not in the categorical score table.")
    if criteria == "range":
        for item in values:
            if float(item["min"]) <= float(value) <= float(item["max"]):
                return int(item["score"])
        raise ValueError(f"Value {value!r} is not in the range score table.")
    raise ValueError(f"Unknown score criteria: {criteria!r}.")


def _load_yaml_list(path: Any) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ModelDataError(f"{path} must contain a YAML list of variable mappings.")
    return data


def _load_yaml_mapping(path: Any) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ModelDataError(f"{path} must contain a YAML mapping.")
    return data


def _resolve_rng(
    seed: int | None, rng: np.random.Generator | None
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    return rng if rng is not None else np.random.default_rng(seed)


def _resolve_model_data(
    *,
    predictive_variables: ModelData | Sequence[Mapping[str, Any]] | None,
    scores: Mapping[str, Mapping[str, Any]] | None,
    model_data: ModelData | None,
    model_data_dir: str | Path | None,
) -> ModelData:
    has_explicit_tables = predictive_variables is not None or scores is not None
    if model_data is not None:
        if has_explicit_tables or model_data_dir is not None:
            raise ModelDataError(
                "Pass only one model source: model_data, model_data_dir, or "
                "predictive_variables with scores."
            )
        return model_data
    if isinstance(predictive_variables, ModelData):
        if scores is not None or model_data_dir is not None:
            raise ModelDataError(
                "A positional ModelData cannot be combined with scores or model_data_dir."
            )
        return predictive_variables
    if has_explicit_tables:
        if model_data_dir is not None:
            raise ModelDataError(
                "Pass either model_data_dir or predictive_variables with scores, not both."
            )
        if predictive_variables is None or scores is None:
            raise ModelDataError(
                "Explicit model data requires both predictive_variables and scores."
            )
        return ModelData(
            tuple(dict(variable) for variable in predictive_variables),
            {name: dict(score) for name, score in scores.items()},
        )
    return load_model_data(model_data_dir)


def _variables_by_name(model_data: ModelData) -> dict[str, dict[str, Any]]:
    variables: dict[str, dict[str, Any]] = {}
    for variable in model_data.predictive_variables:
        name = variable.get("name")
        if not isinstance(name, str) or not name:
            raise ModelDataError(
                "Every predictive variable must have a non-empty string name."
            )
        if name in variables:
            raise ModelDataError(f"Duplicate predictive variable name: {name}.")
        if name not in model_data.scores:
            raise ModelDataError(
                f"No score table was supplied for predictive variable {name}."
            )
        variables[name] = variable
    return variables


def _generation_order(variables: Mapping[str, dict[str, Any]]) -> list[str]:
    contextual = [
        "intoxicant",
        "age",
        "cirrhosis",
        "second_diagnose",
        "gcs",
        "respiratory",
        "dysrhythmia",
        "hr",
        "sbp",
    ]
    return [name for name in contextual if name in variables] + [
        name for name in variables if name not in contextual
    ]


def _create_feature(
    variable: Mapping[str, Any],
    context: _Context,
    scores: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    name = variable["name"]
    value, true_value = _simulate_value_with_context(variable, context, scores, rng)
    feature: dict[str, Any] = {
        "name": name,
        "value": value,
        "score": score_from_value(value, scores[name]),
    }
    if variable["type"] == "categorical":
        feature["in_original_range"] = True
    else:
        feature["true_value"] = true_value
        feature["in_original_range"] = _is_value_in_range(true_value, variable)
        feature["model_range"] = {
            "min": float(variable["allowed_values"]["min"]),
            "max": float(variable["allowed_values"]["max"]),
        }
    return feature


def _simulate_value_with_context(
    variable: Mapping[str, Any],
    context: _Context,
    scores: Mapping[str, Mapping[str, Any]],
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    name = variable["name"]
    if name == "intoxicant":
        value = _simulate_categorical(variable, rng)
        context.intoxicant = str(value)
        return value, None
    if name == "age":
        bounds = variable["allowed_values"]
        value = int(
            round(
                int(bounds["min"])
                + rng.beta(2.2, 3.0) * (int(bounds["max"]) - int(bounds["min"]))
            )
        )
        return value, value

    if context.severity is None and context.intoxicant is not None:
        context.severity = _sample_severity(context.intoxicant, rng)

    intoxicant = context.intoxicant or "Polysubstance"
    severity = context.severity if context.severity is not None else 1
    if name == "gcs":
        value = _sample_gcs_from_bins(scores["gcs"], intoxicant, severity, rng)
        context.gcs = value
        return value, value
    if name == "respiratory":
        value = _sample_respiratory(context.gcs or 15, intoxicant, severity, rng)
        context.respiratory = value
        return value, None
    if name == "dysrhythmia":
        value = _sample_dysrhythmia(intoxicant, severity, rng)
        context.dysrhythmia = value
        return value, None
    if name == "hr":
        true_value = _sample_hr_true(
            intoxicant,
            severity,
            context.dysrhythmia or "No",
            context.respiratory or "No",
            rng,
        )
        value = int(true_value)
        context.hr_true = value
        return value, true_value
    if name == "sbp":
        true_value = _sample_sbp_true(intoxicant, severity, context.hr_true or 90, rng)
        value = int(round(true_value / 2.0) * 2)
        context.sbp_true = value
        return value, true_value

    if variable["type"] == "categorical":
        return _simulate_categorical(variable, rng), None
    if variable["type"] == "continuous":
        value = _simulate_continuous(variable, rng)
        return value, value
    raise ValueError(f"Unknown variable type: {variable.get('type')!r}.")


def _simulate_categorical(variable: Mapping[str, Any], rng: np.random.Generator) -> Any:
    distribution = variable.get("dist")
    if distribution:
        options = [item["value"] for item in distribution]
        probabilities = [float(item["probability"]) for item in distribution]
        return _normalize_bool(_weighted_choice(options, probabilities, rng))
    return _normalize_bool(rng.choice(variable["allowed_values"]).item())


def _simulate_continuous(variable: Mapping[str, Any], rng: np.random.Generator) -> int:
    bounds = variable["allowed_values"]
    low, high = float(bounds["min"]), float(bounds["max"])
    return _truncnorm_int(low, high, (low + high) / 2.0, (high - low) / 6.0, rng)


def _sample_severity(intoxicant: str, rng: np.random.Generator) -> int:
    probabilities = {
        "Alcohol": [0.60, 0.25, 0.12, 0.03],
        "Analgesic": [0.45, 0.28, 0.18, 0.09],
        "Antidepressant": [0.45, 0.30, 0.18, 0.07],
        "Street Drugs": [0.45, 0.30, 0.17, 0.08],
        "Sedatives": [0.40, 0.28, 0.22, 0.10],
        "CO, As, CN": [0.35, 0.30, 0.22, 0.13],
        "Toxins NOS": [0.45, 0.30, 0.18, 0.07],
        "Polysubstance": [0.30, 0.30, 0.25, 0.15],
    }.get(intoxicant, [0.45, 0.30, 0.18, 0.07])
    return int(_weighted_choice([0, 1, 2, 3], probabilities, rng))


def _sample_gcs_from_bins(
    score_table: Mapping[str, Any],
    intoxicant: str,
    severity: int,
    rng: np.random.Generator,
) -> int:
    bins = sorted(score_table["values"], key=lambda item: item["max"], reverse=True)
    probabilities = np.array(
        {
            0: [0.82, 0.14, 0.03, 0.01],
            1: [0.55, 0.28, 0.12, 0.05],
            2: [0.28, 0.35, 0.22, 0.15],
            3: [0.06, 0.22, 0.34, 0.38],
        }[severity],
        dtype=float,
    )
    if intoxicant in {"Sedatives", "Analgesic", "Alcohol", "Polysubstance"}:
        probabilities *= [0.85, 1.10, 1.15, 1.25]
    elif intoxicant == "Street Drugs":
        probabilities *= [1.10, 1.05, 0.90, 0.80]
    selected = _weighted_choice(bins, probabilities, rng)
    return int(rng.integers(int(selected["min"]), int(selected["max"]) + 1))


def _sample_respiratory(
    gcs: int, intoxicant: str, severity: int, rng: np.random.Generator
) -> str:
    probability = 0.05 + 0.07 * severity
    if gcs <= 8:
        probability += 0.45
    if gcs <= 6:
        probability += 0.20
    if intoxicant in {"Sedatives", "Analgesic", "Alcohol", "Polysubstance"}:
        probability += 0.12
    if intoxicant == "CO, As, CN":
        probability += 0.05
    return "Yes" if rng.random() < min(probability, 0.95) else "No"


def _sample_dysrhythmia(
    intoxicant: str, severity: int, rng: np.random.Generator
) -> str:
    probability = 0.06 + 0.06 * severity
    if intoxicant in {"Antidepressant", "Street Drugs", "CO, As, CN"}:
        probability += 0.10
    if intoxicant == "Polysubstance":
        probability += 0.07
    return "Yes" if rng.random() < min(probability, 0.60) else "No"


def _sample_hr_true(
    intoxicant: str,
    severity: int,
    dysrhythmia: str,
    respiratory: str,
    rng: np.random.Generator,
) -> int:
    abnormal_mean = {
        "Street Drugs": 145,
        "Antidepressant": 120,
        "Sedatives": 62,
        "Analgesic": 62,
        "Alcohol": 62,
    }.get(intoxicant, 110)
    probability = 0.10 + 0.15 * severity
    if dysrhythmia == "Yes":
        probability += 0.20
    if respiratory == "Yes":
        probability += 0.10
    mean, deviation = (
        (abnormal_mean, 28) if rng.random() < min(probability, 0.90) else (85, 12)
    )
    return int(round(np.clip(rng.normal(mean, deviation), 40, 250)))


def _sample_sbp_true(
    intoxicant: str, severity: int, hr: int, rng: np.random.Generator
) -> int:
    mean = 125.0
    if intoxicant == "Street Drugs":
        mean += 18
    if intoxicant in {"Sedatives", "Analgesic", "Alcohol"}:
        mean -= 10
    if intoxicant == "Polysubstance":
        mean -= 6
    mean -= 9 * severity
    if severity >= 2 and hr >= 140:
        mean -= 6
    return int(round(np.clip(rng.normal(mean, 14 + 6 * severity), 60, 220)))


def _plausible(context: _Context, rng: np.random.Generator) -> bool:
    if (
        context.gcs is None
        or context.respiratory is None
        or context.hr_true is None
        or context.sbp_true is None
    ):
        return True
    if context.gcs <= 6 and context.respiratory == "No":
        return bool(rng.random() < 0.25)
    if context.respiratory == "Yes" and context.gcs >= 14:
        return bool(rng.random() < 0.20)
    if context.sbp_true <= 80 and context.hr_true <= 55:
        return bool(rng.random() < 0.30)
    return True


def _truncnorm_int(
    low: float, high: float, mean: float, deviation: float, rng: np.random.Generator
) -> int:
    deviation = max(deviation, 1e-6)
    value = truncnorm.rvs(
        (low - mean) / deviation,
        (high - mean) / deviation,
        loc=mean,
        scale=deviation,
        random_state=rng,
    )
    return int(round(value))


def _weighted_choice(
    options: Sequence[Any], probabilities: Sequence[float], rng: np.random.Generator
) -> Any:
    weights = np.asarray(probabilities, dtype=float)
    if (
        len(options) != len(weights)
        or not len(options)
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "Options and non-negative probabilities with a positive sum are required."
        )
    index = rng.choice(len(options), p=weights / weights.sum())
    return options[int(index)]


def _is_value_in_range(value: Any, variable: Mapping[str, Any]) -> bool:
    if value is None:
        return True
    bounds = variable["allowed_values"]
    return float(bounds["min"]) <= float(value) <= float(bounds["max"])


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    return value


def _random_uuid(rng: np.random.Generator) -> UUID:
    raw = bytearray(rng.bytes(16))
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return UUID(bytes=bytes(raw))
