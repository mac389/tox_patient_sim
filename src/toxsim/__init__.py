"""Public API for context-aware synthetic poisoned-patient generation."""

from .simulator import ModelData, ModelDataError, create_patient, create_patients, load_model_data

__all__ = [
    "ModelData",
    "ModelDataError",
    "create_patient",
    "create_patients",
    "load_model_data",
]
