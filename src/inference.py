"""Runtime inference helpers for saved EEG decoding artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.model import load_model_artifacts, predict_with_decoding_model


def build_label_name_map(label_map: dict[str, int] | None) -> dict[int, str]:
    """Invert a string-to-int label map into an int-to-string name map."""

    if not label_map:
        return {}
    return {int(label): event_name for event_name, label in label_map.items()}


def prepare_single_trial(trial: np.ndarray) -> np.ndarray:
    """Normalize one EEG trial to shape (1, n_channels, n_times)."""

    trial_array = np.asarray(trial, dtype=float)
    if trial_array.ndim == 2:
        return np.expand_dims(trial_array, axis=0)
    if trial_array.ndim == 3 and trial_array.shape[0] == 1:
        return trial_array
    raise ValueError(
        "Single-trial inference expects shape (n_channels, n_times) or "
        f"(1, n_channels, n_times), got {trial_array.shape}."
    )


def infer_single_trial(
    trial: np.ndarray,
    artifact_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run one EEG trial through fitted CSP, StandardScaler, and classifier."""

    prepared_trial = prepare_single_trial(trial)
    prediction_output = predict_with_decoding_model(
        csp=artifact_payload["csp"],
        scaler=artifact_payload["scaler"],
        classifier=artifact_payload["classifier"],
        X=prepared_trial,
    )

    metadata = artifact_payload.get("metadata", {})
    label_name_map = build_label_name_map(metadata.get("label_map"))

    predicted_label = int(prediction_output["y_pred"][0])
    probabilities = prediction_output.get("y_proba")
    confidence = prediction_output.get("confidence")
    positive_class_probability = prediction_output.get("positive_class_probability")

    return {
        "predicted_label": predicted_label,
        "predicted_label_name": label_name_map.get(predicted_label),
        "confidence": None if confidence is None else float(confidence[0]),
        "probabilities": (
            None if probabilities is None else probabilities[0].astype(float).tolist()
        ),
        "positive_class_probability": (
            None
            if positive_class_probability is None
            else float(positive_class_probability[0])
        ),
    }


def load_inference_artifacts(artifact_prefix: str) -> dict[str, Any]:
    """Load the saved fitted objects and metadata required for runtime inference."""

    payload = load_model_artifacts(artifact_prefix)
    if payload.get("scaler") is None:
        raise ValueError(
            "Loaded artifact does not include a fitted StandardScaler. "
            "Retrain the model with the scaled Phase 3 pipeline."
        )
    return payload

