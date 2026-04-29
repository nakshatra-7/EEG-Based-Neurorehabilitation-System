"""Trial-by-trial session state and simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data_loader import TrialDataset
from src.inference import build_label_name_map, infer_single_trial


@dataclass
class SessionState:
    """Mutable runtime state for one sequential neurorehabilitation session."""

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    run_ids: np.ndarray
    trial_event_names: list[str]
    label_name_map: dict[int, str]
    current_trial_index: int = 0
    total_trials: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    correct_count: int = 0
    incorrect_count: int = 0
    prediction_counts: dict[int, int] = field(default_factory=dict)
    last_prediction: int | None = None
    last_confidence: float | None = None
    last_correctness: bool | None = None


def subset_trial_dataset(
    dataset: TrialDataset,
    subject_id: int | None = None,
    max_trials: int | None = None,
) -> TrialDataset:
    """Create an ordered subset of a TrialDataset for runtime simulation."""

    indices = np.arange(dataset.trial_count)
    if subject_id is not None:
        subject_mask = dataset.subject_ids == int(subject_id)
        indices = indices[subject_mask]
        if indices.size == 0:
            raise ValueError(f"No trials found for subject_id={subject_id}.")

    if max_trials is not None:
        if max_trials <= 0:
            raise ValueError(f"max_trials must be positive, got {max_trials}.")
        indices = indices[:max_trials]

    return TrialDataset(
        X=np.array(dataset.X[indices], copy=True),
        y=np.array(dataset.y[indices], copy=True),
        subject_ids=np.array(dataset.subject_ids[indices], copy=True),
        run_ids=np.array(dataset.run_ids[indices], copy=True),
        trial_event_names=[dataset.trial_event_names[index] for index in indices],
        selected_event_id=dict(dataset.selected_event_id),
        available_event_id=dict(dataset.available_event_id),
        label_map=dict(dataset.label_map),
        channel_names=list(dataset.channel_names),
        sfreq=float(dataset.sfreq),
        epoch_tmin=float(dataset.epoch_tmin),
        epoch_tmax=float(dataset.epoch_tmax),
        trial_count=int(len(indices)),
        metadata=dict(dataset.metadata),
    )


def initialize_session(dataset: TrialDataset) -> SessionState:
    """Initialize session state from an ordered trial dataset."""

    label_name_map = build_label_name_map(dataset.label_map)
    prediction_counts = {int(label): 0 for label in sorted(set(dataset.label_map.values()))}
    return SessionState(
        X=np.array(dataset.X, copy=False),
        y=np.array(dataset.y, copy=False),
        subject_ids=np.array(dataset.subject_ids, copy=False),
        run_ids=np.array(dataset.run_ids, copy=False),
        trial_event_names=list(dataset.trial_event_names),
        label_name_map=label_name_map,
        total_trials=int(dataset.trial_count),
        prediction_counts=prediction_counts,
    )


def has_remaining_trials(state: SessionState) -> bool:
    """Return whether the session still has unprocessed trials."""

    return state.current_trial_index < state.total_trials


def step_session(
    state: SessionState,
    artifact_payload: dict[str, Any],
) -> dict[str, Any]:
    """Process exactly one trial and update the session state."""

    if not has_remaining_trials(state):
        raise IndexError("No trials remaining in the session.")

    trial_index = state.current_trial_index
    true_label = int(state.y[trial_index])
    inference = infer_single_trial(
        trial=state.X[trial_index],
        artifact_payload=artifact_payload,
    )

    predicted_label = int(inference["predicted_label"])
    correct = predicted_label == true_label
    if correct:
        state.correct_count += 1
    else:
        state.incorrect_count += 1

    state.prediction_counts[predicted_label] = state.prediction_counts.get(predicted_label, 0) + 1
    state.last_prediction = predicted_label
    state.last_confidence = inference["confidence"]
    state.last_correctness = bool(correct)

    trial_result = {
        "trial_index": int(trial_index),
        "subject_id": int(state.subject_ids[trial_index]),
        "run_id": int(state.run_ids[trial_index]),
        "true_label": true_label,
        "true_label_name": state.label_name_map.get(true_label),
        "predicted_label": predicted_label,
        "predicted_label_name": inference["predicted_label_name"],
        "confidence": inference["confidence"],
        "probabilities": inference["probabilities"],
        "positive_class_probability": inference["positive_class_probability"],
        "correct": bool(correct),
    }
    state.history.append(trial_result)
    state.current_trial_index += 1
    return trial_result


def run_session(
    state: SessionState,
    artifact_payload: dict[str, Any],
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Run the session forward for all remaining trials or a bounded number of steps."""

    if max_steps is not None and max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")

    results: list[dict[str, Any]] = []
    while has_remaining_trials(state):
        if max_steps is not None and len(results) >= max_steps:
            break
        results.append(step_session(state, artifact_payload))
    return results


def build_session_summary(state: SessionState) -> dict[str, Any]:
    """Return a compact serializable session summary."""

    processed_trials = len(state.history)
    accuracy = state.correct_count / processed_trials if processed_trials > 0 else 0.0
    return {
        "current_trial_index": int(state.current_trial_index),
        "total_trials": int(state.total_trials),
        "processed_trials": int(processed_trials),
        "remaining_trials": int(state.total_trials - state.current_trial_index),
        "correct_count": int(state.correct_count),
        "incorrect_count": int(state.incorrect_count),
        "accuracy": float(accuracy),
        "prediction_counts": {
            int(label): int(count) for label, count in sorted(state.prediction_counts.items())
        },
        "last_prediction": state.last_prediction,
        "last_prediction_name": (
            None if state.last_prediction is None else state.label_name_map.get(state.last_prediction)
        ),
        "last_confidence": state.last_confidence,
        "last_correctness": state.last_correctness,
        "subjects_present": sorted(int(subject_id) for subject_id in np.unique(state.subject_ids)),
        "run_ids_present": sorted(int(run_id) for run_id in np.unique(state.run_ids)),
    }

