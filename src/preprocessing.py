"""Preprocessing helpers for extracted EEG trial datasets."""

from __future__ import annotations

from typing import Any, Sequence

import mne
import numpy as np

from src.data_loader import TrialDataset

DEFAULT_L_FREQ = 8.0
DEFAULT_H_FREQ = 30.0


def validate_trial_dataset(dataset: TrialDataset) -> None:
    """Validate basic TrialDataset shape consistency before preprocessing."""

    if dataset.X.ndim != 3:
        raise ValueError(
            f"Expected X to have shape (n_trials, n_channels, n_times), got {dataset.X.shape}."
        )
    if dataset.X.shape[0] != dataset.y.shape[0]:
        raise ValueError("X and y have inconsistent trial counts.")
    if dataset.X.shape[0] != dataset.subject_ids.shape[0]:
        raise ValueError("X and subject_ids have inconsistent trial counts.")
    if dataset.X.shape[0] != dataset.run_ids.shape[0]:
        raise ValueError("X and run_ids have inconsistent trial counts.")
    if dataset.X.shape[0] != len(dataset.trial_event_names):
        raise ValueError("X and trial_event_names have inconsistent trial counts.")
    if dataset.X.shape[1] != len(dataset.channel_names):
        raise ValueError("X channel dimension does not match channel_names length.")
    if dataset.trial_count != dataset.X.shape[0]:
        raise ValueError("trial_count does not match the number of trials in X.")
    if dataset.sfreq <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {dataset.sfreq}.")


def resolve_channel_names(
    available_channels: Sequence[str],
    selected_channels: Sequence[str] | None = None,
) -> list[str]:
    """Resolve and validate the channel subset to keep."""

    channel_names = list(available_channels)
    if selected_channels is None:
        return channel_names

    resolved = [channel_name.strip() for channel_name in selected_channels if channel_name.strip()]
    if not resolved:
        raise ValueError("Channel selection was provided but no valid channel names were found.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate channel names are not allowed: {resolved}")

    missing = [channel_name for channel_name in resolved if channel_name not in channel_names]
    if missing:
        raise ValueError(
            f"Requested channels are not present in the dataset: {missing}"
        )
    return resolved


def select_trial_channels(
    X: np.ndarray,
    channel_names: Sequence[str],
    selected_channels: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Subset trial data by channel name while preserving trial order."""

    resolved_channels = resolve_channel_names(
        available_channels=channel_names,
        selected_channels=selected_channels,
    )
    if list(channel_names) == resolved_channels:
        return np.array(X, copy=True), resolved_channels

    channel_index = {channel_name: index for index, channel_name in enumerate(channel_names)}
    indices = [channel_index[channel_name] for channel_name in resolved_channels]
    return np.array(X[:, indices, :], copy=True), resolved_channels


def bandpass_filter_trials(
    X: np.ndarray,
    sfreq: float,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    verbose: str | bool = "ERROR",
) -> np.ndarray:
    """Apply band-pass filtering to trial data."""

    if l_freq <= 0:
        raise ValueError(f"l_freq must be positive, got {l_freq}.")
    if h_freq <= 0:
        raise ValueError(f"h_freq must be positive, got {h_freq}.")
    if l_freq >= h_freq:
        raise ValueError(f"l_freq must be smaller than h_freq, got {l_freq} >= {h_freq}.")

    return mne.filter.filter_data(
        data=X,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=verbose,
    )


def _build_preprocessing_metadata(
    dataset: TrialDataset,
    selected_channel_names: Sequence[str],
    l_freq: float,
    h_freq: float,
) -> dict[str, Any]:
    """Create metadata describing preprocessing choices."""

    metadata = dict(dataset.metadata)
    metadata["preprocessing"] = {
        "bandpass_hz": {
            "l_freq": float(l_freq),
            "h_freq": float(h_freq),
        },
        "selected_channels": list(selected_channel_names),
        "original_channel_count": int(len(dataset.channel_names)),
        "final_channel_count": int(len(selected_channel_names)),
    }
    return metadata


def preprocess_trial_dataset(
    dataset: TrialDataset,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    selected_channels: Sequence[str] | None = None,
    verbose: str | bool = "ERROR",
) -> TrialDataset:
    """Apply channel selection and band-pass filtering to a saved trial dataset."""

    validate_trial_dataset(dataset)

    X_selected, final_channel_names = select_trial_channels(
        X=dataset.X,
        channel_names=dataset.channel_names,
        selected_channels=selected_channels,
    )
    X_filtered = bandpass_filter_trials(
        X=X_selected,
        sfreq=dataset.sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=verbose,
    )
    metadata = _build_preprocessing_metadata(
        dataset=dataset,
        selected_channel_names=final_channel_names,
        l_freq=l_freq,
        h_freq=h_freq,
    )

    return TrialDataset(
        X=X_filtered,
        y=np.array(dataset.y, copy=True),
        subject_ids=np.array(dataset.subject_ids, copy=True),
        run_ids=np.array(dataset.run_ids, copy=True),
        trial_event_names=list(dataset.trial_event_names),
        selected_event_id=dict(dataset.selected_event_id),
        available_event_id=dict(dataset.available_event_id),
        label_map=dict(dataset.label_map),
        channel_names=list(final_channel_names),
        sfreq=float(dataset.sfreq),
        epoch_tmin=float(dataset.epoch_tmin),
        epoch_tmax=float(dataset.epoch_tmax),
        trial_count=int(dataset.trial_count),
        metadata=metadata,
    )


def summarize_preprocessed_dataset(dataset: TrialDataset) -> dict[str, Any]:
    """Return a compact summary for CLI printing or debugging."""

    label_values, label_counts = np.unique(dataset.y, return_counts=True)
    preprocessing = dataset.metadata.get("preprocessing", {})
    return {
        "X_shape": tuple(int(value) for value in dataset.X.shape),
        "y_shape": tuple(int(value) for value in dataset.y.shape),
        "trial_count": int(dataset.trial_count),
        "subjects": sorted(int(subject_id) for subject_id in np.unique(dataset.subject_ids)),
        "run_ids": sorted(int(run_id) for run_id in np.unique(dataset.run_ids)),
        "channel_names": list(dataset.channel_names),
        "sfreq": float(dataset.sfreq),
        "epoch_window": {
            "tmin": float(dataset.epoch_tmin),
            "tmax": float(dataset.epoch_tmax),
        },
        "selected_event_id": dict(dataset.selected_event_id),
        "label_map": dict(dataset.label_map),
        "label_counts": {
            int(label): int(count) for label, count in zip(label_values, label_counts)
        },
        "preprocessing": preprocessing,
    }
