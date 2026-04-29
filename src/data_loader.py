"""Dataset loading, event inspection, epoch extraction, and save utilities."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
from mne.datasets import eegbci

from src.config import (
    DATASET_STORAGE_PATH,
    DEFAULT_EPOCH_TMAX,
    DEFAULT_EPOCH_TMIN,
    DEFAULT_RUNS,
)


@dataclass(frozen=True)
class RunSegment:
    """Sample span for one run inside a concatenated subject recording."""

    run_id: int
    start_sample: int
    stop_sample: int
    file_path: str


@dataclass
class SubjectRawData:
    """Raw EEG data plus run metadata for one subject."""

    subject_id: int
    run_ids: list[int]
    raw: mne.io.BaseRaw
    run_segments: list[RunSegment]


@dataclass
class TrialDataset:
    """Packaged epochs, labels, and metadata for downstream phases."""

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    run_ids: np.ndarray
    trial_event_names: list[str]
    selected_event_id: dict[str, int]
    available_event_id: dict[str, int]
    label_map: dict[str, int]
    channel_names: list[str]
    sfreq: float
    epoch_tmin: float
    epoch_tmax: float
    trial_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for persistence or inspection."""

        packaged = dict(self.metadata)
        packaged.update(
            {
                "selected_event_id": {
                    key: int(value) for key, value in self.selected_event_id.items()
                },
                "available_event_id": {
                    key: int(value) for key, value in self.available_event_id.items()
                },
                "label_map": {key: int(value) for key, value in self.label_map.items()},
                "channel_names": list(self.channel_names),
                "sfreq": float(self.sfreq),
                "epoch_window": {
                    "tmin": float(self.epoch_tmin),
                    "tmax": float(self.epoch_tmax),
                },
                "trial_count": int(self.trial_count),
            }
        )
        return packaged


def _validate_subject_ids(subject_ids: Sequence[int]) -> list[int]:
    """Validate subject IDs and return a normalized list."""

    normalized = [int(subject_id) for subject_id in subject_ids]
    if not normalized:
        raise ValueError("At least one subject ID is required.")
    invalid = [subject_id for subject_id in normalized if subject_id <= 0]
    if invalid:
        raise ValueError(f"Subject IDs must be positive integers: {invalid}")
    return normalized


def _validate_run_ids(run_ids: Sequence[int] | None) -> list[int]:
    """Validate requested run IDs and apply defaults when missing."""

    resolved = DEFAULT_RUNS if run_ids is None else [int(run_id) for run_id in run_ids]
    if not resolved:
        raise ValueError("At least one run ID is required.")
    invalid = [run_id for run_id in resolved if run_id <= 0]
    if invalid:
        raise ValueError(f"Run IDs must be positive integers: {invalid}")
    return resolved


def _resolve_data_path(data_path: str | Path | None) -> Path:
    """Resolve the dataset storage directory and create it if needed."""

    resolved = Path(data_path) if data_path is not None else DATASET_STORAGE_PATH
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_subject_raw(
    subject_id: int,
    run_ids: Sequence[int] | None = None,
    data_path: str | Path | None = None,
    preload: bool = True,
    verbose: str | bool = "ERROR",
) -> SubjectRawData:
    """Load and concatenate EEGBCI runs for one subject."""

    subject = _validate_subject_ids([subject_id])[0]
    runs = _validate_run_ids(run_ids)
    dataset_path = _resolve_data_path(data_path)

    try:
        # MNE changed this API from `subject` to `subjects`; fallback keeps compatibility.
        try:
            edf_paths = eegbci.load_data(
                subjects=subject,
                runs=runs,
                path=str(dataset_path),
            )
        except TypeError:
            edf_paths = eegbci.load_data(
                subject=subject,
                runs=runs,
                path=str(dataset_path),
            )
    except Exception as exc:  # pragma: no cover - depends on external data access
        message = (
            f"Failed to load EEGBCI data for subject {subject} and runs {runs}. "
            "Verify the subject/run selection and dataset path."
        )
        raise ValueError(message) from exc

    if len(edf_paths) != len(runs):
        raise ValueError(
            f"Expected {len(runs)} run files for subject {subject}, got {len(edf_paths)}."
        )

    run_raws: list[mne.io.BaseRaw] = []
    run_segments: list[RunSegment] = []
    start_sample = 0

    for run_id, edf_path in zip(runs, edf_paths):
        raw_run = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)
        eegbci.standardize(raw_run)

        stop_sample = start_sample + raw_run.n_times - 1
        run_segments.append(
            RunSegment(
                run_id=int(run_id),
                start_sample=int(start_sample),
                stop_sample=int(stop_sample),
                file_path=str(edf_path),
            )
        )
        run_raws.append(raw_run)
        start_sample = stop_sample + 1

    raw = (
        mne.concatenate_raws(run_raws, verbose=verbose)
        if len(run_raws) > 1
        else run_raws[0]
    )

    return SubjectRawData(
        subject_id=subject,
        run_ids=runs,
        raw=raw,
        run_segments=run_segments,
    )


def load_subject_raws(
    subject_ids: Sequence[int],
    run_ids: Sequence[int] | None = None,
    data_path: str | Path | None = None,
    preload: bool = True,
    verbose: str | bool = "ERROR",
) -> dict[int, SubjectRawData]:
    """Load raw EEGBCI data for multiple subjects."""

    subjects = _validate_subject_ids(subject_ids)
    return {
        subject_id: load_subject_raw(
            subject_id=subject_id,
            run_ids=run_ids,
            data_path=data_path,
            preload=preload,
            verbose=verbose,
        )
        for subject_id in subjects
    }


def summarize_annotations(raw: mne.io.BaseRaw) -> list[dict[str, Any]]:
    """Summarize annotation descriptions and durations in the raw recording."""

    if len(raw.annotations) == 0:
        return []

    counts = Counter(raw.annotations.description)
    total_durations: dict[str, float] = {}
    for description, duration in zip(raw.annotations.description, raw.annotations.duration):
        total_durations[description] = total_durations.get(description, 0.0) + float(duration)

    summary = []
    for description in sorted(counts):
        count = counts[description]
        summary.append(
            {
                "description": description,
                "count": int(count),
                "total_duration_s": round(total_durations.get(description, 0.0), 6),
                "mean_duration_s": round(total_durations.get(description, 0.0) / count, 6),
            }
        )
    return summary


def extract_events_from_annotations(
    raw: mne.io.BaseRaw,
    event_id: Mapping[str, int] | None = None,
    verbose: str | bool = "ERROR",
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract MNE events and the discovered event-ID mapping from annotations."""

    events, discovered_event_id = mne.events_from_annotations(
        raw,
        event_id=event_id,
        verbose=verbose,
    )
    if events.size == 0:
        raise ValueError("No events were extracted from annotations.")
    return events, {key: int(value) for key, value in discovered_event_id.items()}


def summarize_events(
    events: np.ndarray,
    event_id: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Count extracted events for each annotation label."""

    code_to_name = {int(code): name for name, code in event_id.items()}
    counts = Counter(int(code) for code in events[:, 2])
    summary = []
    for code in sorted(counts):
        summary.append(
            {
                "event_name": code_to_name.get(code, f"unknown_{code}"),
                "event_code": int(code),
                "count": int(counts[code]),
            }
        )
    return summary


def resolve_selected_event_id(
    available_event_id: Mapping[str, int],
    event_names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Select a subset of events for epoch extraction."""

    if event_names is None:
        return {
            name: int(code)
            for name, code in sorted(available_event_id.items(), key=lambda item: item[1])
        }

    selected = [name.strip() for name in event_names if name.strip()]
    if not selected:
        raise ValueError("Event selection was provided but no valid event names were found.")

    missing = [name for name in selected if name not in available_event_id]
    if missing:
        raise ValueError(
            f"Requested events not found in available annotations/events: {missing}"
        )

    return {name: int(available_event_id[name]) for name in selected}


def build_default_label_map(selected_event_id: Mapping[str, int]) -> dict[str, int]:
    """Build a deterministic label map when no explicit mapping is provided."""

    ordered_names = list(selected_event_id.keys())
    return {name: index for index, name in enumerate(ordered_names)}


def validate_label_map(
    label_map: Mapping[str, int] | None,
    selected_event_id: Mapping[str, int],
) -> dict[str, int]:
    """Validate explicit label mapping against the selected events."""

    if label_map is None:
        return build_default_label_map(selected_event_id)

    normalized = {name.strip(): int(value) for name, value in label_map.items()}
    missing = [name for name in selected_event_id if name not in normalized]
    extra = [name for name in normalized if name not in selected_event_id]

    if missing:
        raise ValueError(
            f"Label map is missing selected events: {missing}. Provide labels for every selected event."
        )
    if extra:
        raise ValueError(
            f"Label map contains events that are not selected for extraction: {extra}"
        )
    return normalized


def infer_run_ids_for_samples(
    sample_indices: Sequence[int],
    run_segments: Sequence[RunSegment],
) -> np.ndarray:
    """Map event sample indices back to the run they came from."""

    run_ids = np.full(len(sample_indices), -1, dtype=int)
    for index, sample in enumerate(sample_indices):
        sample_int = int(sample)
        for segment in run_segments:
            if segment.start_sample <= sample_int <= segment.stop_sample:
                run_ids[index] = segment.run_id
                break

    if np.any(run_ids < 0):
        unresolved = [int(sample_indices[index]) for index, value in enumerate(run_ids) if value < 0]
        raise ValueError(
            f"Could not map some event samples to source runs: {unresolved[:10]}"
        )
    return run_ids


def create_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Mapping[str, int],
    tmin: float = DEFAULT_EPOCH_TMIN,
    tmax: float = DEFAULT_EPOCH_TMAX,
    baseline: tuple[float, float] | None = None,
    picks: str | Sequence[str] | None = "eeg",
    preload: bool = True,
    verbose: str | bool = "ERROR",
) -> mne.Epochs:
    """Create MNE epochs around selected events."""

    if not event_id:
        raise ValueError("No event IDs were provided for epoch creation.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=dict(event_id),
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=preload,
        reject_by_annotation=True,
        verbose=verbose,
    )
    if len(epochs) == 0:
        raise ValueError("Epoch extraction produced zero trials.")
    return epochs


def build_trial_dataset(
    subject_data: SubjectRawData,
    event_names: Sequence[str] | None = None,
    label_map: Mapping[str, int] | None = None,
    tmin: float = DEFAULT_EPOCH_TMIN,
    tmax: float = DEFAULT_EPOCH_TMAX,
    baseline: tuple[float, float] | None = None,
    picks: str | Sequence[str] | None = "eeg",
    verbose: str | bool = "ERROR",
) -> TrialDataset:
    """Extract epochs and package trials for one subject."""

    if event_names is None and label_map is not None:
        event_names = list(label_map.keys())

    events, available_event_id = extract_events_from_annotations(subject_data.raw, verbose=verbose)
    selected_event_id = resolve_selected_event_id(available_event_id, event_names=event_names)
    resolved_label_map = validate_label_map(label_map, selected_event_id)

    selected_codes = set(selected_event_id.values())
    selected_mask = np.isin(events[:, 2], list(selected_codes))
    selected_events = events[selected_mask]
    if selected_events.size == 0:
        raise ValueError("Selected event names did not match any extracted events.")

    all_event_run_ids = infer_run_ids_for_samples(events[:, 0], subject_data.run_segments)
    selected_event_run_ids = all_event_run_ids[selected_mask]

    code_to_name = {int(code): name for name, code in available_event_id.items()}
    selected_event_names = np.asarray(
        [code_to_name[int(code)] for code in selected_events[:, 2]],
        dtype=str,
    )

    epochs = create_epochs(
        raw=subject_data.raw,
        events=selected_events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        preload=True,
        verbose=verbose,
    )

    X = epochs.get_data(copy=True)
    kept_run_ids = selected_event_run_ids[epochs.selection]
    kept_event_names = selected_event_names[epochs.selection].tolist()
    y = np.asarray([resolved_label_map[name] for name in kept_event_names], dtype=int)
    subject_ids = np.full(shape=len(y), fill_value=subject_data.subject_id, dtype=int)

    metadata = {
        "subjects": [int(subject_data.subject_id)],
        "source_run_ids": [int(run_id) for run_id in subject_data.run_ids],
        "run_segments": [asdict(segment) for segment in subject_data.run_segments],
        "file_paths": [segment.file_path for segment in subject_data.run_segments],
        "label_names_by_value": {
            str(label): event_name for event_name, label in resolved_label_map.items()
        },
    }

    return TrialDataset(
        X=X,
        y=y,
        subject_ids=subject_ids,
        run_ids=kept_run_ids.astype(int),
        trial_event_names=kept_event_names,
        selected_event_id=selected_event_id,
        available_event_id=available_event_id,
        label_map=resolved_label_map,
        channel_names=list(epochs.ch_names),
        sfreq=float(epochs.info["sfreq"]),
        epoch_tmin=float(tmin),
        epoch_tmax=float(tmax),
        trial_count=int(len(y)),
        metadata=metadata,
    )


def concatenate_trial_datasets(datasets: Sequence[TrialDataset]) -> TrialDataset:
    """Concatenate multiple per-subject trial datasets into one dataset."""

    if not datasets:
        raise ValueError("At least one TrialDataset is required for concatenation.")

    reference = datasets[0]
    for dataset in datasets[1:]:
        if dataset.channel_names != reference.channel_names:
            raise ValueError("Cannot concatenate datasets with different channel layouts.")
        if dataset.selected_event_id != reference.selected_event_id:
            raise ValueError("Cannot concatenate datasets with different selected event IDs.")
        if dataset.label_map != reference.label_map:
            raise ValueError("Cannot concatenate datasets with different label maps.")
        if dataset.sfreq != reference.sfreq:
            raise ValueError("Cannot concatenate datasets with different sampling frequencies.")
        if (
            dataset.epoch_tmin != reference.epoch_tmin
            or dataset.epoch_tmax != reference.epoch_tmax
        ):
            raise ValueError("Cannot concatenate datasets with different epoch windows.")

    combined_metadata = {
        "subjects": sorted(
            {
                int(subject_id)
                for dataset in datasets
                for subject_id in dataset.metadata.get("subjects", [])
            }
        ),
        "run_segments_by_subject": {
            str(dataset.metadata["subjects"][0]): dataset.metadata.get("run_segments", [])
            for dataset in datasets
        },
        "file_paths_by_subject": {
            str(dataset.metadata["subjects"][0]): dataset.metadata.get("file_paths", [])
            for dataset in datasets
        },
        "label_names_by_value": reference.metadata.get("label_names_by_value", {}),
    }

    return TrialDataset(
        X=np.concatenate([dataset.X for dataset in datasets], axis=0),
        y=np.concatenate([dataset.y for dataset in datasets], axis=0),
        subject_ids=np.concatenate([dataset.subject_ids for dataset in datasets], axis=0),
        run_ids=np.concatenate([dataset.run_ids for dataset in datasets], axis=0),
        trial_event_names=[
            event_name
            for dataset in datasets
            for event_name in dataset.trial_event_names
        ],
        selected_event_id=dict(reference.selected_event_id),
        available_event_id=dict(reference.available_event_id),
        label_map=dict(reference.label_map),
        channel_names=list(reference.channel_names),
        sfreq=float(reference.sfreq),
        epoch_tmin=float(reference.epoch_tmin),
        epoch_tmax=float(reference.epoch_tmax),
        trial_count=int(sum(dataset.trial_count for dataset in datasets)),
        metadata=combined_metadata,
    )


def build_multi_subject_trial_dataset(
    subject_data_map: Mapping[int, SubjectRawData],
    event_names: Sequence[str] | None = None,
    label_map: Mapping[str, int] | None = None,
    tmin: float = DEFAULT_EPOCH_TMIN,
    tmax: float = DEFAULT_EPOCH_TMAX,
    baseline: tuple[float, float] | None = None,
    picks: str | Sequence[str] | None = "eeg",
    verbose: str | bool = "ERROR",
) -> TrialDataset:
    """Build one packaged trial dataset across multiple subjects."""

    datasets = [
        build_trial_dataset(
            subject_data=subject_data_map[subject_id],
            event_names=event_names,
            label_map=label_map,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            verbose=verbose,
        )
        for subject_id in sorted(subject_data_map)
    ]
    return concatenate_trial_datasets(datasets)


def save_trial_dataset(
    dataset: TrialDataset,
    output_prefix: str | Path,
) -> tuple[Path, Path]:
    """Save trial arrays to NPZ and metadata to JSON."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        X=dataset.X,
        y=dataset.y,
        subject_ids=dataset.subject_ids,
        run_ids=dataset.run_ids,
        trial_event_names=np.asarray(dataset.trial_event_names, dtype=str),
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset.to_metadata_dict(), handle, indent=2)

    return npz_path, json_path


def load_trial_dataset(
    input_prefix: str | Path,
) -> TrialDataset:
    """Load a saved trial dataset from NPZ and JSON files."""

    prefix = Path(input_prefix)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Trial array file not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Trial metadata file not found: {json_path}")

    with np.load(npz_path, allow_pickle=False) as arrays:
        X = arrays["X"]
        y = arrays["y"]
        subject_ids = arrays["subject_ids"]
        run_ids = arrays["run_ids"]
        trial_event_names = arrays["trial_event_names"].astype(str).tolist()

    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    epoch_window = metadata.get("epoch_window", {})
    return TrialDataset(
        X=X,
        y=y,
        subject_ids=subject_ids,
        run_ids=run_ids,
        trial_event_names=trial_event_names,
        selected_event_id={
            key: int(value) for key, value in metadata.get("selected_event_id", {}).items()
        },
        available_event_id={
            key: int(value) for key, value in metadata.get("available_event_id", {}).items()
        },
        label_map={key: int(value) for key, value in metadata.get("label_map", {}).items()},
        channel_names=list(metadata.get("channel_names", [])),
        sfreq=float(metadata["sfreq"]),
        epoch_tmin=float(epoch_window["tmin"]),
        epoch_tmax=float(epoch_window["tmax"]),
        trial_count=int(metadata["trial_count"]),
        metadata={
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "selected_event_id",
                "available_event_id",
                "label_map",
                "channel_names",
                "sfreq",
                "epoch_window",
                "trial_count",
            }
        },
    )
