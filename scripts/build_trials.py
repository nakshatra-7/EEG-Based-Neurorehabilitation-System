"""Build EEGBCI trial arrays from selected subjects, runs, and event labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    DATASET_STORAGE_PATH,
    DEFAULT_EPOCH_TMAX,
    DEFAULT_EPOCH_TMIN,
    DEFAULT_RUNS,
    DEFAULT_SUBJECTS,
)
from src.data_loader import (
    build_multi_subject_trial_dataset,
    load_subject_raws,
    save_trial_dataset,
)


def _parse_int_list(value: str | None, fallback: list[int]) -> list[int]:
    """Parse a comma-separated integer list."""

    if value is None:
        return list(fallback)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated integer value.")
    return [int(item) for item in items]


def _parse_str_list(value: str | None) -> list[str] | None:
    """Parse a comma-separated string list."""

    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_label_map(value: str | None) -> dict[str, int] | None:
    """Parse mapping strings such as 'T1:0,T2:1'."""

    if value is None:
        return None

    mapping: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid label map format. Use comma-separated pairs like 'T1:0,T2:1'."
            )
        event_name, label = item.split(":", maxsplit=1)
        mapping[event_name.strip()] = int(label.strip())

    if not mapping:
        raise ValueError("Label map string was provided but no valid pairs were parsed.")
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Extract EEGBCI trials into NumPy arrays and metadata.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(str(subject) for subject in DEFAULT_SUBJECTS),
        help="Comma-separated subject IDs. Default: %(default)s",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=",".join(str(run_id) for run_id in DEFAULT_RUNS),
        help="Comma-separated run IDs. Default: %(default)s",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Optional comma-separated annotation labels to epoch, for example 'T1,T2'.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Optional explicit label mapping, for example 'T1:0,T2:1'.",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=DEFAULT_EPOCH_TMIN,
        help="Epoch start time in seconds relative to the event. Default: %(default)s",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_EPOCH_TMAX,
        help="Epoch end time in seconds relative to the event. Default: %(default)s",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATASET_STORAGE_PATH,
        help=f"Directory used by MNE to store EEGBCI EDF files. Default: {DATASET_STORAGE_PATH}",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Optional output prefix. If provided, saves '<prefix>.npz' and '<prefix>.json'.",
    )
    return parser


def main() -> None:
    """Extract trials and optionally persist them to disk."""

    parser = build_arg_parser()
    args = parser.parse_args()

    subject_ids = _parse_int_list(args.subjects, DEFAULT_SUBJECTS)
    run_ids = _parse_int_list(args.runs, DEFAULT_RUNS)
    event_names = _parse_str_list(args.events)
    label_map = _parse_label_map(args.label_map)

    subject_data_map = load_subject_raws(
        subject_ids=subject_ids,
        run_ids=run_ids,
        data_path=args.data_path,
        preload=True,
    )
    dataset = build_multi_subject_trial_dataset(
        subject_data_map=subject_data_map,
        event_names=event_names,
        label_map=label_map,
        tmin=args.tmin,
        tmax=args.tmax,
    )

    print("Trial dataset summary:")
    print(f"  X shape: {dataset.X.shape}")
    print(f"  y shape: {dataset.y.shape}")
    print(f"  trial_count: {dataset.trial_count}")
    print(f"  selected_event_id: {dataset.selected_event_id}")
    print(f"  label_map: {dataset.label_map}")
    print(f"  subjects: {sorted(set(dataset.subject_ids.tolist()))}")
    print(f"  run_ids present: {sorted(set(dataset.run_ids.tolist()))}")
    print(
        "  label_counts: "
        f"{ {int(label): int((dataset.y == label).sum()) for label in sorted(set(dataset.y.tolist()))} }"
    )
    print(
        "Verification note: if you intend to use left-vs-right labels in Phase 2, "
        "confirm the annotation semantics with scripts.inspect_dataset before relying on T1/T2."
    )

    if args.output_prefix is not None:
        npz_path, json_path = save_trial_dataset(dataset, args.output_prefix)
        print(f"Saved arrays to: {npz_path}")
        print(f"Saved metadata to: {json_path}")


if __name__ == "__main__":
    main()
