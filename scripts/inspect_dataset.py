"""Inspect EEGBCI annotations and events before defining class semantics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATASET_STORAGE_PATH, DEFAULT_RUNS, DEFAULT_SUBJECTS
from src.data_loader import (
    extract_events_from_annotations,
    load_subject_raws,
    summarize_annotations,
    summarize_events,
)


def _parse_int_list(value: str | None, fallback: list[int]) -> list[int]:
    """Parse a comma-separated integer list."""

    if value is None:
        return list(fallback)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated integer value.")
    return [int(item) for item in items]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Inspect EEGBCI annotations/events for selected subjects and runs.",
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
        "--data-path",
        type=Path,
        default=DATASET_STORAGE_PATH,
        help=f"Directory used by MNE to store EEGBCI EDF files. Default: {DATASET_STORAGE_PATH}",
    )
    return parser


def main() -> None:
    """Load EEGBCI recordings and print annotation/event summaries."""

    parser = build_arg_parser()
    args = parser.parse_args()

    subject_ids = _parse_int_list(args.subjects, DEFAULT_SUBJECTS)
    run_ids = _parse_int_list(args.runs, DEFAULT_RUNS)

    subject_data_map = load_subject_raws(
        subject_ids=subject_ids,
        run_ids=run_ids,
        data_path=args.data_path,
        preload=True,
    )

    for subject_id in sorted(subject_data_map):
        subject_data = subject_data_map[subject_id]
        raw = subject_data.raw
        events, event_id = extract_events_from_annotations(raw)
        annotation_summary = summarize_annotations(raw)
        event_summary = summarize_events(events, event_id)

        print(f"Subject: {subject_id}")
        print(f"Runs: {subject_data.run_ids}")
        print(
            "Raw info: "
            f"n_channels={len(raw.ch_names)}, "
            f"sfreq={raw.info['sfreq']}, "
            f"duration_s={round(raw.times[-1], 3)}"
        )
        print("Run segments:")
        for segment in subject_data.run_segments:
            print(
                f"  run={segment.run_id} "
                f"samples=[{segment.start_sample}, {segment.stop_sample}] "
                f"path={segment.file_path}"
            )
        print("Annotation summary:")
        for row in annotation_summary:
            print(
                f"  {row['description']}: count={row['count']}, "
                f"mean_duration_s={row['mean_duration_s']}, "
                f"total_duration_s={row['total_duration_s']}"
            )
        print("Event ID mapping:")
        for name, code in sorted(event_id.items(), key=lambda item: item[1]):
            print(f"  {name} -> {code}")
        print("Event counts:")
        for row in event_summary:
            print(
                f"  {row['event_name']} (code={row['event_code']}): count={row['count']}"
            )
        print(
            "Verification note: confirm which annotations map to left/right motor imagery "
            "for the selected runs before defining Phase 2 labels."
        )
        print()


if __name__ == "__main__":
    main()

