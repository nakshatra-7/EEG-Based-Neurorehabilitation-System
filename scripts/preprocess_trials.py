"""Apply simple preprocessing to saved EEG trial datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_trial_dataset, save_trial_dataset
from src.preprocessing import (
    DEFAULT_H_FREQ,
    DEFAULT_L_FREQ,
    preprocess_trial_dataset,
    summarize_preprocessed_dataset,
)


def _parse_channels(value: str | None) -> list[str] | None:
    """Parse a comma-separated channel list."""

    if value is None:
        return None
    channels = [channel_name.strip() for channel_name in value.split(",") if channel_name.strip()]
    if not channels:
        raise ValueError(
            "Channel selection was provided but no valid channel names were parsed."
        )
    return channels


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Band-pass filter a saved EEG trial dataset and optionally subset channels.",
    )
    parser.add_argument(
        "--input-prefix",
        type=Path,
        required=True,
        help="Input dataset prefix from Phase 1, without the .npz/.json suffix.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output dataset prefix for the preprocessed result.",
    )
    parser.add_argument(
        "--l-freq",
        type=float,
        default=DEFAULT_L_FREQ,
        help="Band-pass low cutoff in Hz. Default: %(default)s",
    )
    parser.add_argument(
        "--h-freq",
        type=float,
        default=DEFAULT_H_FREQ,
        help="Band-pass high cutoff in Hz. Default: %(default)s",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Optional comma-separated channel subset, for example 'C3,Cz,C4'.",
    )
    return parser


def main() -> None:
    """Load, preprocess, and save a trial dataset."""

    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = load_trial_dataset(args.input_prefix)
    selected_channels = _parse_channels(args.channels)
    preprocessed = preprocess_trial_dataset(
        dataset=dataset,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        selected_channels=selected_channels,
    )
    npz_path, json_path = save_trial_dataset(preprocessed, args.output_prefix)
    summary = summarize_preprocessed_dataset(preprocessed)

    print("Preprocessed dataset summary:")
    print(f"  X shape: {summary['X_shape']}")
    print(f"  y shape: {summary['y_shape']}")
    print(f"  trial_count: {summary['trial_count']}")
    print(f"  sfreq: {summary['sfreq']}")
    print(f"  epoch_window: {summary['epoch_window']}")
    print(f"  selected_event_id: {summary['selected_event_id']}")
    print(f"  label_map: {summary['label_map']}")
    print(f"  subjects: {summary['subjects']}")
    print(f"  run_ids present: {summary['run_ids']}")
    print(f"  label_counts: {summary['label_counts']}")
    print(f"  preprocessing: {summary['preprocessing']}")
    print(f"Saved arrays to: {npz_path}")
    print(f"Saved metadata to: {json_path}")


if __name__ == "__main__":
    main()
