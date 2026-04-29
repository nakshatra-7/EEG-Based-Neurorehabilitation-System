"""Simulate a sequential EEG neurorehabilitation session from saved trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_loader import load_trial_dataset
from src.inference import load_inference_artifacts
from src.session import (
    build_session_summary,
    initialize_session,
    run_session,
    subset_trial_dataset,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Simulate a trial-by-trial EEG session using saved model artifacts.",
    )
    parser.add_argument(
        "--input-prefix",
        type=Path,
        required=True,
        help="Input preprocessed dataset prefix, without the .npz/.json suffix.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=Path,
        required=True,
        help="Saved model artifact prefix, without the .joblib/.json suffix.",
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=None,
        help="Optional subject ID to simulate, for example 5.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional maximum number of trials to simulate in order.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step", "full"],
        default="full",
        help="Simulation mode. Default: %(default)s",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for saving session history and summary.",
    )
    return parser


def _print_trial_result(result: dict[str, object]) -> None:
    """Print one readable per-trial session line."""

    confidence = result["confidence"]
    confidence_text = "None" if confidence is None else f"{confidence:.4f}"
    print(
        f"trial={result['trial_index']} "
        f"subject={result['subject_id']} "
        f"run={result['run_id']} "
        f"true={result['true_label_name']}({result['true_label']}) "
        f"pred={result['predicted_label_name']}({result['predicted_label']}) "
        f"confidence={confidence_text} "
        f"correct={result['correct']}"
    )


def main() -> None:
    """Run the session simulation CLI."""

    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = load_trial_dataset(args.input_prefix)
    session_dataset = subset_trial_dataset(
        dataset=dataset,
        subject_id=args.subject_id,
    )
    artifact_payload = load_inference_artifacts(str(args.artifact_prefix))
    state = initialize_session(session_dataset)

    print("Session setup:")
    print(f"  total_trials: {state.total_trials}")
    print(f"  subjects_present: {sorted(set(state.subject_ids.tolist()))}")
    print(f"  run_ids_present: {sorted(set(state.run_ids.tolist()))}")

    if args.mode == "step":
        steps_to_run = 1 if args.max_trials is None else args.max_trials
        results = run_session(state, artifact_payload, max_steps=steps_to_run)
    else:
        results = run_session(state, artifact_payload, max_steps=args.max_trials)

    print("Per-trial results:")
    for result in results:
        _print_trial_result(result)

    summary = build_session_summary(state)
    print("Final session summary:")
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_prefix": str(args.input_prefix),
            "artifact_prefix": str(args.artifact_prefix),
            "subject_id": args.subject_id,
            "mode": args.mode,
            "history": state.history,
            "summary": summary,
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved session JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
