"""Simulate a sequential EEG session with adaptive rehab controller outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.adaptive_controller import (
    build_adaptive_session_summary,
    initialize_controller_state,
    update_adaptive_controller,
)
from src.data_loader import load_trial_dataset
from src.inference import load_inference_artifacts
from src.session import (
    build_session_summary,
    has_remaining_trials,
    initialize_session,
    step_session,
    subset_trial_dataset,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Simulate an adaptive EEG neurorehabilitation session.",
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
        help="Optional JSON path for saving adaptive session history and summary.",
    )
    return parser


def _print_adaptive_trial(trial_result: dict[str, object]) -> None:
    """Print per-trial model output plus adaptive controller state."""

    confidence = trial_result["confidence"]
    confidence_text = "None" if confidence is None else f"{confidence:.4f}"
    controller = trial_result["controller"]

    print(
        f"trial={trial_result['trial_index']} "
        f"subject={trial_result['subject_id']} "
        f"run={trial_result['run_id']} "
        f"true={trial_result['true_label_name']}({trial_result['true_label']}) "
        f"pred={trial_result['predicted_label_name']}({trial_result['predicted_label']}) "
        f"confidence={confidence_text} "
        f"correct={trial_result['correct']} "
        f"difficulty={controller['difficulty_level']} "
        f"bias={controller['practice_bias']} "
        f"ratio={controller['schedule_ratio']} "
        f"reason={controller['reason_code']}"
    )
    print(f"  feedback={controller['feedback_message']}")


def main() -> None:
    """Run the adaptive session simulation CLI."""

    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = load_trial_dataset(args.input_prefix)
    session_dataset = subset_trial_dataset(
        dataset=dataset,
        subject_id=args.subject_id,
    )
    artifact_payload = load_inference_artifacts(str(args.artifact_prefix))
    session_state = initialize_session(session_dataset)
    controller_state = initialize_controller_state(initial_difficulty="easy")

    print("Adaptive session setup:")
    print(f"  total_trials: {session_state.total_trials}")
    print(f"  subjects_present: {sorted(set(session_state.subject_ids.tolist()))}")
    print(f"  run_ids_present: {sorted(set(session_state.run_ids.tolist()))}")

    max_steps = 1 if args.mode == "step" and args.max_trials is None else args.max_trials
    processed_steps = 0
    while has_remaining_trials(session_state):
        if max_steps is not None and processed_steps >= max_steps:
            break
        trial_result = step_session(session_state, artifact_payload)
        controller_output = update_adaptive_controller(
            history=session_state.history,
            state=controller_state,
        )
        trial_result["controller"] = controller_output
        session_state.history[-1]["controller"] = controller_output
        _print_adaptive_trial(trial_result)
        processed_steps += 1

    session_summary = build_session_summary(session_state)
    adaptive_summary = build_adaptive_session_summary(
        session_summary=session_summary,
        controller_state=controller_state,
    )

    print("Final adaptive session summary:")
    print(json.dumps(adaptive_summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_prefix": str(args.input_prefix),
            "artifact_prefix": str(args.artifact_prefix),
            "subject_id": args.subject_id,
            "mode": args.mode,
            "history": session_state.history,
            "session_summary": session_summary,
            "adaptive_summary": adaptive_summary,
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved adaptive session JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
