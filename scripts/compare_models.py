"""Compare CSP + SVM versus CSP + Logistic Regression on one dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_trial_dataset
from src.evaluation import (
    compute_classification_metrics,
    format_comparison_results,
    save_results_json,
)
from src.features import DEFAULT_CSP_COMPONENTS
from src.model import (
    AVAILABLE_MODEL_CONFIGS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    predict_with_classifier,
    split_trial_dataset,
    train_decoding_model,
)


def _parse_subject_list(value: str | None) -> list[int] | None:
    """Parse a comma-separated subject list."""

    if value is None:
        return None
    subjects = [item.strip() for item in value.split(",") if item.strip()]
    if not subjects:
        raise ValueError("Expected at least one subject ID in the subject list.")
    return [int(subject_id) for subject_id in subjects]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compare CSP + SVM and CSP + Logistic Regression on a preprocessed dataset.",
    )
    parser.add_argument(
        "--input-prefix",
        type=Path,
        required=True,
        help="Input preprocessed dataset prefix, without the .npz/.json suffix.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Held-out test fraction. Default: %(default)s",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for the train/test split and classifiers. Default: %(default)s",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_CSP_COMPONENTS,
        help="Number of CSP components. Default: %(default)s",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["random", "subject"],
        default="random",
        help="Split mode to use. Default: %(default)s",
    )
    parser.add_argument(
        "--train-subjects",
        type=str,
        default=None,
        help="Comma-separated train subject IDs for subject-wise splitting.",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        default=None,
        help="Comma-separated test subject IDs for subject-wise splitting.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for saving comparison metrics.",
    )
    return parser


def main() -> None:
    """Run a side-by-side comparison on one fixed train/test split."""

    parser = build_arg_parser()
    args = parser.parse_args()

    dataset = load_trial_dataset(args.input_prefix)
    train_subjects = _parse_subject_list(args.train_subjects)
    test_subjects = _parse_subject_list(args.test_subjects)
    split = split_trial_dataset(
        dataset=dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        split_mode=args.split_mode,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )
    labels = sorted({int(label) for label in dataset.label_map.values()})

    print("Split summary:")
    print(f"  split_mode: {split['split_mode']}")
    print(f"  train_samples: {len(split['y_train'])}")
    print(f"  test_samples: {len(split['y_test'])}")
    print(f"  train_subjects: {split['train_subjects']}")
    print(f"  test_subjects: {split['test_subjects']}")

    results = []
    for config_name in AVAILABLE_MODEL_CONFIGS:
        trained = train_decoding_model(
            config_name=config_name,
            X_train=split["X_train"],
            y_train=split["y_train"],
            X_test=split["X_test"],
            n_components=args.n_components,
            random_state=args.random_state,
        )
        predictions = predict_with_classifier(
            classifier=trained["classifier"],
            X_features=trained["X_test_features"],
        )
        result = compute_classification_metrics(
            y_true=split["y_test"],
            y_pred=predictions["y_pred"],
            labels=labels,
            config_name=config_name,
            train_size=len(split["y_train"]),
            test_size=len(split["y_test"]),
            n_channels=dataset.X.shape[1],
            n_components=args.n_components,
        )
        results.append(result)

    print("Model comparison summary:")
    print(format_comparison_results(results))

    if args.output_json is not None:
        payload = {
            "input_prefix": str(args.input_prefix),
            "subjects": sorted(int(subject_id) for subject_id in set(dataset.subject_ids.tolist())),
            "run_ids_present": sorted(int(run_id) for run_id in set(dataset.run_ids.tolist())),
            "split_mode": split["split_mode"],
            "train_subjects": split["train_subjects"],
            "test_subjects": split["test_subjects"],
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "n_components": int(args.n_components),
            "scaling": {"enabled": True, "scaler_class": "StandardScaler"},
            "results": results,
        }
        output_path = save_results_json(payload, args.output_json)
        print(f"Saved comparison JSON to: {output_path}")


if __name__ == "__main__":
    main()
