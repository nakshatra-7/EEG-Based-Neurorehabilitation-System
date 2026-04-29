"""Evaluation helpers for baseline EEG decoding models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    config_name: str,
    train_size: int,
    test_size: int,
    n_channels: int,
    n_components: int,
) -> dict[str, Any]:
    """Compute serializable binary-classification metrics."""

    ordered_labels = [int(label) for label in labels]
    matrix = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    return {
        "config_name": config_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": ordered_labels,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "n_channels": int(n_channels),
        "n_csp_components": int(n_components),
    }


def format_result_summary(result: dict[str, Any]) -> str:
    """Format one evaluation result for CLI output."""

    lines = [
        f"{result['config_name']}:",
        f"  accuracy={result['accuracy']:.4f}",
        f"  precision={result['precision']:.4f}",
        f"  recall={result['recall']:.4f}",
        f"  f1_score={result['f1_score']:.4f}",
        f"  confusion_matrix={result['confusion_matrix']}",
        f"  train_size={result['train_size']}, test_size={result['test_size']}",
        f"  n_channels={result['n_channels']}, n_csp_components={result['n_csp_components']}",
    ]
    return "\n".join(lines)


def format_comparison_results(results: Sequence[dict[str, Any]]) -> str:
    """Format multiple evaluation results for comparison output."""

    return "\n\n".join(format_result_summary(result) for result in results)


def save_results_json(results: dict[str, Any], output_path: str | Path) -> Path:
    """Save evaluation results to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return path

