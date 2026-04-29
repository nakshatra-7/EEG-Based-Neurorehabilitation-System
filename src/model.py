"""Model-building, splitting, prediction, and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_loader import TrialDataset
from src.features import (
    DEFAULT_CSP_COMPONENTS,
    fit_transform_csp_train_test,
    transform_with_csp,
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_LOGREG_MAX_ITER = 1000

CONFIG_CSP_SVM = "csp_svm"
CONFIG_CSP_LOGREG = "csp_logreg"
AVAILABLE_MODEL_CONFIGS = (CONFIG_CSP_SVM, CONFIG_CSP_LOGREG)


def build_svm_classifier(random_state: int = DEFAULT_RANDOM_STATE) -> SVC:
    """Build the baseline linear SVM classifier."""

    return SVC(
        kernel="linear",
        probability=True,
        random_state=random_state,
    )


def build_logistic_regression_classifier(
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_LOGREG_MAX_ITER,
) -> LogisticRegression:
    """Build the baseline logistic regression classifier."""

    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )


def build_classifier(
    config_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_LOGREG_MAX_ITER,
) -> Any:
    """Create a classifier instance from a supported configuration name."""

    if config_name == CONFIG_CSP_SVM:
        return build_svm_classifier(random_state=random_state)
    if config_name == CONFIG_CSP_LOGREG:
        return build_logistic_regression_classifier(
            random_state=random_state,
            max_iter=max_iter,
        )
    raise ValueError(
        f"Unsupported config '{config_name}'. Choose from {list(AVAILABLE_MODEL_CONFIGS)}."
    )


def split_trial_dataset(
    dataset: TrialDataset,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    split_mode: str = "random",
    train_subjects: list[int] | None = None,
    test_subjects: list[int] | None = None,
) -> dict[str, Any]:
    """Create either a random stratified split or a subject-wise split."""

    n_trials = int(dataset.X.shape[0])
    if dataset.trial_count != n_trials:
        raise ValueError(
            f"trial_count ({dataset.trial_count}) does not match X trials ({n_trials})."
        )

    if split_mode == "random":
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

        unique_labels = np.unique(dataset.y)
        if unique_labels.size < 2:
            raise ValueError(
                f"Need at least two classes for stratified splitting, got labels {unique_labels.tolist()}."
            )

        indices = np.arange(n_trials)
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=dataset.y,
        )
    elif split_mode == "subject":
        train_indices, test_indices = subject_wise_split_indices(
            subject_ids=dataset.subject_ids,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
        )
    else:
        raise ValueError(
            f"Unsupported split_mode '{split_mode}'. Choose from ['random', 'subject']."
        )

    return {
        "split_mode": split_mode,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train": dataset.X[train_indices],
        "X_test": dataset.X[test_indices],
        "y_train": dataset.y[train_indices],
        "y_test": dataset.y[test_indices],
        "subject_ids_train": dataset.subject_ids[train_indices],
        "subject_ids_test": dataset.subject_ids[test_indices],
        "run_ids_train": dataset.run_ids[train_indices],
        "run_ids_test": dataset.run_ids[test_indices],
        "trial_event_names_train": [dataset.trial_event_names[index] for index in train_indices],
        "trial_event_names_test": [dataset.trial_event_names[index] for index in test_indices],
        "train_subjects": sorted(int(subject_id) for subject_id in np.unique(dataset.subject_ids[train_indices])),
        "test_subjects": sorted(int(subject_id) for subject_id in np.unique(dataset.subject_ids[test_indices])),
    }


def subject_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_subjects: list[int],
    test_subjects: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays by subject IDs without random mixing between train and test."""

    train_indices, test_indices = subject_wise_split_indices(
        subject_ids=subject_ids,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
    )
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def subject_wise_split_indices(
    subject_ids: np.ndarray,
    train_subjects: list[int] | None,
    test_subjects: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trial indices for a subject-wise train/test split."""

    if train_subjects is None or not train_subjects:
        raise ValueError("train_subjects must be provided for split_mode='subject'.")
    if test_subjects is None or not test_subjects:
        raise ValueError("test_subjects must be provided for split_mode='subject'.")

    normalized_train = [int(subject_id) for subject_id in train_subjects]
    normalized_test = [int(subject_id) for subject_id in test_subjects]

    overlap = sorted(set(normalized_train).intersection(normalized_test))
    if overlap:
        raise ValueError(
            f"Train and test subject lists must not overlap. Overlap found: {overlap}"
        )

    available_subjects = {int(subject_id) for subject_id in np.unique(subject_ids)}
    missing_train = sorted(set(normalized_train) - available_subjects)
    missing_test = sorted(set(normalized_test) - available_subjects)
    if missing_train:
        raise ValueError(f"Train subjects not present in dataset: {missing_train}")
    if missing_test:
        raise ValueError(f"Test subjects not present in dataset: {missing_test}")

    train_mask = np.isin(subject_ids, normalized_train)
    test_mask = np.isin(subject_ids, normalized_test)
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    if train_indices.size == 0:
        raise ValueError("Subject-wise split produced zero training samples.")
    if test_indices.size == 0:
        raise ValueError("Subject-wise split produced zero test samples.")

    return train_indices, test_indices


def fit_classifier(classifier: Any, X_train_features: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit a classifier on training features."""

    classifier.fit(X_train_features, y_train)
    return classifier


def create_standard_scaler() -> StandardScaler:
    """Create the feature scaler used after CSP."""

    return StandardScaler()


def fit_transform_scaler_train_test(
    X_train_features: np.ndarray,
    X_test_features: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    """Fit StandardScaler on train features only and transform both splits."""

    scaler = create_standard_scaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    return scaler, np.asarray(X_train_scaled, dtype=float), np.asarray(X_test_scaled, dtype=float)


def transform_features_with_scaler(
    scaler: StandardScaler,
    X_features: np.ndarray,
) -> np.ndarray:
    """Apply a fitted StandardScaler to CSP features."""

    return np.asarray(scaler.transform(X_features), dtype=float)


def predict_with_classifier(classifier: Any, X_features: np.ndarray) -> dict[str, Any]:
    """Generate labels and probability-compatible confidence outputs."""

    y_pred = classifier.predict(X_features)

    y_proba = None
    positive_class_probability = None
    confidence = None
    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(X_features)
        class_labels = np.asarray(getattr(classifier, "classes_", []))
        if class_labels.size > 0 and np.any(class_labels == 1):
            positive_class_index = int(np.where(class_labels == 1)[0][0])
        else:
            positive_class_index = int(y_proba.shape[1] - 1)
        positive_class_probability = y_proba[:, positive_class_index]
        confidence = np.max(y_proba, axis=1)
    elif hasattr(classifier, "decision_function"):
        decision_scores = classifier.decision_function(X_features)
        positive_class_probability = np.asarray(decision_scores, dtype=float)
        confidence = np.abs(positive_class_probability)

    return {
        "y_pred": np.asarray(y_pred, dtype=int),
        "y_proba": None if y_proba is None else np.asarray(y_proba, dtype=float),
        "positive_class_probability": (
            None
            if positive_class_probability is None
            else np.asarray(positive_class_probability, dtype=float)
        ),
        "confidence": None if confidence is None else np.asarray(confidence, dtype=float),
    }


def predict_with_decoding_model(
    csp: Any,
    scaler: StandardScaler | None,
    classifier: Any,
    X: np.ndarray,
) -> dict[str, Any]:
    """Run inference with fitted CSP, StandardScaler, and classifier."""

    if scaler is None:
        raise ValueError("A fitted StandardScaler is required for this decoding model.")

    X_features = transform_with_csp(csp, X)
    X_scaled = transform_features_with_scaler(scaler, X_features)
    return predict_with_classifier(classifier=classifier, X_features=X_scaled)


def train_decoding_model(
    config_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = DEFAULT_CSP_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_iter: int = DEFAULT_LOGREG_MAX_ITER,
) -> dict[str, Any]:
    """Fit CSP and StandardScaler on training data, then train one classifier."""

    csp, X_train_csp, X_test_csp = fit_transform_csp_train_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        n_components=n_components,
    )
    scaler, X_train_features, X_test_features = fit_transform_scaler_train_test(
        X_train_features=X_train_csp,
        X_test_features=X_test_csp,
    )
    classifier = build_classifier(
        config_name=config_name,
        random_state=random_state,
        max_iter=max_iter,
    )
    fitted_classifier = fit_classifier(
        classifier=classifier,
        X_train_features=X_train_features,
        y_train=y_train,
    )
    return {
        "csp": csp,
        "scaler": scaler,
        "classifier": fitted_classifier,
        "X_train_features": X_train_features,
        "X_test_features": X_test_features,
    }


def save_model_artifacts(
    artifact_prefix: str | Path,
    csp: Any,
    scaler: StandardScaler,
    classifier: Any,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Save fitted objects to joblib and metadata/metrics to JSON."""

    prefix = Path(artifact_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    joblib_path = prefix.with_suffix(".joblib")
    json_path = prefix.with_suffix(".json")

    joblib.dump({"csp": csp, "scaler": scaler, "classifier": classifier}, joblib_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return joblib_path, json_path


def load_model_artifacts(artifact_prefix: str | Path) -> dict[str, Any]:
    """Load saved fitted objects and metadata."""

    prefix = Path(artifact_prefix)
    joblib_path = prefix.with_suffix(".joblib")
    json_path = prefix.with_suffix(".json")

    if not joblib_path.exists():
        raise FileNotFoundError(f"Model artifact file not found: {joblib_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Model metadata file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    payload = joblib.load(joblib_path)
    if "scaler" not in payload:
        payload["scaler"] = None
    payload["metadata"] = metadata
    return payload
