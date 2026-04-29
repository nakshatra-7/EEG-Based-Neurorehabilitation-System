"""CSP feature extraction helpers for EEG decoding."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP

DEFAULT_CSP_COMPONENTS = 4


def create_csp(n_components: int = DEFAULT_CSP_COMPONENTS) -> CSP:
    """Create a CSP transformer with explicit MVP-friendly defaults."""

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")

    return CSP(
        n_components=n_components,
        reg=None,
        log=True,
        norm_trace=False,
    )


def fit_csp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = DEFAULT_CSP_COMPONENTS,
) -> CSP:
    """Fit CSP using training data only."""

    csp = create_csp(n_components=n_components)
    csp.fit(X_train, y_train)
    return csp


def transform_with_csp(csp: CSP, X: np.ndarray) -> np.ndarray:
    """Transform EEG trials into CSP features."""

    features = csp.transform(X)
    return np.asarray(features, dtype=float)


def fit_transform_csp_train_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = DEFAULT_CSP_COMPONENTS,
) -> tuple[CSP, np.ndarray, np.ndarray]:
    """Fit CSP on training data and transform train/test splits."""

    csp = fit_csp(
        X_train=X_train,
        y_train=y_train,
        n_components=n_components,
    )
    X_train_features = transform_with_csp(csp, X_train)
    X_test_features = transform_with_csp(csp, X_test)
    return csp, X_train_features, X_test_features
