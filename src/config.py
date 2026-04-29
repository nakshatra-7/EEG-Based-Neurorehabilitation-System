"""Central configuration for dataset ingestion defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_STORAGE_PATH = DATA_DIR / "eegbci"

DEFAULT_SUBJECTS = [1]

# EEGBCI motor imagery runs commonly used for left-vs-right imagery.
# Final semantic mapping must still be verified by inspecting annotations/events.
DEFAULT_RUNS = [4, 8, 12]

DEFAULT_EPOCH_TMIN = 0.0
DEFAULT_EPOCH_TMAX = 4.0
DEFAULT_RANDOM_SEED = 42

# Placeholder aliases only. Do not treat these as verified clinical labels until
# the chosen runs are inspected with the event inspection helpers.
PLACEHOLDER_EVENT_ALIASES = {
    "T0": "rest_or_baseline",
    "T1": "verify_after_inspection_class_1",
    "T2": "verify_after_inspection_class_2",
}

# Optional explicit event selection for extraction. Keep unset until inspection
# confirms which annotations should be included for the current experiment.
DEFAULT_EVENT_SELECTION = None

