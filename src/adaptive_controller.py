"""Rule-based adaptive rehabilitation controller for sequential sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ROLLING_WINDOW_SIZE = 10
LEFT_LABEL = 0
RIGHT_LABEL = 1
DIFFICULTY_ORDER = ("easy", "medium", "hard")
DIFFICULTY_SETTINGS = {
    "easy": {"cursor_speed": 3, "target_size": 80, "timeout_sec": 5.0},
    "medium": {"cursor_speed": 5, "target_size": 60, "timeout_sec": 4.0},
    "hard": {"cursor_speed": 7, "target_size": 40, "timeout_sec": 3.0},
}


@dataclass
class AdaptiveControllerState:
    """Mutable controller state across a sequential rehab session."""

    difficulty_level: str = "easy"
    practice_bias: str = "balanced"
    schedule_ratio: str = "1:1"
    last_difficulty_change_trial_index: int = -3
    difficulty_change_count: int = 0
    left_bias_count: int = 0
    right_bias_count: int = 0
    last_floor_hold_trial_index: int = -5
    last_floor_hold_error_streak: int = 0


def initialize_controller_state(
    initial_difficulty: str = "easy",
) -> AdaptiveControllerState:
    """Initialize adaptive controller state."""

    if initial_difficulty not in DIFFICULTY_SETTINGS:
        raise ValueError(
            f"Unsupported initial difficulty '{initial_difficulty}'. "
            f"Choose from {list(DIFFICULTY_SETTINGS)}."
        )
    return AdaptiveControllerState(difficulty_level=initial_difficulty)


def _mean(values: list[float]) -> float:
    """Return a safe arithmetic mean."""

    return sum(values) / len(values) if values else 0.0


def _side_accuracy(window: list[dict[str, Any]], label: int) -> tuple[float, int]:
    """Compute side-specific rolling accuracy and sample count."""

    side_trials = [item for item in window if int(item["true_label"]) == int(label)]
    if not side_trials:
        return 0.0, 0
    accuracy = _mean([1.0 if item["correct"] else 0.0 for item in side_trials])
    return float(accuracy), len(side_trials)


def _error_streak(history: list[dict[str, Any]]) -> int:
    """Count consecutive errors from the end of the session history."""

    streak = 0
    for item in reversed(history):
        if item["correct"]:
            break
        streak += 1
    return streak


def compute_rolling_metrics(
    history: list[dict[str, Any]],
    window_size: int = ROLLING_WINDOW_SIZE,
) -> dict[str, Any]:
    """Compute rolling performance metrics from sequential session history."""

    if not history:
        return {
            "processed_trials": 0,
            "rolling_accuracy": 0.0,
            "rolling_mean_confidence": 0.0,
            "last_six_accuracy": 0.0,
            "last_six_mean_confidence": 0.0,
            "left_accuracy": 0.0,
            "right_accuracy": 0.0,
            "side_accuracy_gap": 0.0,
            "left_sample_count": 0,
            "right_sample_count": 0,
            "error_streak": 0,
            "previous_five_mean_confidence": None,
            "last_five_mean_confidence": None,
            "confidence_trend_delta": None,
            "confidence_decline_detected": False,
            "side_imbalance_detected": False,
            "weaker_side": "balanced",
        }

    window = history[-window_size:]
    rolling_accuracy = _mean([1.0 if item["correct"] else 0.0 for item in window])
    confidence_values = [
        float(item["confidence"])
        for item in window
        if item.get("confidence") is not None
    ]
    rolling_mean_confidence = _mean(confidence_values)
    last_six_window = history[-6:]
    last_six_accuracy = _mean([1.0 if item["correct"] else 0.0 for item in last_six_window])
    last_six_confidences = [
        float(item["confidence"])
        for item in last_six_window
        if item.get("confidence") is not None
    ]
    last_six_mean_confidence = _mean(last_six_confidences)

    left_accuracy, left_sample_count = _side_accuracy(window, LEFT_LABEL)
    right_accuracy, right_sample_count = _side_accuracy(window, RIGHT_LABEL)
    side_accuracy_gap = abs(left_accuracy - right_accuracy)
    error_streak = _error_streak(history)

    previous_five_mean_confidence = None
    last_five_mean_confidence = None
    confidence_trend_delta = None
    confidence_decline_detected = False
    if len(window) >= 10:
        previous_five = [
            float(item["confidence"])
            for item in window[-10:-5]
            if item.get("confidence") is not None
        ]
        last_five = [
            float(item["confidence"])
            for item in window[-5:]
            if item.get("confidence") is not None
        ]
        if previous_five and last_five:
            previous_five_mean_confidence = _mean(previous_five)
            last_five_mean_confidence = _mean(last_five)
            confidence_trend_delta = last_five_mean_confidence - previous_five_mean_confidence
            confidence_decline_detected = confidence_trend_delta <= -0.08

    side_imbalance_detected = (
        left_sample_count >= 4
        and right_sample_count >= 4
        and side_accuracy_gap >= 0.30
    )
    weaker_side = "balanced"
    if side_imbalance_detected:
        weaker_side = "left" if left_accuracy < right_accuracy else "right"

    return {
        "processed_trials": len(history),
        "rolling_accuracy": float(rolling_accuracy),
        "rolling_mean_confidence": float(rolling_mean_confidence),
        "last_six_accuracy": float(last_six_accuracy),
        "last_six_mean_confidence": float(last_six_mean_confidence),
        "left_accuracy": float(left_accuracy),
        "right_accuracy": float(right_accuracy),
        "side_accuracy_gap": float(side_accuracy_gap),
        "left_sample_count": int(left_sample_count),
        "right_sample_count": int(right_sample_count),
        "error_streak": int(error_streak),
        "previous_five_mean_confidence": previous_five_mean_confidence,
        "last_five_mean_confidence": last_five_mean_confidence,
        "confidence_trend_delta": confidence_trend_delta,
        "confidence_decline_detected": bool(confidence_decline_detected),
        "side_imbalance_detected": bool(side_imbalance_detected),
        "weaker_side": weaker_side,
    }


def _change_difficulty(current_level: str, direction: str) -> str:
    """Move one difficulty step up or down with boundary protection."""

    current_index = DIFFICULTY_ORDER.index(current_level)
    if direction == "increase":
        return DIFFICULTY_ORDER[min(current_index + 1, len(DIFFICULTY_ORDER) - 1)]
    if direction == "decrease":
        return DIFFICULTY_ORDER[max(current_index - 1, 0)]
    raise ValueError(f"Unsupported difficulty direction '{direction}'.")


def _is_cooldown_active(
    state: AdaptiveControllerState,
    current_trial_index: int,
) -> bool:
    """Return whether difficulty changes are currently on cooldown."""

    return (current_trial_index - state.last_difficulty_change_trial_index) < 3


def _should_emit_floor_hold(
    state: AdaptiveControllerState,
    current_trial_index: int,
    error_streak: int,
) -> bool:
    """Throttle repeated floor-hold messages while difficulty is already easy."""

    return (
        (current_trial_index - state.last_floor_hold_trial_index) >= 5
        or error_streak > state.last_floor_hold_error_streak
    )


def update_adaptive_controller(
    history: list[dict[str, Any]],
    state: AdaptiveControllerState,
) -> dict[str, Any]:
    """Apply rule-based adaptation after the latest processed trial."""

    if not history:
        raise ValueError("Adaptive controller requires at least one processed trial.")

    metrics = compute_rolling_metrics(history)
    current_trial_index = int(history[-1]["trial_index"])
    current_level = state.difficulty_level
    cooldown_active = _is_cooldown_active(state, current_trial_index)

    practice_bias = "balanced"
    schedule_ratio = "1:1"
    side_feedback_message = "Practice remains balanced at 1:1."
    if metrics["side_imbalance_detected"]:
        practice_bias = metrics["weaker_side"]
        schedule_ratio = "2:1"
        if practice_bias == "left":
            side_feedback_message = "Biasing practice toward the left side with a 2:1 ratio."
        else:
            side_feedback_message = "Biasing practice toward the right side with a 2:1 ratio."
    elif (
        state.practice_bias != "balanced"
        and metrics["side_accuracy_gap"] < 0.10
    ):
        side_feedback_message = "Side performance recovered; returning to balanced practice at 1:1."

    reason_code = "maintain_difficulty"
    feedback_parts = []
    next_level = current_level
    requested_change: str | None = None

    increase_allowed = current_trial_index >= 8

    increase_path_a = (
        increase_allowed
        and metrics["processed_trials"] >= 6
        and metrics["rolling_accuracy"] >= 0.62
        and metrics["rolling_mean_confidence"] >= 0.62
        and not metrics["side_imbalance_detected"]
    )
    increase_path_b = (
        increase_allowed
        and metrics["processed_trials"] >= 6
        and metrics["last_six_accuracy"] >= 0.67
        and metrics["last_six_mean_confidence"] >= 0.60
    )

    if metrics["rolling_accuracy"] < 0.45 or metrics["error_streak"] >= 4:
        requested_change = "decrease"
        reason_code = "decrease_difficulty"
        feedback_parts.append("Recent performance dropped; reducing task difficulty.")
    elif metrics["confidence_decline_detected"] and current_level == "hard":
        requested_change = "decrease"
        reason_code = "confidence_decline_reduce"
        feedback_parts.append("Confidence dropped on hard difficulty; reducing one level.")
    elif increase_path_a or increase_path_b:
        requested_change = "increase"
        if increase_path_a:
            reason_code = "increase_difficulty_path_a"
            feedback_parts.append("Rolling performance is stable enough to increase difficulty.")
        else:
            reason_code = "increase_difficulty_path_b"
            feedback_parts.append("Recent short-window performance supports increasing difficulty.")
    elif metrics["confidence_decline_detected"]:
        reason_code = "confidence_decline_warning"
        feedback_parts.append("Confidence is trending downward; monitor fatigue or caution.")
    else:
        feedback_parts.append("Maintaining current difficulty.")

    if requested_change is not None:
        if cooldown_active:
            reason_code = "cooldown_hold"
            feedback_parts = ["Difficulty change blocked by cooldown; keeping current level."]
            if metrics["confidence_decline_detected"]:
                feedback_parts.append("Confidence is trending downward; monitor fatigue.")
        else:
            next_level = _change_difficulty(current_level, requested_change)
            if next_level != current_level:
                state.last_difficulty_change_trial_index = current_trial_index
                state.difficulty_change_count += 1
            elif requested_change == "decrease":
                if _should_emit_floor_hold(
                    state=state,
                    current_trial_index=current_trial_index,
                    error_streak=metrics["error_streak"],
                ):
                    reason_code = "difficulty_floor_hold"
                    feedback_parts = [
                        "Performance dropped, but difficulty is already at the minimum level."
                    ]
                    state.last_floor_hold_trial_index = current_trial_index
                    state.last_floor_hold_error_streak = metrics["error_streak"]
                else:
                    reason_code = "maintain_difficulty"
                    feedback_parts = ["Maintaining easy difficulty for continued support."]
            elif requested_change == "increase":
                reason_code = "difficulty_ceiling_hold"
                feedback_parts = [
                    "Performance is strong, but difficulty is already at the maximum level."
                ]

    if practice_bias == "left":
        state.left_bias_count += 1
    elif practice_bias == "right":
        state.right_bias_count += 1
    feedback_parts.append(side_feedback_message)

    state.difficulty_level = next_level
    state.practice_bias = practice_bias
    state.schedule_ratio = schedule_ratio

    settings = DIFFICULTY_SETTINGS[next_level]
    return {
        "difficulty_level": next_level,
        "cursor_speed": int(settings["cursor_speed"]),
        "target_size": int(settings["target_size"]),
        "timeout_sec": float(settings["timeout_sec"]),
        "practice_bias": practice_bias,
        "schedule_ratio": schedule_ratio,
        "feedback_message": " ".join(feedback_parts),
        "reason_code": reason_code,
        "rolling_accuracy": float(metrics["rolling_accuracy"]),
        "rolling_mean_confidence": float(metrics["rolling_mean_confidence"]),
        "last_six_accuracy": float(metrics["last_six_accuracy"]),
        "last_six_mean_confidence": float(metrics["last_six_mean_confidence"]),
        "left_accuracy": float(metrics["left_accuracy"]),
        "right_accuracy": float(metrics["right_accuracy"]),
        "side_accuracy_gap": float(metrics["side_accuracy_gap"]),
        "error_streak": int(metrics["error_streak"]),
        "confidence_trend_delta": metrics["confidence_trend_delta"],
        "processed_trials": int(metrics["processed_trials"]),
        "cooldown_active": bool(cooldown_active),
    }


def build_adaptive_session_summary(
    session_summary: dict[str, Any],
    controller_state: AdaptiveControllerState,
) -> dict[str, Any]:
    """Build a final session summary with controller-specific outcomes."""

    return {
        "total_trials": int(session_summary["total_trials"]),
        "correct_count": int(session_summary["correct_count"]),
        "incorrect_count": int(session_summary["incorrect_count"]),
        "final_accuracy": float(session_summary["accuracy"]),
        "final_difficulty_level": controller_state.difficulty_level,
        "number_of_difficulty_changes": int(controller_state.difficulty_change_count),
        "number_of_left_bias_decisions": int(controller_state.left_bias_count),
        "number_of_right_bias_decisions": int(controller_state.right_bias_count),
        "final_practice_bias": controller_state.practice_bias,
        "final_schedule_ratio": controller_state.schedule_ratio,
        "remaining_trials": int(session_summary["remaining_trials"]),
    }
