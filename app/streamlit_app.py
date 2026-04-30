"""Streamlit UI for EEG-based adaptive neurorehabilitation session playback."""

from __future__ import annotations

import json
import time
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_JSON = PROJECT_ROOT / "data" / "results" / "subject5_adaptive_session_v2.json"
MODEL_ARTIFACT_JSON = PROJECT_ROOT / "data" / "artifacts" / "eegbci_subjectwise_final.json"
PLAYBACK_DELAY_SEC = 0.6


def inject_styles() -> None:
    """Inject lightweight CSS for a cleaner demo layout."""

    st.markdown(
        """
        <style>
        .app-subtitle {
            color: #475569;
            font-size: 1.02rem;
            margin-bottom: 1rem;
        }
        .status-banner {
            background: #f8fafc;
            border: 1px solid #dbe4ee;
            border-radius: 14px;
            color: #334155;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        .complete-banner {
            background: #ecfeff;
            border: 1px solid #99f6e4;
            border-radius: 16px;
            color: #115e59;
            font-size: 1rem;
            font-weight: 800;
            padding: 0.95rem 1rem;
            margin: 1.25rem 0 0.9rem 0;
        }
        .demo-card {
            background: #ffffff;
            border: 1px solid #dbe4ee;
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.05);
            min-height: 360px;
        }
        .card-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }
        .track-labels {
            display: flex;
            justify-content: space-between;
            color: #475569;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .track-line {
            position: relative;
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #e2e8f0, #cbd5e1);
            margin: 1.15rem 0 1.5rem 0;
        }
        .target-marker {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 18px;
            height: 18px;
            border-radius: 50%;
            border: 3px solid #94a3b8;
            background: #ffffff;
        }
        .target-marker.left {
            left: 12%;
        }
        .target-marker.right {
            left: 88%;
        }
        .target-marker.active {
            width: 28px;
            height: 28px;
            border-color: #0f766e;
            background: #99f6e4;
            box-shadow: 0 0 0 8px rgba(20, 184, 166, 0.18);
        }
        .cursor-marker {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18);
        }
        .cursor-hit {
            background: #15803d;
        }
        .cursor-miss {
            background: #b91c1c;
        }
        .cursor-neutral {
            background: #475569;
        }
        .pill-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-bottom: 0.95rem;
        }
        .pill {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            display: inline-block;
        }
        .pill-hit {
            background: #dcfce7;
            color: #166534;
        }
        .pill-miss {
            background: #fee2e2;
            color: #991b1b;
        }
        .pill-neutral {
            background: #e2e8f0;
            color: #334155;
        }
        .result-text {
            font-size: 0.96rem;
            font-weight: 800;
            display: inline-block;
            padding: 0.3rem 0;
        }
        .result-hit {
            color: #15803d;
        }
        .result-miss {
            color: #b91c1c;
        }
        .result-waiting {
            color: #475569;
        }
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.75rem;
        }
        .detail-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
        }
        .detail-label {
            color: #64748b;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }
        .detail-value {
            color: #0f172a;
            font-size: 1rem;
            font-weight: 700;
        }
        .reason-chip {
            display: inline-block;
            margin: 0.35rem 0 0.85rem 0;
            padding: 0.45rem 0.7rem;
            background: #eff6ff;
            color: #1d4ed8;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.82rem;
        }
        .feedback-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            color: #334155;
            line-height: 1.45;
            padding: 0.9rem 1rem;
            margin-top: 0.45rem;
        }
        .trial-meta {
            color: #64748b;
            font-size: 0.86rem;
            margin-top: 0.8rem;
        }
        .summary-card {
            background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
            border: 1px solid #cbd5e1;
            border-left: 6px solid #0f766e;
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            margin-top: 0.25rem;
        }
        .summary-title {
            color: #0f172a;
            font-size: 1.12rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .summary-text {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-top: 0.6rem;
        }
        html {
            scroll-behavior: smooth;
        }
        .stApp {
            background: #000000;
            color: #dbeafe;
        }
        [data-testid="stHeader"] {
            background: #000000;
        }
        [data-testid="stAppViewContainer"] {
            background: #000000;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stDecoration"] {
            background: #000000;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown,
        .stMetric label,
        .stMetric [data-testid="stMetricValue"] {
            color: #e5f6ff;
        }
        .landing-hero {
            min-height: 92vh;
            display: grid;
            align-items: center;
            padding: 2.2rem 0 4.5rem 0;
        }
        .project-title {
            color: #f8fbff;
            font-size: 3.25rem;
            font-weight: 900;
            letter-spacing: 0;
            line-height: 1.05;
            margin-bottom: 2.8rem;
            padding-bottom: 1.1rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.26);
        }
        .hero-shell {
            display: grid;
            grid-template-columns: minmax(0, 1.12fr) minmax(340px, 0.78fr);
            gap: 2rem;
            align-items: center;
        }
        .hero-kicker,
        .section-kicker {
            color: #5eead4;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }
        .hero-title {
            color: #f8fbff;
            font-size: 4.7rem;
            line-height: 0.98;
            font-weight: 900;
            letter-spacing: 0;
            margin: 0 0 1.35rem 0;
        }
        .hero-copy {
            color: #c6d6e2;
            font-size: 1.24rem;
            line-height: 1.75;
            max-width: 720px;
            margin-bottom: 1.55rem;
        }
        .hero-action-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
            margin-top: 1.6rem;
        }
        .hero-button,
        .hero-button-secondary {
            border-radius: 14px;
            display: inline-block;
            font-weight: 900;
            padding: 0.86rem 1.08rem;
            text-decoration: none !important;
        }
        .hero-button {
            background: #19c6c0;
            color: #001012 !important;
            box-shadow: none;
        }
        .hero-button-secondary {
            background: #000000;
            border: 1px solid rgba(148, 163, 184, 0.34);
            color: #dff8ff !important;
        }
        .hero-visual,
        .landing-card,
        .metric-showcase,
        .future-card,
        .demo-shell,
        .footer-shell {
            background: #000000;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 14px;
            box-shadow: none;
        }
        .hero-visual {
            padding: 1.25rem;
        }
        .signal-panel {
            border-radius: 14px;
            background: #000000;
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 1rem;
        }
        .signal-row {
            display: grid;
            grid-template-columns: 0.8fr 1fr 0.8fr;
            gap: 0.65rem;
            align-items: center;
            margin: 0.9rem 0;
        }
        .wave-line {
            height: 48px;
            border-radius: 10px;
            background:
                repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.14), rgba(148, 163, 184, 0.14) 1px, transparent 1px, transparent 18px),
                #020405;
            position: relative;
            overflow: hidden;
        }
        .wave-line:after {
            content: "";
            position: absolute;
            inset: 16px 0;
            background: #5eead4;
            clip-path: polygon(0 55%, 8% 48%, 16% 65%, 24% 35%, 32% 52%, 40% 28%, 48% 68%, 56% 42%, 64% 59%, 72% 32%, 80% 63%, 88% 46%, 100% 54%, 100% 62%, 88% 54%, 80% 71%, 72% 41%, 64% 67%, 56% 50%, 48% 78%, 40% 36%, 32% 61%, 24% 45%, 16% 75%, 8% 58%, 0 65%);
        }
        .flow-node {
            color: #e0f2fe;
            font-weight: 900;
            text-align: center;
            border-radius: 14px;
            background: rgba(3, 7, 9, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 0.75rem 0.7rem;
        }
        .hero-stat-grid,
        .landing-grid,
        .metric-grid,
        .future-grid {
            display: grid;
            gap: 1rem;
        }
        .hero-stat-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1rem;
        }
        .hero-stat {
            border-radius: 16px;
            background: rgba(0, 0, 0, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 0.9rem;
        }
        .hero-stat strong {
            color: #ffffff;
            display: block;
            font-size: 1.45rem;
        }
        .hero-stat span {
            color: #9fb8ca;
            font-size: 0.83rem;
        }
        .landing-section {
            padding: 2.4rem 0;
        }
        .landing-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .landing-card,
        .metric-showcase,
        .future-card {
            padding: 1.25rem;
        }
        .landing-card h3,
        .metric-showcase h3,
        .future-card h3 {
            color: #f8fbff;
            font-size: 1.18rem;
            margin: 0 0 0.55rem 0;
        }
        .landing-card p,
        .metric-showcase p,
        .future-card p,
        .footer-shell p {
            color: #adc8dc;
            font-size: 1.08rem;
            line-height: 1.65;
            margin: 0;
        }
        .metric-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .metric-showcase strong {
            color: #5eead4;
            display: block;
            font-size: 2rem;
            margin-bottom: 0.2rem;
        }
        .future-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .demo-shell {
            padding: 1.2rem;
            margin-top: 1rem;
        }
        .demo-heading {
            margin-bottom: 1.1rem;
        }
        .demo-heading h2 {
            color: #f8fbff;
            font-size: 2.7rem;
            line-height: 1.05;
            margin: 0.25rem 0 0.55rem 0;
        }
        .demo-heading p {
            color: #adc8dc;
            line-height: 1.65;
            max-width: 840px;
        }
        .app-subtitle {
            color: #adc8dc;
            font-size: 1.12rem;
        }
        .status-banner,
        .demo-card,
        .detail-box,
        .feedback-box,
        .summary-card {
            background: rgba(5, 8, 10, 0.9);
            border-color: rgba(148, 163, 184, 0.2);
            color: #cfe7f7;
        }
        .demo-card {
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
        }
        .card-title,
        .detail-value,
        .summary-title,
        .transcript-header h2 {
            color: #f8fbff;
        }
        .detail-label,
        .track-labels,
        .trial-meta,
        .summary-text {
            color: #9fb8ca;
        }
        .track-line {
            background: linear-gradient(90deg, #080d10, #111b21);
        }
        .target-marker {
            border-color: #557184;
            background: #0b1620;
        }
        .target-marker.active {
            border-color: #5eead4;
            background: #134e4a;
            box-shadow: 0 0 0 8px rgba(20, 184, 166, 0.16);
        }
        .pill-neutral {
            background: rgba(148, 163, 184, 0.16);
            color: #cbd5e1;
        }
        .reason-chip {
            background: rgba(37, 99, 235, 0.18);
            color: #bfdbfe;
        }
        .complete-banner {
            background: rgba(20, 184, 166, 0.12);
            border-color: rgba(94, 234, 212, 0.32);
            color: #99f6e4;
        }
        .stButton > button {
            background: linear-gradient(135deg, #22d3ee, #14b8a6);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 14px;
            color: #001012;
            font-weight: 900;
        }
        .stButton > button:disabled {
            background: rgba(16, 24, 28, 0.9);
            color: #7f95a8;
            border-color: rgba(148, 163, 184, 0.12);
        }
        .stDataFrame,
        [data-testid="stDataFrame"] {
            background: rgba(8, 21, 31, 0.88);
        }
        .footer-shell {
            padding: 1.4rem;
            margin-top: 2.2rem;
        }
        .footer-grid {
            display: grid;
            grid-template-columns: 1.4fr repeat(3, minmax(0, 0.6fr));
            gap: 1rem;
            align-items: start;
        }
        .footer-shell h3,
        .footer-shell strong {
            color: #f8fbff;
        }
        .footer-link {
            color: #5eead4;
            display: block;
            margin-top: 0.35rem;
            text-decoration: none !important;
        }
        @media (max-width: 920px) {
            .hero-shell,
            .landing-grid,
            .metric-grid,
            .future-grid,
            .footer-grid {
                grid-template-columns: 1fr;
            }
            .hero-stat-grid {
                grid-template-columns: 1fr;
            }
            .landing-hero {
                min-height: auto;
                padding-top: 1rem;
            }
            .hero-title {
                font-size: 3rem;
            }
            .project-title {
                font-size: 2.2rem;
            }
            .demo-heading h2 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_app_state() -> None:
    """Ensure required Streamlit state fields exist."""

    defaults = {
        "session_payload": None,
        "trial_history": [],
        "current_trial_index": -1,
        "correct_count": 0,
        "incorrect_count": 0,
        "status_message": "Load the saved adaptive session JSON to start playback.",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_session_payload(json_path: Path) -> dict[str, Any]:
    """Load the adaptive session JSON from disk."""

    if not json_path.exists():
        raise FileNotFoundError(f"Session JSON not found: {json_path}")

    payload = json.loads(json_path.read_text())
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Session JSON must contain a non-empty 'history' list.")
    return payload


def load_project_metrics() -> dict[str, Any]:
    """Load project metrics for the landing-page summary cards."""

    metrics: dict[str, Any] = {
        "model_accuracy": 0.0,
        "model_precision": 0.0,
        "model_f1": 0.0,
        "session_accuracy": 0.0,
        "total_trials": 0,
        "difficulty_changes": 0,
        "final_difficulty": "N/A",
        "final_bias": "N/A",
        "train_subjects": "N/A",
        "test_subjects": "N/A",
        "channels": 10,
        "epoch_window": "0.5-2.5s",
    }

    try:
        artifact_payload = json.loads(MODEL_ARTIFACT_JSON.read_text())
        evaluation = artifact_payload.get("evaluation", {})
        metrics.update(
            {
                "model_accuracy": float(evaluation.get("accuracy", 0.0)),
                "model_precision": float(evaluation.get("precision", 0.0)),
                "model_f1": float(evaluation.get("f1_score", 0.0)),
                "train_subjects": ", ".join(
                    str(item) for item in artifact_payload.get("train_subjects", [])
                ) or "N/A",
                "test_subjects": ", ".join(
                    str(item) for item in artifact_payload.get("test_subjects", [])
                ) or "N/A",
                "channels": int(evaluation.get("n_channels", metrics["channels"])),
                "epoch_window": (
                    f"{artifact_payload.get('epoch_window', {}).get('tmin', 0.5)}-"
                    f"{artifact_payload.get('epoch_window', {}).get('tmax', 2.5)}s"
                ),
            }
        )
    except Exception:
        pass

    try:
        session_payload = json.loads(DEFAULT_SESSION_JSON.read_text())
        adaptive_summary = session_payload.get("adaptive_summary", {})
        metrics.update(
            {
                "session_accuracy": float(adaptive_summary.get("final_accuracy", 0.0)),
                "total_trials": int(adaptive_summary.get("total_trials", 0)),
                "difficulty_changes": int(
                    adaptive_summary.get("number_of_difficulty_changes", 0)
                ),
                "final_difficulty": str(
                    adaptive_summary.get("final_difficulty_level", "N/A")
                ).title(),
                "final_bias": str(adaptive_summary.get("final_practice_bias", "N/A")).title(),
            }
        )
    except Exception:
        pass

    return metrics


def percent_text(value: float) -> str:
    """Format a fractional metric as a percentage string."""

    return f"{value * 100:.2f}%"


def render_landing_page() -> None:
    """Render the narrative hero and project overview before the demo controls."""

    metrics = load_project_metrics()
    st.markdown(
        f"""
        <section class="landing-hero">
          <div class="project-title">EEG-Based Neurorehabilitation System</div>
          <div class="hero-shell">
            <div>
              <div class="hero-kicker">EEG motor-imagery classification and adaptive feedback</div>
              <h1 class="hero-title">A signal-driven rehabilitation demo for left and right motor intent.</h1>
              <p class="hero-copy">
                This project uses EEG motor-imagery data to classify imagined left and right
                movement. The decoded output is connected to a rehabilitation-style task where
                each prediction updates the session state, feedback, and adaptive controller.
              </p>
              <p class="hero-copy">
                The system is built as an offline research playback: EEG trials are processed,
                model confidence is tracked, and the controller adjusts difficulty and practice
                bias. The aim is to show how machine learning can support more responsive
                neurorehabilitation sessions.
              </p>
              <div class="hero-action-row">
                <a class="hero-button" href="#demo-workflow">Open the playback demo</a>
                <a class="hero-button-secondary" href="#project-metrics">See model accuracy</a>
              </div>
            </div>
            <div class="hero-visual">
              <div class="signal-panel">
                <div class="signal-row">
                  <div class="flow-node">EEG Signal</div>
                  <div class="wave-line"></div>
                  <div class="flow-node">ML Decoder</div>
                </div>
                <div class="signal-row">
                  <div class="flow-node">Motor Intent</div>
                  <div class="wave-line"></div>
                  <div class="flow-node">Adaptive Task</div>
                </div>
                <div class="hero-stat-grid">
                  <div class="hero-stat">
                    <strong>{metrics["channels"]}</strong>
                    <span>motor cortex EEG channels</span>
                  </div>
                  <div class="hero-stat">
                    <strong>{metrics["epoch_window"]}</strong>
                    <span>signal window</span>
                  </div>
                  <div class="hero-stat">
                    <strong>{metrics["final_difficulty"]}</strong>
                    <span>final adaptive level</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section class="landing-section">
          <div class="section-kicker">How the system helps</div>
          <div class="landing-grid">
            <div class="landing-card">
              <h3>Transforms EEG into intent</h3>
              <p>
                Motor-imagery EEG is filtered, windowed, and converted into CSP features so the
                model can classify imagined left/right movement.
              </p>
            </div>
            <div class="landing-card">
              <h3>Adapts therapy pressure</h3>
              <p>
                The controller uses correctness, confidence, error streaks, and side imbalance to
                tune difficulty and practice ratio across the session.
              </p>
            </div>
            <div class="landing-card">
              <h3>Explains each decision</h3>
              <p>
                Every trial records prediction, confidence, outcome, difficulty, bias, and feedback
                so the rehab workflow is easier to inspect and present.
              </p>
            </div>
          </div>
        </section>

        <section class="landing-section" id="project-metrics">
          <div class="section-kicker">Model and session performance</div>
          <div class="metric-grid">
            <div class="metric-showcase">
              <strong>{percent_text(metrics["model_accuracy"])}</strong>
              <h3>Model accuracy</h3>
              <p>Subject-wise CSP + SVM evaluation trained on subjects {escape(metrics["train_subjects"])} and tested on subject {escape(metrics["test_subjects"])}.</p>
            </div>
            <div class="metric-showcase">
              <strong>{percent_text(metrics["model_precision"])}</strong>
              <h3>Precision</h3>
              <p>Precision from the saved evaluation artifact, useful for explaining prediction reliability during the demo.</p>
            </div>
            <div class="metric-showcase">
              <strong>{percent_text(metrics["session_accuracy"])}</strong>
              <h3>Adaptive session accuracy</h3>
              <p>{metrics["total_trials"]} replayed trials with {metrics["difficulty_changes"]} difficulty changes and final bias set to {escape(metrics["final_bias"])}.</p>
            </div>
            <div class="metric-showcase">
              <strong>{percent_text(metrics["model_f1"])}</strong>
              <h3>F1 score</h3>
              <p>A balanced view of precision and recall for the motor-imagery classifier in this prototype.</p>
            </div>
          </div>
        </section>

        """,
        unsafe_allow_html=True,
    )


def render_demo_intro() -> None:
    """Render a visual wrapper before the existing playback workflow."""

    st.markdown(
        """
        <section class="demo-shell" id="demo-workflow">
          <div class="demo-heading">
            <div class="section-kicker">Interactive playback</div>
            <h2>Load the saved EEG rehabilitation session and watch the controller adapt.</h2>
            <p>
              The original app starts here. Use the controls below to load the dataset, play
              the session, step through individual trials, and inspect the model/controller state.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render footer contact and project information."""

    st.markdown(
        """
        <footer class="footer-shell">
          <div class="footer-grid">
            <div>
              <h3>Contact Us</h3>
              <p>
                EEG-Based Adaptive Neurorehabilitation System. Built as a final year project
                demonstrating AI-assisted motor-imagery rehabilitation and adaptive feedback.
              </p>
            </div>
            <div>
              <strong>Project</strong>
              <a class="footer-link" href="#demo-workflow">Playback demo</a>
              <a class="footer-link" href="#project-metrics">Accuracy</a>
            </div>
            <div>
              <strong>Research Focus</strong>
              <p>EEG, CSP, SVM, adaptive control, motor imagery.</p>
            </div>
            <div>
              <strong>Get in touch</strong>
              <p>For demo queries, model details, or collaboration.</p>
              <a class="footer-link" href="mailto:contact@example.com">contact@example.com</a>
            </div>
          </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def reset_playback_state(payload: dict[str, Any]) -> None:
    """Reset counters and playback index for a loaded session payload."""

    st.session_state.session_payload = payload
    st.session_state.trial_history = payload["history"]
    st.session_state.current_trial_index = -1
    st.session_state.correct_count = 0
    st.session_state.incorrect_count = 0
    st.session_state.status_message = (
        f"Loaded {len(payload['history'])} trials from {DEFAULT_SESSION_JSON.name}."
    )


def label_to_side(label_name: str | None, label_value: int | None) -> str:
    """Convert stored labels into a presentation-friendly side name."""

    if label_name:
        normalized = str(label_name).strip().upper()
        if normalized in {"LEFT", "T1"}:
            return "Left"
        if normalized in {"RIGHT", "T2"}:
            return "Right"
    if label_value == 0:
        return "Left"
    if label_value == 1:
        return "Right"
    return "Unknown"


def get_current_trial() -> dict[str, Any] | None:
    """Return the currently displayed trial if playback has started."""

    index = st.session_state.current_trial_index
    history = st.session_state.trial_history
    if index < 0 or index >= len(history):
        return None
    return history[index]


def get_processed_history() -> list[dict[str, Any]]:
    """Return trials that have already been processed during playback."""

    index = st.session_state.current_trial_index
    if index < 0:
        return []
    return st.session_state.trial_history[: index + 1]


def advance_one_trial() -> bool:
    """Advance playback by one trial and update counters."""

    history = st.session_state.trial_history
    next_index = st.session_state.current_trial_index + 1
    if next_index >= len(history):
        st.session_state.status_message = "Reached the end of the session."
        return False

    trial = history[next_index]
    st.session_state.current_trial_index = next_index
    if bool(trial.get("correct")):
        st.session_state.correct_count += 1
    else:
        st.session_state.incorrect_count += 1

    st.session_state.status_message = (
        f"Showing trial {next_index + 1} of {len(history)}."
    )
    return True


def compute_running_accuracy() -> float:
    """Return running accuracy for processed trials."""

    processed = st.session_state.correct_count + st.session_state.incorrect_count
    if processed == 0:
        return 0.0
    return st.session_state.correct_count / processed


def get_error_streak(trial: dict[str, Any] | None) -> int:
    """Return the current controller error streak."""

    if not trial:
        return 0
    controller = trial.get("controller", {})
    return int(controller.get("error_streak", 0))


def is_session_complete() -> bool:
    """Return whether playback has reached the final trial."""

    total_trials = len(st.session_state.trial_history)
    if total_trials == 0:
        return False
    processed_trials = max(st.session_state.current_trial_index + 1, 0)
    return processed_trials >= total_trials


def build_final_summary() -> dict[str, Any]:
    """Build a final session summary from payload metadata and processed history."""

    payload = st.session_state.session_payload or {}
    adaptive_summary = payload.get("adaptive_summary", {})
    processed_history = get_processed_history()
    total_trials = len(st.session_state.trial_history)
    prediction_counts: dict[str, int] = {"Left": 0, "Right": 0, "Unknown": 0}

    for item in processed_history:
        side = label_to_side(
            item.get("predicted_label_name"),
            item.get("predicted_label"),
        )
        prediction_counts[side] = prediction_counts.get(side, 0) + 1

    most_common_predicted_side = max(
        prediction_counts.items(),
        key=lambda item: item[1],
    )[0]
    first_difficulty = "N/A"
    final_difficulty = str(adaptive_summary.get("final_difficulty_level", "N/A")).title()
    if processed_history:
        first_controller = processed_history[0].get("controller", {})
        last_controller = processed_history[-1].get("controller", {})
        first_difficulty = str(first_controller.get("difficulty_level", "N/A")).title()
        if final_difficulty == "N/A":
            final_difficulty = str(last_controller.get("difficulty_level", "N/A")).title()

    return {
        "total_trials": int(adaptive_summary.get("total_trials", total_trials)),
        "correct_count": int(adaptive_summary.get("correct_count", st.session_state.correct_count)),
        "incorrect_count": int(
            adaptive_summary.get("incorrect_count", st.session_state.incorrect_count)
        ),
        "final_accuracy": float(
            adaptive_summary.get("final_accuracy", compute_running_accuracy())
        ),
        "first_difficulty_level": first_difficulty,
        "final_difficulty_level": final_difficulty,
        "number_of_difficulty_changes": int(
            adaptive_summary.get("number_of_difficulty_changes", 0)
        ),
        "final_practice_bias": str(
            adaptive_summary.get("final_practice_bias", "balanced")
        ).title(),
        "final_schedule_ratio": str(adaptive_summary.get("final_schedule_ratio", "1:1")),
        "number_of_left_bias_decisions": int(
            adaptive_summary.get("number_of_left_bias_decisions", 0)
        ),
        "number_of_right_bias_decisions": int(
            adaptive_summary.get("number_of_right_bias_decisions", 0)
        ),
        "difficulty_changed": int(
            adaptive_summary.get("number_of_difficulty_changes", 0)
        ) > 0,
        "most_common_predicted_side": most_common_predicted_side,
    }


def build_summary_interpretation(summary: dict[str, Any]) -> str:
    """Build a short human-readable interpretation for the final summary."""

    difficulty_text = (
        "The controller changed difficulty during the session"
        if summary["difficulty_changed"]
        else "The controller kept the same difficulty level throughout the session"
    )
    bias_text = "balanced practice" if summary["final_practice_bias"] == "Balanced" else (
        f"{summary['final_practice_bias'].lower()}-focused practice"
    )
    return (
        "The session completed successfully. "
        f"{difficulty_text}, ending at {summary['final_difficulty_level'].lower()} difficulty. "
        f"It finished with {bias_text} at a {summary['final_schedule_ratio']} schedule and "
        f"the most common predicted side was {summary['most_common_predicted_side'].lower()}."
    )


def render_final_summary() -> None:
    """Render a final summary section when playback is complete."""

    summary = build_final_summary()
    interpretation = build_summary_interpretation(summary)

    st.markdown(
        '<div class="complete-banner">Session Complete</div>',
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            """
            <div class="summary-card">
              <div class="summary-title">Session Summary</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        perf_col1.metric("Total Trials", summary["total_trials"])
        perf_col2.metric("Correct Count", summary["correct_count"])
        perf_col3.metric("Incorrect Count", summary["incorrect_count"])
        perf_col4.metric("Final Accuracy", f"{summary['final_accuracy'] * 100:.2f}%")

        st.markdown("#### Adaptive Behavior")
        adapt_col1, adapt_col2, adapt_col3, adapt_col4 = st.columns(4)
        adapt_col1.metric("Final Difficulty", summary["final_difficulty_level"])
        adapt_col2.metric("Difficulty Changes", summary["number_of_difficulty_changes"])
        adapt_col3.metric("Final Practice Bias", summary["final_practice_bias"])
        adapt_col4.metric("Final Schedule Ratio", summary["final_schedule_ratio"])

        extra_col1, extra_col2, extra_col3, extra_col4 = st.columns(4)
        extra_col1.metric("Left-Bias Decisions", summary["number_of_left_bias_decisions"])
        extra_col2.metric("Right-Bias Decisions", summary["number_of_right_bias_decisions"])
        extra_col3.metric("First Difficulty", summary["first_difficulty_level"])
        extra_col4.metric(
            "Most Common Predicted Side",
            summary["most_common_predicted_side"],
        )

        st.markdown(
            f'<div class="summary-text">{escape(interpretation)}</div>',
            unsafe_allow_html=True,
        )


def build_rehab_task_card(
    trial: dict[str, Any] | None,
    processed_trials: int,
    total_trials: int,
) -> str:
    """Build the rehab task simulation card as lightweight HTML."""

    if trial is None:
        return f"""
        <div class="demo-card">
          <div class="card-title">Rehab Task Simulation</div>
          <div class="track-labels">
            <span>Left Target</span>
            <span>Right Target</span>
          </div>
          <div class="track-line">
            <div class="target-marker left"></div>
            <div class="target-marker right"></div>
            <div class="cursor-marker cursor-neutral" style="left: 50%;">EEG</div>
          </div>
          <div class="pill-row">
            <span class="pill pill-neutral">Awaiting Playback</span>
            <span class="result-text result-waiting">Waiting</span>
          </div>
          <div class="detail-grid">
            <div class="detail-box">
              <div class="detail-label">Current Trial</div>
              <div class="detail-value">0 / {total_trials}</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Target Side</div>
              <div class="detail-value">Not started</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Predicted Side</div>
              <div class="detail-value">Not started</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Result</div>
              <div class="detail-value">Waiting</div>
            </div>
          </div>
          <div class="trial-meta">Load the session and step through trials to begin playback.</div>
        </div>
        """

    true_side = label_to_side(trial.get("true_label_name"), trial.get("true_label"))
    predicted_side = label_to_side(
        trial.get("predicted_label_name"),
        trial.get("predicted_label"),
    )
    is_correct = bool(trial.get("correct"))
    cursor_left = "30%" if predicted_side == "Left" else "70%" if predicted_side == "Right" else "50%"
    cursor_class = "cursor-hit" if is_correct else "cursor-miss"
    outcome_text = "✔ Hit" if is_correct else "✘ Miss"
    outcome_class = "result-hit" if is_correct else "result-miss"
    left_target_class = "target-marker left active" if true_side == "Left" else "target-marker left"
    right_target_class = (
        "target-marker right active" if true_side == "Right" else "target-marker right"
    )

    return f"""
    <div class="demo-card">
      <div class="card-title">Rehab Task Simulation</div>
      <div class="track-labels">
        <span>Left Target</span>
        <span>Right Target</span>
      </div>
      <div class="track-line">
        <div class="{left_target_class}"></div>
        <div class="{right_target_class}"></div>
        <div class="cursor-marker {cursor_class}" style="left: {cursor_left};">EEG</div>
      </div>
      <div class="pill-row">
        <span class="pill pill-neutral">Trial {processed_trials} / {total_trials}</span>
        <span class="result-text {outcome_class}">{outcome_text}</span>
      </div>
      <div class="detail-grid">
        <div class="detail-box">
          <div class="detail-label">Target Side</div>
          <div class="detail-value">{escape(true_side)}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Predicted Side</div>
          <div class="detail-value">{escape(predicted_side)}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Confidence</div>
          <div class="detail-value">{float(trial.get("confidence", 0.0)):.2f}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Outcome</div>
          <div class="detail-value">{outcome_text}</div>
        </div>
      </div>
      <div class="trial-meta">Subject {trial.get('subject_id')} | Run {trial.get('run_id')}</div>
    </div>
    """


def build_controller_card(trial: dict[str, Any] | None) -> str:
    """Build the adaptive controller state card as HTML."""

    if trial is None:
        return """
        <div class="demo-card">
          <div class="card-title">Adaptive Controller State</div>
          <div class="detail-grid">
            <div class="detail-box">
              <div class="detail-label">Difficulty</div>
              <div class="detail-value">Easy</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Bias</div>
              <div class="detail-value">Balanced</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Ratio</div>
              <div class="detail-value">1:1</div>
            </div>
            <div class="detail-box">
              <div class="detail-label">Confidence</div>
              <div class="detail-value">0.00</div>
            </div>
          </div>
          <div class="reason-chip">reason: waiting_for_session</div>
          <div class="feedback-box">Adaptive controller output will appear after the first processed trial.</div>
        </div>
        """

    controller = trial.get("controller", {})
    confidence = float(trial.get("confidence", 0.0))
    difficulty = str(controller.get("difficulty_level", "N/A")).title()
    bias = str(controller.get("practice_bias", "balanced")).title()
    ratio = str(controller.get("schedule_ratio", "1:1"))
    reason = escape(str(controller.get("reason_code", "N/A")))
    feedback = escape(str(controller.get("feedback_message", "No feedback available.")))

    return f"""
    <div class="demo-card">
      <div class="card-title">Adaptive Controller State</div>
      <div class="detail-grid">
        <div class="detail-box">
          <div class="detail-label">Difficulty</div>
          <div class="detail-value">{escape(difficulty)}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Bias</div>
          <div class="detail-value">{escape(bias)}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Ratio</div>
          <div class="detail-value">{escape(ratio)}</div>
        </div>
        <div class="detail-box">
          <div class="detail-label">Confidence</div>
          <div class="detail-value">{confidence:.2f}</div>
        </div>
      </div>
      <div class="reason-chip">reason: {reason}</div>
      <div class="feedback-box">{feedback}</div>
    </div>
    """


def build_recent_history_rows() -> list[dict[str, Any]]:
    """Build a compact table for the last 10 processed trials."""

    rows: list[dict[str, Any]] = []
    for item in reversed(get_processed_history()[-10:]):
        controller = item.get("controller", {})
        rows.append(
            {
                "trial": int(item.get("trial_index", 0)) + 1,
                "true": label_to_side(item.get("true_label_name"), item.get("true_label")),
                "pred": label_to_side(
                    item.get("predicted_label_name"),
                    item.get("predicted_label"),
                ),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "correct": "Yes" if item.get("correct") else "No",
                "difficulty": str(controller.get("difficulty_level", "N/A")).title(),
                "bias": str(controller.get("practice_bias", "balanced")).title(),
            }
        )
    return rows


def render_dashboard(main_placeholder: Any, bottom_placeholder: Any) -> None:
    """Render the main cards, metrics, and recent history."""

    current_trial = get_current_trial()
    total_trials = len(st.session_state.trial_history)
    processed_trials = max(st.session_state.current_trial_index + 1, 0)
    running_accuracy = compute_running_accuracy()
    progress_value = (processed_trials / total_trials) if total_trials else 0.0
    recent_rows = build_recent_history_rows()

    with main_placeholder.container():
        left_col, right_col = st.columns([1.2, 1.0], gap="large")
        with left_col:
            st.markdown(
                build_rehab_task_card(
                    trial=current_trial,
                    processed_trials=processed_trials,
                    total_trials=total_trials,
                ),
                unsafe_allow_html=True,
            )
        with right_col:
            st.markdown(build_controller_card(current_trial), unsafe_allow_html=True)

    with bottom_placeholder.container():
        st.markdown("### Session Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Processed Trials", processed_trials)
        metric_col2.metric("Running Accuracy", f"{running_accuracy * 100:.2f}%")
        metric_col3.metric("Correct Count", st.session_state.correct_count)
        metric_col4.metric("Incorrect Count", st.session_state.incorrect_count)
        st.progress(progress_value, text=f"Session progress: {processed_trials}/{total_trials}")

        st.markdown("### Recent History")
        if recent_rows:
            st.dataframe(recent_rows, use_container_width=True)
        else:
            st.info("No processed trials yet. Step through the session to populate history.")

        if current_trial is not None:
            st.caption(f"Current error streak: {get_error_streak(current_trial)}")

        if is_session_complete():
            render_final_summary()


def autoplay_session(main_placeholder: Any, bottom_placeholder: Any) -> None:
    """Play all remaining trials sequentially with a small delay."""

    advanced = False
    while advance_one_trial():
        advanced = True
        render_dashboard(main_placeholder=main_placeholder, bottom_placeholder=bottom_placeholder)
        time.sleep(PLAYBACK_DELAY_SEC)

    if advanced:
        st.session_state.status_message = "Session playback complete."


def reset_current_session() -> None:
    """Reset the current loaded session back to trial zero."""

    payload = st.session_state.session_payload
    if payload is None:
        st.session_state.status_message = "No loaded session to reset."
        return
    reset_playback_state(payload)
    st.session_state.status_message = "Playback reset. Ready to replay the session."


def main() -> None:
    """Render the Streamlit playback application."""

    st.set_page_config(
        page_title="EEG-Based Adaptive Neurorehabilitation Demo",
        layout="wide",
    )
    inject_styles()
    initialize_app_state()

    render_landing_page()
    render_demo_intro()

    st.title("EEG-Based Adaptive Neurorehabilitation Demo")
    st.markdown(
        '<div class="app-subtitle">Offline playback of an EEG motor imagery decoder driving an explainable adaptive rehabilitation session.</div>',
        unsafe_allow_html=True,
    )

    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    with control_col1:
        load_clicked = st.button("Load Session", use_container_width=True)
    with control_col2:
        play_clicked = st.button(
            "Play",
            use_container_width=True,
            disabled=not st.session_state.trial_history,
        )
    with control_col3:
        next_clicked = st.button(
            "Next Trial",
            use_container_width=True,
            disabled=not st.session_state.trial_history,
        )
    with control_col4:
        reset_clicked = st.button(
            "Reset",
            use_container_width=True,
            disabled=st.session_state.session_payload is None,
        )

    if load_clicked:
        try:
            payload = load_session_payload(DEFAULT_SESSION_JSON)
            reset_playback_state(payload)
        except Exception as exc:
            st.session_state.status_message = f"Failed to load session JSON: {exc}"

    if reset_clicked:
        reset_current_session()

    main_placeholder = st.empty()
    bottom_placeholder = st.empty()

    if next_clicked and st.session_state.trial_history:
        advance_one_trial()

    if play_clicked and st.session_state.trial_history:
        autoplay_session(main_placeholder=main_placeholder, bottom_placeholder=bottom_placeholder)

    st.markdown(
        f'<div class="status-banner">{escape(st.session_state.status_message)}</div>',
        unsafe_allow_html=True,
    )
    render_dashboard(main_placeholder=main_placeholder, bottom_placeholder=bottom_placeholder)


if __name__ == "__main__":
    main()
