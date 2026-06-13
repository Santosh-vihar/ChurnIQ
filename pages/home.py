"""
pages/home.py — ChurnIQ Home & Model Overview page

Shows:
  - Hero section with project description
  - Model performance metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Confusion matrix heatmap
  - ROC curve
  - Feature importance bar chart
  - "How it works" explainer
  - Footer with contact placeholders
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Helpers ───────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "pipeline.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.joblib")

# Colour palette (keeping it consistent across the app)
PALETTE = {
    "primary": "#4F46E5",      # indigo
    "secondary": "#7C3AED",    # violet
    "accent": "#06B6D4",       # cyan
    "success": "#10B981",      # emerald
    "warning": "#F59E0B",      # amber
    "danger": "#EF4444",       # red
    "bg": "#FFFFFF",
    "surface": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#1E293B",
    "muted": "#64748B",
}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load saved pipeline and metadata once and cache in session."""
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


def metric_card(label: str, value: str, delta: str = "", color: str = "#4F46E5"):
    """Render a styled metric inside a container."""
    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid {PALETTE['border']};
            border-top: 4px solid {color};
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="font-size: 13px; color: {PALETTE['muted']}; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; line-height: 1;">
                {value}
            </div>
            {"<div style='font-size:12px;color:" + PALETTE['muted'] + ";margin-top:6px;'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_confusion_matrix(cm: list):
    """Return a Plotly heatmap for the confusion matrix."""
    labels = ["No Churn", "Churn"]
    z = np.array(cm)
    z_text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            colorscale=[
                [0, "#EEF2FF"],
                [0.5, "#818CF8"],
                [1, "#4338CA"],
            ],
            showscale=False,
        )
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=16, color=PALETTE["text"])),
        xaxis=dict(title="Predicted", tickfont=dict(size=13)),
        yaxis=dict(title="Actual", tickfont=dict(size=13)),
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_roc_curve(roc_data: dict, auc_score: float):
    """Return a Plotly figure for the ROC curve."""
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]

    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#CBD5E1", width=1.5),
            name="Random Classifier",
            showlegend=True,
        )
    )

    # ROC curve with fill
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color=PALETTE["primary"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.08)",
            name=f"XGBoost  (AUC = {auc_score:.3f})",
        )
    )

    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(size=16, color=PALETTE["text"])),
        xaxis=dict(title="False Positive Rate", range=[0, 1], tickfont=dict(size=12)),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02], tickfont=dict(size=12)),
        legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.8)"),
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_feature_importance(importance: dict, top_n: int = 15):
    """Return a horizontal bar chart of the top-N features by importance."""
    df = pd.DataFrame(importance).sort_values("importance", ascending=True).tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["importance"],
                colorscale=[
                    [0, "#C7D2FE"],
                    [1, "#4338CA"],
                ],
                showscale=False,
            ),
            text=[f"{v:.4f}" for v in df["importance"]],
            textposition="outside",
            textfont=dict(size=11, color=PALETTE["muted"]),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Top {top_n} Feature Importances (XGBoost gain)",
            font=dict(size=16, color=PALETTE["text"]),
        ),
        xaxis=dict(title="Importance Score", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12)),
        height=420,
        margin=dict(l=10, r=60, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Custom CSS ────────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hero gradient banner */
        .hero-banner {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #06B6D4 100%);
            border-radius: 16px;
            padding: 48px 40px;
            color: white;
            margin-bottom: 32px;
        }
        .hero-title {
            font-size: 42px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin: 0 0 12px 0;
        }
        .hero-sub {
            font-size: 17px;
            opacity: 0.88;
            max-width: 620px;
            line-height: 1.6;
            margin: 0;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.20);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
        }

        /* Section headers */
        .section-header {
            font-size: 22px;
            font-weight: 700;
            color: #1E293B;
            margin: 32px 0 16px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-divider {
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 24px 0;
        }

        /* How-it-works steps */
        .step-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            height: 100%;
        }
        .step-number {
            width: 32px; height: 32px;
            background: #4F46E5;
            color: white;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px;
            margin-bottom: 12px;
        }

        /* Footer */
        .footer {
            background: #F1F5F9;
            border-radius: 12px;
            padding: 28px 32px;
            margin-top: 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        .footer-links a {
            color: #4F46E5;
            text-decoration: none;
            font-weight: 500;
            margin-right: 20px;
            font-size: 14px;
        }
        .footer-links a:hover { text-decoration: underline; }

        /* Hide Streamlit default chrome on this page */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Main render function ──────────────────────────────────────────────────────

def render():
    inject_styles()

    # ── Hero section ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">📊 Machine Learning · Customer Analytics</div>
            <h1 class="hero-title">ChurnIQ</h1>
            <p class="hero-sub">
                Identify customers at risk of leaving before they do.
                ChurnIQ uses XGBoost and behavioural telco data to predict churn
                with explainable, actionable insights — helping retention teams
                focus on the customers who matter most.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load artifacts ────────────────────────────────────────────────────────
    pipeline, metadata = load_artifacts()

    if pipeline is None:
        st.warning(
            "⚠️  Model artifacts not found. Please run `python train_model.py` first.",
            icon="⚠️",
        )
        return

    metrics = metadata["metrics"]
    churn_rate = metadata.get("churn_rate", 0.265)

    # ── KPI Metrics row ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Accuracy", f"{metrics['accuracy']:.1%}", color=PALETTE["primary"])
    with col2:
        metric_card("Precision", f"{metrics['precision']:.1%}", color=PALETTE["secondary"])
    with col3:
        metric_card("Recall", f"{metrics['recall']:.1%}", color=PALETTE["accent"])
    with col4:
        metric_card("F1 Score", f"{metrics['f1_score']:.1%}", color=PALETTE["success"])
    with col5:
        metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}", color=PALETTE["warning"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Confusion matrix + ROC curve ──────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Evaluation Charts</div>', unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)
    with col_cm:
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        fig_roc = plot_roc_curve(metrics["roc_curve"], metrics["roc_auc"])
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Feature Importance</div>', unsafe_allow_html=True)
    fig_fi = plot_feature_importance(metadata["feature_importance"])
    st.plotly_chart(fig_fi, use_container_width=True)

    # Training dataset stats
    with st.expander("📋 Dataset & Training Details", expanded=False):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.metric("Training samples", f"{metadata['n_train']:,}")
        with d_col2:
            st.metric("Test samples", f"{metadata['n_test']:,}")
        with d_col3:
            st.metric("Dataset churn rate", f"{churn_rate:.1%}")

        st.markdown("**Features used for prediction:**")
        feature_tags = "  ".join(
            [f"`{f}`" for f in metadata["feature_names"]]
        )
        st.markdown(feature_tags)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Ingestion", "Upload a CSV with customer attributes matching the Telco dataset schema. No coding required."),
        ("2", "Preprocessing", "The pipeline automatically encodes categorical fields and scales numeric values using the same transformer fitted during training."),
        ("3", "XGBoost Prediction", "A gradient-boosted tree ensemble predicts churn probability for every customer in under a second."),
        ("4", "Threshold Tuning", "Adjust the decision threshold to balance precision vs. recall — useful for different business risk tolerances."),
        ("5", "Explainability", "SHAP values (when available) reveal which features drove each individual prediction, making results interpretable."),
        ("6", "Export", "Download the scored dataset as CSV and share with retention teams immediately."),
    ]

    col_a, col_b, col_c = st.columns(3)
    cols = [col_a, col_b, col_c]
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <strong style="font-size:15px; color:#1E293B;">{title}</strong>
                    <p style="font-size:13px; color:{PALETTE['muted']}; margin-top:8px; line-height:1.6;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if i == 2:
            st.markdown("")  # spacing between rows

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="footer">
            <div>
                <strong style="font-size:15px; color:#1E293B;">ChurnIQ</strong><br>
                <span style="font-size:13px; color:#64748B;">
                    Built with XGBoost · Streamlit · SHAP · Plotly
                </span>
            </div>
            <div class="footer-links">
                <a href="https://github.com/Santosh-vihar/ChurnIQ" target="_blank">🐙 GitHub</a>
                <a href="mailto:your@email.com">✉️ Email</a>
                <a href="https://linkedin.com/in/yourprofile" target="_blank">💼 LinkedIn</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
