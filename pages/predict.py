"""
pages/predict.py — ChurnIQ Predict & Upload page

Features:
  - CSV upload with instant preview
  - Batch churn prediction using saved joblib pipeline
  - Adjustable probability threshold slider
  - High-risk customer highlighting
  - Churn count & probability distribution charts
  - SHAP waterfall / summary (optional — works without it)
  - Groq LLM retention summary (optional — works without it)
  - Download predictions as CSV
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "artifacts"
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "pipeline.joblib")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.joblib")

PALETTE = {
    "primary": "#4F46E5",
    "secondary": "#7C3AED",
    "accent": "#06B6D4",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "surface": "#F8FAFC",
    "text": "#1E293B",
}


@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load saved pipeline and metadata once and cache in session. Trains model on-the-fly if missing."""
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(METADATA_PATH):
        try:
            from train_model import main as train_main
            train_main()
        except Exception as e:
            st.error(f"Failed to auto-train model: {e}")
            return None, None

    if os.path.exists(PIPELINE_PATH) and os.path.exists(METADATA_PATH):
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            metadata = joblib.load(METADATA_PATH)
            return pipeline, metadata
        except Exception as e:
            st.error(f"Failed to load model artifacts: {e}")
            return None, None
    return None, None


# ── Prediction helpers ─────────────────────────────────────────────────────────

def align_columns(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensure the uploaded DataFrame has exactly the expected feature columns
    in the right order. Missing columns are filled with a sensible default.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan   # will be handled by the encoder's unknown strategy
    return df[feature_names]


def run_predictions(pipeline, df: pd.DataFrame, feature_names: list):
    """Return predicted class and probability arrays."""
    X = align_columns(df.copy(), feature_names)
    probs = pipeline.predict_proba(X)[:, 1]
    return probs


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


# ── SHAP helpers ───────────────────────────────────────────────────────────────

def try_shap_summary(pipeline, df_features: pd.DataFrame, feature_names: list):
    """
    Attempt to generate a SHAP bar summary plot using matplotlib.
    Returns a matplotlib Figure or None if SHAP is unavailable.
    """
    try:
        import shap
        import matplotlib.pyplot as plt

        xgb_model = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(df_features)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            color=PALETTE["primary"],
        )
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ── Groq LLM summary ──────────────────────────────────────────────────────────

def try_groq_summary(high_risk_df: pd.DataFrame, metadata: dict) -> str | None:
    """
    Generate a plain-English retention summary using the Groq API.
    Returns a string or None if the API key is missing / unavailable.
    """
    try:
        from groq import Groq

        # Look for key in env var first, then Streamlit secrets
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", None)
            except Exception:
                api_key = None
        if not api_key:
            return None

        client = Groq(api_key=api_key)

        churn_count = len(high_risk_df)
        avg_prob = high_risk_df["Churn Probability"].mean() if "Churn Probability" in high_risk_df.columns else 0
        avg_tenure = high_risk_df["tenure"].mean() if "tenure" in high_risk_df.columns else "N/A"
        avg_monthly = high_risk_df["MonthlyCharges"].mean() if "MonthlyCharges" in high_risk_df.columns else "N/A"

        prompt = f"""
You are a customer retention analyst. Based on the following churn model results, write a short (3-4 sentence)
business-friendly summary that a non-technical manager can read. Focus on actionable insights.

High-risk customers detected: {churn_count}
Average churn probability: {avg_prob:.1%}
Average tenure (months): {avg_tenure:.1f if isinstance(avg_tenure, float) else avg_tenure}
Average monthly charges ($): {avg_monthly:.2f if isinstance(avg_monthly, float) else avg_monthly}

Key risk features (from model): {', '.join(metadata.get('feature_importance', {}).get('feature', [])[:5])}

Write the summary now:
"""
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


# ── Chart helpers ──────────────────────────────────────────────────────────────

def plot_churn_distribution(probs: np.ndarray, threshold: float):
    """Histogram of churn probabilities with threshold line."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=probs,
            nbinsx=30,
            marker_color=PALETTE["primary"],
            opacity=0.75,
            name="All customers",
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=PALETTE["danger"],
        line_width=2,
        annotation_text=f"Threshold = {threshold:.0%}",
        annotation_position="top right",
        annotation_font_color=PALETTE["danger"],
    )
    fig.update_layout(
        title=dict(text="Churn Probability Distribution", font=dict(size=15, color=PALETTE["text"])),
        xaxis_title="Predicted Churn Probability",
        yaxis_title="Number of Customers",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
    )
    return fig


def plot_churn_counts(labels: np.ndarray):
    """Simple donut chart of predicted churn vs. retained."""
    churn = int(labels.sum())
    retained = int(len(labels) - churn)
    fig = go.Figure(
        go.Pie(
            labels=["Retained", "At-Risk Churn"],
            values=[retained, churn],
            hole=0.58,
            marker=dict(colors=[PALETTE["success"], PALETTE["danger"]]),
            textinfo="label+percent",
            textfont=dict(size=13),
        )
    )
    fig.update_layout(
        title=dict(text="Predicted Outcome Split", font=dict(size=15, color=PALETTE["text"])),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
    )
    return fig


# ── Custom CSS ────────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .page-title {
            font-size: 30px; font-weight: 800;
            color: #1E293B; margin-bottom: 4px;
        }
        .page-sub {
            font-size: 15px; color: #64748B; margin-bottom: 28px;
        }
        .upload-box {
            background: #F8FAFC;
            border: 2px dashed #CBD5E1;
            border-radius: 12px;
            padding: 28px;
            text-align: center;
        }
        .risk-badge-high {
            background: #FEE2E2; color: #DC2626;
            padding: 2px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 600;
        }
        .risk-badge-low {
            background: #DCFCE7; color: #16A34A;
            padding: 2px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 600;
        }
        .section-header {
            font-size: 20px; font-weight: 700;
            color: #1E293B; margin: 28px 0 14px 0;
        }
        .info-box {
            background: #EEF2FF;
            border-left: 4px solid #4F46E5;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 14px;
            color: #3730A3;
            margin-bottom: 20px;
        }
        .groq-box {
            background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
            border: 1px solid #6EE7B7;
            border-radius: 12px;
            padding: 20px 24px;
            font-size: 15px;
            line-height: 1.7;
            color: #064E3B;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Main render function ───────────────────────────────────────────────────────

def render():
    inject_styles()

    st.markdown('<div class="page-title">🔮 Predict & Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a customer CSV, tune the threshold, and get instant churn predictions.</div>',
        unsafe_allow_html=True,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    pipeline, metadata = load_artifacts()

    if pipeline is None:
        st.error(
            "Model artifacts not found. Run `python train_model.py` to generate them.",
            icon="🚫",
        )
        return

    feature_names = metadata["feature_names"]

    # ── Schema helper ─────────────────────────────────────────────────────────
    with st.expander("📋 Expected CSV columns", expanded=False):
        st.markdown(
            "Your CSV must contain the following columns (extra columns are ignored):"
        )
        st.code(", ".join(feature_names), language=None)

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📁 Upload Customer Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload a telco customer dataset. The Churn column is ignored if present.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="info-box">
                ℹ️ Upload a CSV with customer data to get predictions.
                Need a sample? Download the training dataset from the
                <code>data/</code> folder and remove the <code>Churn</code> column.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Read & preview ────────────────────────────────────────────────────────
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    st.markdown('<div class="section-header">👀 Data Preview</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(raw_df):,} rows · {len(raw_df.columns)} columns**")
    st.dataframe(raw_df.head(10), use_container_width=True)

    # ── Threshold slider ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚖️ Threshold Configuration</div>', unsafe_allow_html=True)
    col_sl, col_info = st.columns([2, 1])
    with col_sl:
        threshold = st.slider(
            "Churn Probability Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
            format="%.2f",
            help=(
                "Customers with churn probability ≥ this value are flagged as 'At Risk'. "
                "Lower = more sensitive (catches more churners but more false positives)."
            ),
        )
    with col_info:
        st.markdown(
            f"""
            <div style="background:#F1F5F9; border-radius:10px; padding:16px; margin-top:4px; font-size:13px;">
                <strong>Current threshold: {threshold:.0%}</strong><br><br>
                🔴 <strong>≥ {threshold:.0%}</strong> → At Risk<br>
                🟢 <strong>&lt; {threshold:.0%}</strong> → Retained
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Run predictions ───────────────────────────────────────────────────────
    with st.spinner("Running predictions..."):
        try:
            # Clean TotalCharges just like we did during training
            df_work = raw_df.copy()
            if "TotalCharges" in df_work.columns:
                df_work["TotalCharges"] = pd.to_numeric(df_work["TotalCharges"], errors="coerce")
                df_work["TotalCharges"].fillna(df_work["TotalCharges"].median(), inplace=True)

            probs = run_predictions(pipeline, df_work, feature_names)
            labels = apply_threshold(probs, threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    # Attach results to a display DataFrame
    result_df = raw_df.copy()
    result_df["Churn Probability"] = np.round(probs, 4)
    result_df["Predicted Churn"] = labels
    result_df["Risk Level"] = result_df["Churn Probability"].apply(
        lambda p: "🔴 High Risk" if p >= threshold else "🟢 Low Risk"
    )

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Prediction Summary</div>', unsafe_allow_html=True)
    total = len(result_df)
    at_risk = int(labels.sum())
    retained = total - at_risk
    avg_prob = float(probs.mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("At-Risk Customers", f"{at_risk:,}", delta=f"{at_risk/total:.1%} of base")
    k3.metric("Retained Customers", f"{retained:,}")
    k4.metric("Avg Churn Probability", f"{avg_prob:.1%}")

    # ── Visualisations ────────────────────────────────────────────────────────
    c_dist, c_pie = st.columns(2)
    with c_dist:
        st.plotly_chart(plot_churn_distribution(probs, threshold), use_container_width=True)
    with c_pie:
        st.plotly_chart(plot_churn_counts(labels), use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Customer Predictions</div>', unsafe_allow_html=True)

    # Sort high-risk first
    display_df = result_df.sort_values("Churn Probability", ascending=False)

    # Colour-code the Risk Level column
    def highlight_risk(val):
        if "High" in str(val):
            return "background-color: #FEE2E2; color: #DC2626; font-weight: 600;"
        return "background-color: #DCFCE7; color: #16A34A; font-weight: 600;"

    styled = display_df.style.applymap(highlight_risk, subset=["Risk Level"]).format(
        {"Churn Probability": "{:.2%}"}
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # ── High-risk detail ──────────────────────────────────────────────────────
    high_risk_df = result_df[result_df["Predicted Churn"] == 1]
    if len(high_risk_df) > 0:
        with st.expander(f"🔴 View {len(high_risk_df)} High-Risk Customers", expanded=False):
            st.dataframe(
                high_risk_df.sort_values("Churn Probability", ascending=False),
                use_container_width=True,
            )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬇️ Download Results</div>', unsafe_allow_html=True)
    csv_buffer = io.BytesIO()
    result_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_buffer,
        file_name="churniq_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # ── SHAP Summary (optional) ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 SHAP Explainability (Optional)</div>', unsafe_allow_html=True)
    shap_available = True
    try:
        import shap  # noqa
    except ImportError:
        shap_available = False

    if not shap_available:
        st.info(
            "SHAP is not installed. Add `shap` to your environment to enable feature-level explanations.",
            icon="💡",
        )
    else:
        if st.button("Generate SHAP Summary (may take a few seconds)", key="shap_btn"):
            with st.spinner("Computing SHAP values..."):
                X_aligned = align_columns(df_work.copy(), feature_names)
                # Limit to first 200 rows for speed
                X_sample = X_aligned.head(200)
                fig_shap = try_shap_summary(pipeline, X_sample, feature_names)
            if fig_shap is not None:
                st.pyplot(fig_shap)
            else:
                st.warning("SHAP computation failed. The model might not support TreeExplainer.")

    # ── Groq LLM Summary (optional) ───────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 AI Retention Summary (Optional)</div>', unsafe_allow_html=True)

    # Safely check for Groq API key — env var takes priority over secrets.toml
    _groq_from_env = os.getenv("GROQ_API_KEY")
    _groq_from_secrets = None
    try:
        _groq_from_secrets = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        pass  # secrets.toml not present — that's fine
    groq_key_present = bool(_groq_from_env or _groq_from_secrets)

    if not groq_key_present:
        st.info(
            "Add your `GROQ_API_KEY` to environment variables or `.streamlit/secrets.toml` "
            "to enable AI-generated retention summaries via Groq LLaMA.",
            icon="🔑",
        )
    else:
        if st.button("Generate AI Retention Summary", key="groq_btn"):
            with st.spinner("Consulting the AI..."):
                summary = try_groq_summary(high_risk_df, metadata)
            if summary:
                st.markdown(
                    f'<div class="groq-box">💬 <strong>AI Summary</strong><br><br>{summary}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Could not generate summary. Check your Groq API key and quota.")
