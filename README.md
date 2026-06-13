# ChurnIQ — Customer Churn Intelligence

> A polished, 2-page Streamlit app that predicts telecom customer churn using XGBoost — with SHAP explainability and optional Groq LLM summaries.

---

## 🗂️ Project Structure

```
ChurnIQ/
├── app.py                    # Main Streamlit entry point
├── train_model.py            # One-time model training script
├── requirements.txt
├── README.md
│
├── data/
│   └── telco_customer_churn.csv   # Source dataset (IBM Telco)
│
├── artifacts/                # Auto-created after training
│   ├── pipeline.joblib       # Full sklearn Pipeline (preprocessor + XGBoost)
│   └── metadata.joblib       # Metrics, feature names, ROC data, importances
│
├── pages/
│   ├── home.py               # Page 1 — Model overview & performance
│   └── predict.py            # Page 2 — Upload CSV & predict churn
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py      # Shared data cleaning & feature utilities
│
└── .streamlit/
    ├── config.toml           # Light theme + brand colours
    └── secrets.toml          # API keys (never commit this)
```

---

## 📋 What ChurnIQ Does

ChurnIQ is a production-style ML application for customer churn prediction in the telecommunications domain. It lets you:

- **Understand your model** — metrics, confusion matrix, ROC curve, and feature importances presented on a clean dashboard.
- **Score new customers** — upload any CSV with the right schema and get per-row churn probabilities instantly.
- **Tune for your business** — slide the decision threshold to balance false positives vs. false negatives depending on your retention budget.
- **Explain predictions** — SHAP values show which features drove each prediction.
- **Summarise in plain English** — Groq LLaMA integration generates a one-paragraph business-friendly retention brief.

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run once to produce the `artifacts/` directory:

```bash
python train_model.py
```

You should see output like:

```
[1/5] Loading data ...
      Dataset shape: (7032, 21)  |  Churn rate: 26.54%
...
[5/5] Saving artifacts ...
  Saved pipeline  → artifacts/pipeline.joblib
  Saved metadata  → artifacts/metadata.joblib

  Done! Run the app with:  streamlit run app.py
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📄 Pages

### Page 1 — Home & Model Overview

- Hero banner with project description
- Live KPI tiles: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix heatmap
- ROC curve with AUC score
- Top-15 feature importance chart
- Dataset & training details
- "How it works" explainer
- GitHub / Email / LinkedIn footer

### Page 2 — Predict & Upload

- CSV file uploader with schema validator
- Data preview (first 10 rows)
- Probability threshold slider (0.10 → 0.90)
- Batch predictions: churn probability + risk label per row
- Summary KPIs: total, at-risk, retained, average probability
- Churn distribution histogram + donut chart
- Colour-coded results table (red = high risk, green = low risk)
- High-risk customer detail expander
- Download predictions as CSV
- Optional: SHAP feature importance summary
- Optional: Groq LLM retention narrative

---

## 🔑 Adding the Groq API Key

The Groq integration is **completely optional** — the app works fully without it.

### Option A — Environment Variable (local / any server)

```bash
# Linux / macOS
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

### Option B — Streamlit Secrets (recommended for Streamlit Community Cloud)

Edit `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

> ⚠️ **Never commit `secrets.toml` to Git.** It is already listed in `.gitignore`.

On **Streamlit Community Cloud**, go to your app → **Settings → Secrets** and paste:

```
GROQ_API_KEY = "gsk_your_key_here"
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and entry point `app.py`.
4. Add your `GROQ_API_KEY` in the **Secrets** tab (optional).
5. Click **Deploy**.

> **Note:** The `artifacts/` directory must be committed to Git, or the training step must be run as a startup script. The simplest approach is to commit the trained artifacts.

---

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features.

| Column | Type | Description |
|---|---|---|
| `tenure` | numeric | Months with the company |
| `MonthlyCharges` | numeric | Current monthly bill |
| `TotalCharges` | numeric | Total spend to date |
| `Contract` | categorical | Month-to-month / One year / Two year |
| `InternetService` | categorical | DSL / Fiber optic / None |
| `PaymentMethod` | categorical | Electronic check / Mailed check / etc. |
| `Churn` | target | Yes / No |

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| ML model | XGBoost |
| Preprocessing | scikit-learn ColumnTransformer |
| Explainability | SHAP (optional) |
| AI summaries | Groq LLaMA (optional) |
| App framework | Streamlit |
| Charts | Plotly |
| Serialisation | joblib |

---

## 📬 Contact

- **GitHub**: [github.com/Santosh-vihar/ChurnIQ](https://github.com/Santosh-vihar/ChurnIQ)
- **Email**: your@email.com
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
