
import json
import re
import string
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="CampusQuery AI | Intent Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# Supports both the expanded 5,000-row model bundle and the original project layout.
MODEL_CANDIDATES = [
    BASE / "college_query_best_model.joblib",
    BASE / "artifacts" / "college_query_best_model.joblib",
    BASE / "models" / "college_intent_model.pkl",
]
DATA_CANDIDATES = [
    BASE / "college_query_dataset_5000.csv",
    BASE / "artifacts" / "college_query_dataset_5000.csv",
    BASE / "dataset" / "college_query_dataset.csv",
]
META_CANDIDATES = [
    BASE / "model_metadata.json",
    BASE / "artifacts" / "model_metadata.json",
]
CV_CANDIDATES = [
    BASE / "model_comparison_cv.csv",
    BASE / "artifacts" / "model_comparison_cv.csv",
]
TEST_CANDIDATES = [
    BASE / "model_comparison_test.csv",
    BASE / "artifacts" / "model_comparison_test.csv",
]

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def clean_for_legacy(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)

@st.cache_data
def load_csv(path_string):
    return pd.read_csv(path_string)

MODEL_PATH = first_existing(MODEL_CANDIDATES)
DATA_PATH = first_existing(DATA_CANDIDATES)
META_PATH = first_existing(META_CANDIDATES)
CV_PATH = first_existing(CV_CANDIDATES)
TEST_PATH = first_existing(TEST_CANDIDATES)

if MODEL_PATH is None:
    st.error("No trained model found. Add college_query_best_model.joblib or the original models/college_intent_model.pkl.")
    st.stop()

model = load_model(str(MODEL_PATH))
is_advanced_model = MODEL_PATH.name == "college_query_best_model.joblib"

df = load_csv(str(DATA_PATH)) if DATA_PATH else pd.DataFrame()
metadata = {}
if META_PATH:
    try:
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        metadata = {}

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "feedback_rows" not in st.session_state:
    st.session_state.feedback_rows = []

# --------------------------- THEME ---------------------------
st.markdown("""
<style>
:root { --ink:#14213d; --muted:#65758b; --line:#e4eaf2; --blue:#315efb; }
.block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1440px; }
[data-testid="stSidebar"] { border-right: 1px solid #e8edf4; }
.hero {
  padding: 2rem 2.1rem; border-radius: 22px;
  background: linear-gradient(120deg,#101c3b 0%,#244b88 55%,#3b78c8 100%);
  color: white; margin-bottom: 1.2rem;
}
.hero h1 { font-size: 2.2rem; line-height: 1.15; margin: 0 0 .5rem 0; color:white; }
.hero p { color:#e3edff; font-size:1.02rem; margin:0; }
.eyebrow { text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; color:#b9d1ff; }
.panel {
  border:1px solid #e5eaf2; border-radius:16px; padding:1rem 1.1rem;
  background:rgba(255,255,255,.82);
}
.small-muted { color:#6b7890; font-size:.88rem; }
.result-title { font-size:1.45rem; font-weight:750; color:#17335f; }
.pill {
  display:inline-block; padding:.25rem .65rem; border-radius:999px;
  background:#eaf1ff; color:#244e9a; font-size:.78rem; font-weight:700;
}
div[data-testid="stMetric"] {
  border:1px solid #e7edf5; padding: .8rem 1rem; border-radius:14px;
  background:#fff;
}
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:650; }
hr { border-color:#e7edf5; }
</style>
""", unsafe_allow_html=True)

# --------------------------- HELPERS ---------------------------
def get_prediction(text):
    # The expanded notebook model was trained on raw strings through its own TF-IDF pipeline.
    # The legacy project model was previously fed cleaned text.
    model_input = str(text) if is_advanced_model else clean_for_legacy(text)
    predicted = model.predict([model_input])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([model_input])[0]
        classes = list(model.classes_)
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        confidence = float(max(probs))
    else:
        classes = list(getattr(model, "classes_", []))
        pairs = [(predicted, 1.0)]
        confidence = 1.0

    return predicted, confidence, pairs

def record_prediction(query, predicted, confidence):
    st.session_state.query_history.insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Query": query,
        "Predicted category": predicted,
        "Confidence (%)": round(confidence * 100, 2),
    })

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def read_optional(path):
    return load_csv(str(path)) if path else pd.DataFrame()

# --------------------------- SIDEBAR ---------------------------
st.sidebar.markdown("## 🎓 CampusQuery AI")
st.sidebar.caption("College Query Intent Intelligence")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "AI Query Lab", "Batch Classifier", "Analytics", "Model Benchmark", "Feedback Studio", "Project Info"],
)
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Model loaded")
st.sidebar.caption(f"Model file: `{MODEL_PATH.name}`")
st.sidebar.caption("Advanced 10-class model" if is_advanced_model else "Legacy 6-class model")
if metadata.get("best_model_name"):
    st.sidebar.caption(f"Selected: {metadata['best_model_name']}")

# --------------------------- OVERVIEW ---------------------------
if page == "Overview":
    category_count = int(df["intent"].nunique()) if not df.empty and "intent" in df else len(getattr(model, "classes_", []))
    row_count = len(df) if not df.empty else 0
    model_label = metadata.get("best_model_name", "Logistic Regression" if not is_advanced_model else "Selected ML model")

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">NLP · MACHINE LEARNING · STUDENT SERVICES</div>
      <h1>CampusQuery AI</h1>
      <p>Turn unstructured student questions into clear, actionable intent categories.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{row_count:,}" if row_count else "Not loaded")
    c2.metric("Intent categories", category_count)
    c3.metric("Selected model", model_label)
    c4.metric("Prediction mode", "Calibrated" if is_advanced_model else "Legacy")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Explore the workspace")
        st.markdown("""
        - **AI Query Lab:** classify one question and inspect the top predicted categories.
        - **Batch Classifier:** upload a CSV and classify many student questions at once.
        - **Analytics:** inspect category distribution and query activity.
        - **Model Benchmark:** compare cross-validation and holdout metrics.
        - **Feedback Studio:** review predictions and export corrections for future training.
        """)
    with right:
        st.markdown("### Quick start")
        sample = st.selectbox("Try a sample question", [
            "How can I check my attendance percentage?",
            "What is the last date to pay semester fees?",
            "When will the placement drive take place?",
            "How do I renew a library book?",
            "Where can I find the college bus route?",
        ])
        if st.button("Classify sample question", use_container_width=True):
            pred, conf, pairs = get_prediction(sample)
            record_prediction(sample, pred, conf)
            st.session_state.quick_result = (sample, pred, conf, pairs)
            st.success(f"{pred} · {conf*100:.1f}% confidence")
        if "quick_result" in st.session_state:
            q, pred, conf, pairs = st.session_state.quick_result
            st.caption(f"Latest sample: {q}")
            st.markdown(f"**Prediction:** {pred}  ·  **Confidence:** {conf*100:.1f}%")

    if not df.empty and "intent" in df:
        st.markdown("### Supported intent categories")
        counts = df["intent"].value_counts()
        cols = st.columns(min(5, len(counts)))
        for i, (cat, count) in enumerate(counts.items()):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="panel"><div class="pill">{cat}</div><h3>{count:,}</h3><div class="small-muted">dataset examples</div></div>', unsafe_allow_html=True)

# --------------------------- SINGLE QUERY ---------------------------
elif page == "AI Query Lab":
    st.title("🤖 AI Query Lab")
    st.caption("Classify a student question, inspect alternative categories, and flag uncertain predictions.")
    query = st.text_area("Enter a college-related question", height=130, placeholder="Example: Can I apply for a merit-based scholarship?")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        threshold = st.slider("Low-confidence threshold", 0.30, 0.90, 0.50, 0.05)
    with col_b:
        st.caption("Predictions below this threshold are flagged for human review. Confidence is not a guarantee of correctness.")

    if st.button("Analyze query", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            pred, conf, pairs = get_prediction(query.strip())
            record_prediction(query.strip(), pred, conf)
            st.session_state.latest_prediction = {
                "query": query.strip(), "pred": pred, "conf": conf, "pairs": pairs
            }

    if "latest_prediction" in st.session_state:
        result = st.session_state.latest_prediction
        st.divider()
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("#### Predicted intent")
            st.markdown(f'<div class="panel"><div class="result-title">{result["pred"]}</div><p class="small-muted">Top predicted category</p></div>', unsafe_allow_html=True)
        with r2:
            st.metric("Confidence estimate", f'{result["conf"]*100:.1f}%')
            if result["conf"] < threshold:
                st.warning("Low confidence — verify this prediction or collect a labelled example.")
            else:
                st.success("Prediction is above the selected review threshold.")

        st.markdown("#### Top category predictions")
        top = pd.DataFrame(result["pairs"][:5], columns=["Category", "Probability"])
        top["Confidence (%)"] = (top["Probability"] * 100).round(2)
        st.bar_chart(top.set_index("Category")["Confidence (%)"])
        st.dataframe(top[["Category", "Confidence (%)"]], use_container_width=True, hide_index=True)

        with st.expander("Give feedback on this prediction"):
            options = sorted(list(getattr(model, "classes_", [])))
            correct = st.selectbox("What is the correct category?", options, key="single_feedback_category")
            note = st.text_input("Optional note", key="single_feedback_note")
            if st.button("Save feedback"):
                st.session_state.feedback_rows.append({
                    "Timestamp": datetime.now().isoformat(timespec="seconds"),
                    "Query": result["query"],
                    "Predicted category": result["pred"],
                    "Correct category": correct,
                    "Confidence (%)": round(result["conf"]*100, 2),
                    "Note": note,
                })
                st.success("Feedback saved in this session. Download it from Feedback Studio.")

# --------------------------- BATCH CLASSIFICATION ---------------------------
elif page == "Batch Classifier":
    st.title("📦 Batch Query Classifier")
    st.write("Upload a CSV containing a column of student questions. The app will add predicted intent and confidence.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        st.write("Uploaded preview")
        st.dataframe(batch.head(10), use_container_width=True)
        text_col = st.selectbox("Select the question/text column", batch.columns.tolist())
        if st.button("Classify all rows", type="primary"):
            questions = batch[text_col].fillna("").astype(str)
            outputs = []
            for q in questions:
                if not q.strip():
                    outputs.append({"Predicted category": "", "Confidence (%)": 0.0, "Top 3 categories": ""})
                    continue
                pred, conf, pairs = get_prediction(q)
                outputs.append({
                    "Predicted category": pred,
                    "Confidence (%)": round(conf*100, 2),
                    "Top 3 categories": " | ".join([f"{c}: {p*100:.1f}%" for c,p in pairs[:3]])
                })
            result_df = pd.concat([batch.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
            st.session_state.batch_result = result_df
            st.success(f"Classified {len(result_df):,} rows.")
    if "batch_result" in st.session_state:
        st.dataframe(st.session_state.batch_result, use_container_width=True)
        st.download_button(
            "Download classified CSV",
            st.session_state.batch_result.to_csv(index=False).encode("utf-8"),
            "college_query_predictions.csv",
            "text/csv",
            use_container_width=True,
        )

# --------------------------- ANALYTICS ---------------------------
elif page == "Analytics":
    st.title("📊 Query & Dataset Analytics")
    if df.empty:
        st.info("Dataset CSV not found. Add college_query_dataset_5000.csv or the original dataset file.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total dataset records", f"{len(df):,}")
        b.metric("Unique categories", df["intent"].nunique())
        if "source" in df:
            c.metric("Synthetic records", f'{(df["source"]=="synthetic").sum():,}')
        else:
            c.metric("Unique questions", f'{df["text"].nunique():,}')

        st.subheader("Intent distribution")
        st.bar_chart(df["intent"].value_counts())
        if {"intent", "source"}.issubset(df.columns):
            st.subheader("Original vs synthetic data")
            st.bar_chart(df.groupby(["intent", "source"]).size().unstack(fill_value=0))
        if "text" in df:
            st.subheader("Search the dataset")
            term = st.text_input("Find a keyword in sample questions")
            if term:
                matches = df[df["text"].astype(str).str.contains(term, case=False, na=False)]
                st.dataframe(matches[["text", "intent"]].head(100), use_container_width=True)
            st.download_button("Download training dataset", df.to_csv(index=False).encode("utf-8"), "college_query_dataset.csv", "text/csv")

    if st.session_state.query_history:
        st.subheader("This-session query activity")
        hist = pd.DataFrame(st.session_state.query_history)
        st.metric("Queries classified in this session", len(hist))
        st.dataframe(hist.head(30), use_container_width=True)
        st.download_button("Export query history", hist.to_csv(index=False).encode("utf-8"), "query_history.csv", "text/csv")

# --------------------------- MODEL BENCHMARK ---------------------------
elif page == "Model Benchmark":
    st.title("🏆 Model Benchmark")
    cv = read_optional(CV_PATH)
    test = read_optional(TEST_PATH)
    if not cv.empty:
        st.subheader("Cross-validation comparison")
        st.dataframe(cv, use_container_width=True, hide_index=True)
        metric_cols = [c for c in ["CV Accuracy", "CV F1 (macro)", "CV Precision (macro)", "CV Recall (macro)"] if c in cv.columns]
        if metric_cols:
            st.bar_chart(cv.set_index("Model")[metric_cols])
    if not test.empty:
        st.subheader("Held-out test comparison")
        st.dataframe(test, use_container_width=True, hide_index=True)
        metric_cols = [c for c in ["Test Accuracy", "Test F1 (macro)", "Test Precision (macro)", "Test Recall (macro)"] if c in test.columns]
        if metric_cols:
            st.bar_chart(test.set_index("Model")[metric_cols])
    if not cv.empty or not test.empty:
        st.caption("Choose the final model using cross-validation results. Keep the held-out test set untouched until final evaluation.")
    else:
        st.info("Model comparison files not found. Run the 5,000-row Colab training notebook and upload model_comparison_cv.csv and model_comparison_test.csv beside app.py.")
    if metadata:
        st.subheader("Selected model metadata")
        st.json(metadata)

# --------------------------- FEEDBACK ---------------------------
elif page == "Feedback Studio":
    st.title("📝 Feedback Studio")
    st.write("Review model mistakes and export corrected labels. Feedback is kept in the current app session unless you download it.")
    if st.session_state.feedback_rows:
        feedback = pd.DataFrame(st.session_state.feedback_rows)
        st.dataframe(feedback, use_container_width=True)
        st.download_button("Download labelled feedback CSV", feedback.to_csv(index=False).encode("utf-8"), "college_query_feedback.csv", "text/csv")
    else:
        st.info("No feedback recorded in this session yet. Use the feedback form in AI Query Lab.")
    st.warning("Feedback is not automatically added to the training dataset. Review labels before retraining.")

# --------------------------- PROJECT INFO ---------------------------
elif page == "Project Info":
    st.title("📚 Project & Technical Details")
    st.markdown("""
    **Problem:** College questions arrive in free-text form and need routing to the right student-service category.

    **NLP pipeline:** Text input → TF-IDF feature extraction → trained classifier → predicted intent and confidence estimate.

    **Expanded categories:** Admission, Fees, Exam, Hostel, Scholarship, Timetable, Library, Transport, Placement and Attendance.

    **Models compared in the training notebook:** Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression, Linear SVM, SGD Classifier and Random Forest.

    **Application features:** Single-query prediction, top category probabilities, low-confidence review flag, batch CSV classification, dataset analytics, model benchmark, query history and feedback export.
    """)
    st.info("The app classifies intent. It does not generate official college answers or verify university policies.")
    st.caption("Model file: " + str(MODEL_PATH))
    if metadata:
        st.json(metadata)

st.divider()
st.caption("CampusQuery AI · Academic prototype · Validate predictions before operational use.")
