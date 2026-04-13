import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from explain import explain_prediction, plot_shap_summary, plot_shap_waterfall
from model import load_project_artifacts, train_and_save_pipeline
from preprocess import FEATURE_COLUMNS, PREPROCESSING_NOTE, encode_user_input, get_feature_schema


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "project_bundle.pkl"


st.set_page_config(
    page_title="Enhancing Heart Disease Prediction Using Deep Learning with Explainable AI",
    page_icon="Heart",
    layout="wide",
)


st.markdown(
    """
    <style>
        :root {
            --page-text: #0f172a;
            --muted-text: #475569;
            --card-bg: rgba(255, 255, 255, 0.96);
            --card-border: rgba(15, 23, 42, 0.08);
            --accent: #1f4e79;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 76, 129, 0.12), transparent 36%),
                radial-gradient(circle at top right, rgba(203, 67, 53, 0.12), transparent 28%),
                linear-gradient(180deg, #f5f7fb 0%, #ecf1f7 100%);
            color: var(--page-text);
        }
        .main-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
            color: var(--page-text);
        }
        .hero {
            background: linear-gradient(135deg, #102542 0%, #1f4e79 55%, #cf5c36 100%);
            color: white;
            border-radius: 28px;
            padding: 1.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 24px 48px rgba(16, 37, 66, 0.22);
        }
        .metric-box {
            background: white;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            border: 1px solid var(--card-border);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        }
        .metric-box [data-testid="stMetricLabel"],
        .metric-box [data-testid="stMetricValue"],
        .metric-box [data-testid="stMetricDelta"],
        .metric-box .stCaptionContainer {
            color: var(--page-text);
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp div, .stApp span {
            color: inherit;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.98);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: var(--page-text);
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stAlert,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--page-text) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] [data-baseweb="textarea"] {
            background: #ffffff;
            color: var(--page-text) !important;
            border-radius: 10px;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select {
            color: var(--page-text) !important;
            -webkit-text-fill-color: var(--page-text) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] .stSlider svg {
            fill: var(--page-text) !important;
            color: var(--page-text) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #e2e8f0 !important;
            color: var(--page-text) !important;
        }
        [data-testid="stSidebar"] .stInfo {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
        }
        [data-testid="stTabs"] button {
            color: var(--muted-text);
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: white;
            border-radius: 14px;
        }
        .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            color: var(--page-text);
        }
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label {
            color: var(--page-text);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def ensure_pipeline():
    if not BUNDLE_PATH.exists():
        train_and_save_pipeline(project_dir=PROJECT_DIR)
    return load_project_artifacts(project_dir=PROJECT_DIR)


@st.cache_resource(show_spinner=False)
def rebuild_pipeline():
    train_and_save_pipeline(project_dir=PROJECT_DIR, force_retrain=True)
    return load_project_artifacts(project_dir=PROJECT_DIR)


def render_sidebar():
    st.sidebar.header("Patient Input")
    schema = get_feature_schema()
    user_values = {}
    for field in schema:
        key = field["name"]
        if field["type"] == "slider_int":
            user_values[key] = st.sidebar.slider(
                field["label"],
                min_value=field["min"],
                max_value=field["max"],
                value=field["default"],
                help=field["help"],
            )
        elif field["type"] == "slider_float":
            user_values[key] = st.sidebar.slider(
                field["label"],
                min_value=float(field["min"]),
                max_value=float(field["max"]),
                value=float(field["default"]),
                step=float(field["step"]),
                help=field["help"],
            )
        else:
            user_values[key] = st.sidebar.selectbox(
                field["label"],
                options=list(field["options"].keys()),
                index=field["default_index"],
                help=field["help"],
            )

    st.sidebar.markdown("---")
    st.sidebar.info(PREPROCESSING_NOTE)
    return user_values


def render_prediction_cards(prediction_table: pd.DataFrame):
    cols = st.columns(len(prediction_table))
    for idx, (_, row) in enumerate(prediction_table.iterrows()):
        with cols[idx]:
            st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
            st.metric(row["Model"], f"{row['Heart Disease Probability (%)']:.1f}%", row["Predicted Class"])
            st.caption(f"Confidence: {row['Confidence (%)']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)


def plot_model_comparison(comparison_df: pd.DataFrame):
    long_df = comparison_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax, palette="crest")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Comparison Across Evaluation Metrics")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def build_prediction_table(probabilities):
    rows = []
    for model_name, positive_probability in probabilities.items():
        rows.append(
            {
                "Model": model_name,
                "Heart Disease Probability (%)": positive_probability * 100,
                "Predicted Class": "High risk" if positive_probability >= 0.5 else "Low risk",
                "Confidence (%)": max(positive_probability, 1 - positive_probability) * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("Heart Disease Probability (%)", ascending=False)


def main():
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin-bottom:0.35rem;">Enhancing Heart Disease Prediction Using Deep Learning with Explainable AI (SHAP)</h1>
            <p style="font-size:1.05rem; margin-bottom:0;">
                Research-style comparison of Logistic Regression, Random Forest, and a TensorFlow/Keras ANN.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 2.1], gap="large")
    with left:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Clinical Context")
        st.write(
            "This project compares classical machine learning and deep learning models for binary heart disease prediction. "
            "Recall is highlighted because missing a true-risk patient is usually more costly than a false positive."
        )
        st.write(
            "The pipeline scales features before Logistic Regression and ANN training, and it supports SelectKBest to reduce noise."
        )
        retrain = st.button("Retrain Full Pipeline", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Patient Form")
        raw_input = render_sidebar()
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if retrain:
        try:
            with st.spinner("Training all models and saving fresh artifacts..."):
                artifacts = rebuild_pipeline()
            st.success("Pipeline retrained successfully.")
        except RuntimeError as exc:
            st.error(str(exc))
            st.info(
                "TensorFlow is not available in this Windows environment. Run the full ANN pipeline inside the Ubuntu Vagrant VM "
                "or another Linux environment after installing requirements there."
            )
            st.stop()
    else:
        try:
            with st.spinner("Loading trained artifacts..."):
                artifacts = ensure_pipeline()
        except RuntimeError as exc:
            st.error(str(exc))
            st.info(
                "TensorFlow is required for the ANN portion of this project. Install it in the Ubuntu Vagrant VM or a Linux environment, "
                "then retrain with `python train_and_save_model.py`."
            )
            st.stop()

    encoded_input = encode_user_input(raw_input)
    prediction_probs = artifacts["predict_patient"](encoded_input)
    prediction_table = build_prediction_table(prediction_probs)
    comparison_df = artifacts["comparison_df"]
    ann_available = artifacts["ann_model"] is not None

    st.markdown("## Predictions")
    render_prediction_cards(prediction_table)
    if not ann_available:
        st.warning(
            "ANN/TensorFlow is not installed in this environment yet, so the UI is currently serving the Logistic Regression and Random Forest parts of the project."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model Comparison", "Selected Prediction", "SHAP Explanations", "Research Notes"]
    )

    with tab1:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.subheader("Evaluation Results")
        st.dataframe(comparison_df, use_container_width=True)
        plot_model_comparison(comparison_df)
        top_recall = comparison_df.sort_values("Recall", ascending=False).iloc[0]
        st.success(f"Highest recall: {top_recall['Model']} ({top_recall['Recall']:.3f})")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        available_models = ["Logistic Regression", "Random Forest"] + (["ANN"] if ann_available else [])
        model_choice = st.selectbox("Choose a model to inspect", available_models, index=1 if len(available_models) > 1 else 0)
        selected_probability = prediction_table.loc[
            prediction_table["Model"] == model_choice, "Heart Disease Probability (%)"
        ].iloc[0]
        st.metric("Heart disease probability", f"{selected_probability:.1f}%")
        st.caption("Predictions are probabilities, not medical diagnoses.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        shap_options = ["Random Forest"] + (["ANN"] if ann_available else [])
        shap_model = st.selectbox("Select model for SHAP explanation", shap_options, index=0)
        shap_bundle = explain_prediction(artifacts, shap_model, encoded_input)
        st.write(
            "Red SHAP values increase predicted risk. Blue SHAP values decrease predicted risk. "
            "The waterfall plot explains the current patient and the summary plot shows feature behavior across the held-out sample."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_shap_summary(shap_bundle), clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(plot_shap_waterfall(shap_bundle), clear_figure=True, use_container_width=True)
        impact_df = shap_bundle["patient_impacts"].copy()
        impact_df["Direction"] = impact_df["SHAP value"].apply(
            lambda value: "Increases risk" if value > 0 else "Decreases risk"
        )
        st.dataframe(impact_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown(
            """
            - `StandardScaler` is necessary because Logistic Regression and ANN models are sensitive to feature magnitude.
            - `SelectKBest` is included as an optional feature engineering step for simpler and cleaner experiments.
            - `EarlyStopping` restores the best ANN weights and helps prevent overfitting.
            - The ANN uses Dense 128 -> 64 -> 32 with ReLU, Batch Normalization, Dropout, Adam, and binary crossentropy.
            - Metrics include Accuracy, Precision, Recall, F1-score, and ROC-AUC for a complete academic comparison.
            """
        )
        st.caption(f"Pipeline feature order: {', '.join(FEATURE_COLUMNS)}")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
