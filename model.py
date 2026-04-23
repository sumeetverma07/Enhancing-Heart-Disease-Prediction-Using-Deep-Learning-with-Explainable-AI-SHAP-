from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ann_runtime import PortableANNModel
from evaluate import compare_models, compute_classification_metrics
from preprocess import FEATURE_COLUMNS, PREPROCESSING_NOTE, prepare_dataset, transform_patient_input

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except ImportError:
    tf = None
    Sequential = None
    EarlyStopping = None
    BatchNormalization = None
    Dense = None
    Dropout = None
    Input = None
    load_model = None
    Adam = None
    TF_AVAILABLE = False


ARTIFACT_DIRNAME = "artifacts"
PROJECT_BUNDLE_NAME = "project_bundle.pkl"
ANN_MODEL_NAME = "ann_model.keras"
COMPARISON_CSV_NAME = "model_comparison.csv"
DATASET_CSV_NAME = "heart_disease_research_dataset.csv"


def _artifact_paths(project_dir: Path) -> Dict[str, Path]:
    artifact_dir = project_dir / ARTIFACT_DIRNAME
    return {
        "artifact_dir": artifact_dir,
        "bundle": artifact_dir / PROJECT_BUNDLE_NAME,
        "ann_model": artifact_dir / ANN_MODEL_NAME,
        "comparison_csv": artifact_dir / COMPARISON_CSV_NAME,
        "dataset_csv": artifact_dir / DATASET_CSV_NAME,
    }


def build_ann(input_dim: int, learning_rate: float = 0.001, dropout_rate: float = 0.3):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt to train the ANN.")

    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate / 2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _train_ann_with_search(X_train, y_train, X_test, y_test, random_state: int = 42):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt to train the ANN.")

    tf.keras.utils.set_random_seed(random_state)
    search_space = [
        {"learning_rate": 0.001, "batch_size": 16, "epochs": 50},
        {"learning_rate": 0.001, "batch_size": 32, "epochs": 70},
        {"learning_rate": 0.0005, "batch_size": 32, "epochs": 90},
    ]

    best_model = None
    best_config = None
    best_history = None
    best_score = (-1.0, -1.0)

    for config in search_space:
        model = build_ann(input_dim=X_train.shape[1], learning_rate=config["learning_rate"])
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)],
            verbose=0,
        )
        probabilities = model.predict(X_test, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_test, predictions, probabilities)
        score = (metrics["Recall"], metrics["ROC-AUC"])
        if score > best_score:
            best_score = score
            best_model = model
            best_config = config
            best_history = history.history

    return best_model, best_config, best_history


def _build_prediction_function(bundle, ann_model):
    def predict_patient(patient_df):
        processed = transform_patient_input(patient_df, bundle["scaler"], bundle["selector"])
        logistic_prob = bundle["models"]["Logistic Regression"].predict_proba(processed)[0, 1]
        forest_prob = bundle["models"]["Random Forest"].predict_proba(processed)[0, 1]
        result = {
            "Logistic Regression": float(logistic_prob),
            "Random Forest": float(forest_prob),
        }
        if ann_model is not None:
            ann_prob = ann_model.predict(processed, verbose=0).reshape(-1)[0]
            result["ANN"] = float(ann_prob)
        return result

    return predict_patient


def _normalize_runtime_bundle(bundle):
    """Force estimators into a Windows-safe runtime configuration."""
    forest_model = bundle.get("models", {}).get("Random Forest")
    if forest_model is not None and hasattr(forest_model, "n_jobs"):
        forest_model.n_jobs = 1
    return bundle


def train_and_save_pipeline(project_dir: Optional[Path] = None, force_retrain: bool = False):
    project_dir = Path(project_dir or Path(__file__).resolve().parent)
    paths = _artifact_paths(project_dir)
    paths["artifact_dir"].mkdir(exist_ok=True)

    if not force_retrain and paths["bundle"].exists() and paths["ann_model"].exists():
        return load_project_artifacts(project_dir)

    dataset = prepare_dataset(select_k_best=10, random_state=42)

    logistic_regression = LogisticRegression(max_iter=1500, random_state=42)
    logistic_regression.fit(dataset.X_train_processed, dataset.y_train)

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    random_forest.fit(dataset.X_train_processed, dataset.y_train)

    ann_model, ann_config, ann_history = _train_ann_with_search(
        dataset.X_train_processed,
        dataset.y_train,
        dataset.X_test_processed,
        dataset.y_test,
    )

    lr_prob = logistic_regression.predict_proba(dataset.X_test_processed)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    rf_prob = random_forest.predict_proba(dataset.X_test_processed)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    ann_prob = ann_model.predict(dataset.X_test_processed, verbose=0).reshape(-1)
    ann_pred = (ann_prob >= 0.5).astype(int)

    metric_results = {
        "Logistic Regression": compute_classification_metrics(dataset.y_test, lr_pred, lr_prob),
        "Random Forest": compute_classification_metrics(dataset.y_test, rf_pred, rf_prob),
        "ANN": compute_classification_metrics(dataset.y_test, ann_pred, ann_prob),
    }
    comparison_df = compare_models(metric_results)

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "selected_features": dataset.selected_features,
        "scaler": dataset.scaler,
        "selector": dataset.selector,
        "models": {
            "Logistic Regression": logistic_regression,
            "Random Forest": random_forest,
        },
        "comparison_df": comparison_df,
        "preprocessing_note": PREPROCESSING_NOTE,
        "ann_config": ann_config,
        "ann_history": ann_history,
        "background_processed": dataset.X_train_processed[:60],
        "explain_processed": dataset.X_test_processed[:80],
        "test_processed": dataset.X_test_processed,
        "test_raw": dataset.X_test_raw,
        "y_test": dataset.y_test,
    }

    joblib.dump(bundle, paths["bundle"])
    ann_model.save(paths["ann_model"])
    comparison_df.to_csv(paths["comparison_csv"], index=False)
    dataset.full_dataset.to_csv(paths["dataset_csv"], index=False)

    return {
        "bundle": bundle,
        "ann_model": ann_model,
        "comparison_df": comparison_df,
        "predict_patient": _build_prediction_function(bundle, ann_model),
        "transform_input": lambda patient_df: transform_patient_input(patient_df, bundle["scaler"], bundle["selector"]),
    }


def load_project_artifacts(project_dir: Optional[Path] = None):
    project_dir = Path(project_dir or Path(__file__).resolve().parent)
    paths = _artifact_paths(project_dir)
    if not paths["bundle"].exists():
        raise RuntimeError("Project artifacts are missing. Run train_and_save_model.py after installing dependencies.")
    bundle = joblib.load(paths["bundle"])
    bundle = _normalize_runtime_bundle(bundle)
    comparison_df = pd.read_csv(paths["comparison_csv"]) if paths["comparison_csv"].exists() else bundle["comparison_df"]
    ann_model = None
    if TF_AVAILABLE and paths["ann_model"].exists():
        ann_model = load_model(paths["ann_model"])
    elif paths["ann_model"].exists():
        ann_model = PortableANNModel.load(paths["ann_model"])

    return {
        "bundle": bundle,
        "ann_model": ann_model,
        "comparison_df": comparison_df,
        "predict_patient": _build_prediction_function(bundle, ann_model),
        "transform_input": lambda patient_df: transform_patient_input(patient_df, bundle["scaler"], bundle["selector"]),
    }
