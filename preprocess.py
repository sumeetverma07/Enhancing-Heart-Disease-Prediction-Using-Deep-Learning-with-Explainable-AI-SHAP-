from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

PREPROCESSING_NOTE = (
    "StandardScaler places features on a comparable scale, which is especially important for "
    "Logistic Regression and ANN optimization. SelectKBest can be enabled to keep the strongest "
    "predictors and reduce noise."
)

ENCODERS = {
    "sex": {"Female": 0, "Male": 1},
    "cp": {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3,
    },
    "fbs": {"No": 0, "Yes": 1},
    "restecg": {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2,
    },
    "exang": {"No": 0, "Yes": 1},
    "slope": {"Upsloping": 0, "Flat": 1, "Downsloping": 2},
    "thal": {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3},
}


@dataclass
class PreparedDataset:
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    selector: Optional[SelectKBest]
    selected_features: List[str]
    full_dataset: pd.DataFrame


def get_feature_schema() -> List[Dict]:
    return [
        {"name": "age", "label": "Age", "type": "slider_int", "min": 29, "max": 77, "default": 54, "help": "Patient age in years."},
        {"name": "sex", "label": "Sex", "type": "select", "options": ENCODERS["sex"], "default_index": 1, "help": "Biological sex used in the dataset."},
        {"name": "cp", "label": "Chest pain type", "type": "select", "options": ENCODERS["cp"], "default_index": 2, "help": "Categorical chest pain description."},
        {"name": "trestbps", "label": "Resting blood pressure (mm Hg)", "type": "slider_int", "min": 94, "max": 200, "default": 132, "help": "Resting systolic blood pressure."},
        {"name": "chol", "label": "Serum cholesterol (mg/dl)", "type": "slider_int", "min": 126, "max": 564, "default": 240, "help": "Total serum cholesterol."},
        {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "type": "select", "options": ENCODERS["fbs"], "default_index": 0, "help": "Binary fasting blood sugar flag."},
        {"name": "restecg", "label": "Resting ECG result", "type": "select", "options": ENCODERS["restecg"], "default_index": 0, "help": "Resting electrocardiographic result."},
        {"name": "thalach", "label": "Maximum heart rate", "type": "slider_int", "min": 71, "max": 202, "default": 150, "help": "Maximum heart rate achieved."},
        {"name": "exang", "label": "Exercise-induced angina", "type": "select", "options": ENCODERS["exang"], "default_index": 0, "help": "Angina observed during exercise."},
        {"name": "oldpeak", "label": "ST depression", "type": "slider_float", "min": 0.0, "max": 6.2, "default": 1.2, "step": 0.1, "help": "ST depression induced by exercise relative to rest."},
        {"name": "slope", "label": "Slope of peak exercise ST segment", "type": "select", "options": ENCODERS["slope"], "default_index": 1, "help": "Slope category during peak exercise."},
        {"name": "ca", "label": "Major vessels colored by fluoroscopy", "type": "slider_int", "min": 0, "max": 3, "default": 0, "help": "Number of observed major vessels."},
        {"name": "thal", "label": "Thalassemia", "type": "select", "options": ENCODERS["thal"], "default_index": 0, "help": "Thalassemia test category."},
    ]


def encode_user_input(raw_values: Dict) -> pd.DataFrame:
    encoded = {}
    for feature in FEATURE_COLUMNS:
        value = raw_values[feature]
        encoded[feature] = ENCODERS[feature][value] if feature in ENCODERS else value
    return pd.DataFrame([encoded], columns=FEATURE_COLUMNS)


def create_synthetic_heart_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(53, 9, n_samples).round().astype(int), 29, 77)
    sex = rng.choice([0, 1], n_samples, p=[0.35, 0.65])
    cp = rng.choice([0, 1, 2, 3], n_samples, p=[0.22, 0.24, 0.26, 0.28])
    trestbps = np.clip((118 + (age - 45) * 0.75 + rng.normal(0, 14, n_samples)).round(), 94, 200).astype(int)
    chol = np.clip((190 + age * 1.4 + sex * 18 + rng.normal(0, 32, n_samples)).round(), 126, 564).astype(int)
    fbs = rng.binomial(1, np.clip(0.08 + (age > 55) * 0.12 + (chol > 250) * 0.07, 0, 0.85))
    restecg = rng.choice([0, 1, 2], n_samples, p=[0.56, 0.31, 0.13])
    thalach = np.clip((203 - age + rng.normal(0, 16, n_samples)).round(), 71, 202).astype(int)
    exang = rng.binomial(1, np.clip(0.14 + (age > 57) * 0.18 + (cp == 3) * 0.11, 0, 0.9))
    oldpeak = np.clip(rng.gamma(shape=1.8, scale=0.85, size=n_samples), 0, 6.2)
    slope = rng.choice([0, 1, 2], n_samples, p=[0.42, 0.41, 0.17])
    ca = np.clip(rng.poisson(0.45 + (age > 58) * 0.5 + exang * 0.35), 0, 3)
    thal = rng.choice([1, 2, 3], n_samples, p=[0.63, 0.18, 0.19])

    logits = (
        -10.5 + 0.05 * age + 0.85 * sex + 0.55 * cp + 0.018 * trestbps + 0.007 * chol + 0.55 * fbs
        + 0.35 * restecg - 0.028 * thalach + 0.95 * exang + 0.8 * oldpeak + 0.48 * slope
        + 0.7 * ca + 0.68 * (thal == 3) + 0.35 * (thal == 2)
    )
    probability = 1 / (1 + np.exp(-logits))
    target = rng.binomial(1, np.clip(probability, 0.02, 0.98))

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak.round(2),
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target,
        }
    )


def prepare_dataset(select_k_best: Optional[int] = None, test_size: float = 0.2, random_state: int = 42) -> PreparedDataset:
    df = create_synthetic_heart_dataset(random_state=random_state)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = None
    selected_features = FEATURE_COLUMNS.copy()
    if select_k_best is not None and select_k_best < len(FEATURE_COLUMNS):
        selector = SelectKBest(score_func=f_classif, k=select_k_best)
        X_train_processed = selector.fit_transform(X_train_scaled, y_train)
        X_test_processed = selector.transform(X_test_scaled)
        selected_features = list(np.array(FEATURE_COLUMNS)[selector.get_support()])
    else:
        X_train_processed = X_train_scaled
        X_test_processed = X_test_scaled

    return PreparedDataset(
        X_train_raw=X_train.reset_index(drop=True),
        X_test_raw=X_test.reset_index(drop=True),
        X_train_processed=np.asarray(X_train_processed, dtype=np.float32),
        X_test_processed=np.asarray(X_test_processed, dtype=np.float32),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        scaler=scaler,
        selector=selector,
        selected_features=selected_features,
        full_dataset=df,
    )


def transform_patient_input(patient_df: pd.DataFrame, scaler: StandardScaler, selector: Optional[SelectKBest] = None) -> np.ndarray:
    scaled = scaler.transform(patient_df[FEATURE_COLUMNS])
    if selector is not None:
        scaled = selector.transform(scaled)
    return np.asarray(scaled, dtype=np.float32)
