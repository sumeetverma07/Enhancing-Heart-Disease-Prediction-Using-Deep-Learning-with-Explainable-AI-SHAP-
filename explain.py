from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def explain_prediction(artifacts, model_name: str, patient_df: pd.DataFrame) -> Dict:
    bundle = artifacts["bundle"]
    processed_patient = artifacts["transform_input"](patient_df)
    feature_names = bundle["selected_features"]
    background = bundle["background_processed"]
    summary_data = bundle["explain_processed"]

    if model_name == "Random Forest":
        model = bundle["models"]["Random Forest"]
        explainer = shap.TreeExplainer(model)
        shap_summary_values = explainer.shap_values(summary_data)
        shap_patient_values = explainer.shap_values(processed_patient)
        if isinstance(shap_summary_values, list):
            summary_values = shap_summary_values[1]
            patient_values = shap_patient_values[1][0]
            expected_value = explainer.expected_value[1]
        elif getattr(shap_summary_values, "ndim", 0) == 3:
            summary_values = shap_summary_values[:, :, 1]
            patient_values = shap_patient_values[0, :, 1]
            expected_value = explainer.expected_value[1]
        else:
            summary_values = shap_summary_values
            patient_values = shap_patient_values[0]
            expected_value = explainer.expected_value
    else:
        model = artifacts["ann_model"]

        def predict_fn(values):
            return model.predict(values, verbose=0).reshape(-1)

        explainer = shap.KernelExplainer(predict_fn, background)
        summary_values = explainer.shap_values(summary_data, nsamples=100)
        patient_values = explainer.shap_values(processed_patient, nsamples=100)
        if isinstance(summary_values, list):
            summary_values = summary_values[0]
        if isinstance(patient_values, list):
            patient_values = patient_values[0]
        summary_values = np.asarray(summary_values)
        patient_values = np.asarray(patient_values)[0]
        expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    explanation = shap.Explanation(
        values=patient_values,
        base_values=expected_value,
        data=processed_patient[0],
        feature_names=feature_names,
    )

    patient_impacts = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Patient value": processed_patient[0],
                "SHAP value": patient_values,
            }
        )
        .assign(abs_value=lambda frame: frame["SHAP value"].abs())
        .sort_values("abs_value", ascending=False)
        .drop(columns="abs_value")
    )

    return {
        "summary_values": summary_values,
        "summary_data": summary_data,
        "feature_names": feature_names,
        "explanation": explanation,
        "patient_impacts": patient_impacts,
    }


def plot_shap_summary(shap_bundle: Dict):
    plt.figure(figsize=(8, 4.6))
    shap.summary_plot(
        shap_bundle["summary_values"],
        shap_bundle["summary_data"],
        feature_names=shap_bundle["feature_names"],
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_waterfall(shap_bundle: Dict):
    plt.figure(figsize=(8, 4.8))
    shap.plots.waterfall(shap_bundle["explanation"], max_display=10, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
