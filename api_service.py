from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap

from model import load_project_artifacts
from preprocess import encode_user_input, get_feature_schema

PROJECT_DIR = Path(__file__).resolve().parent


class ValidationError(Exception):
    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        super().__init__("Invalid patient input.")


@dataclass
class PredictionBundle:
    raw_input: Dict[str, object]
    encoded_input: pd.DataFrame
    probabilities: Dict[str, float]
    primary_model: str
    primary_probability: float
    primary_label: str
    confidence: float
    explanation_model: str
    shap_values: List[Dict[str, float | str]]
    feature_importance: List[Dict[str, float | str]]
    ann_visualization: Dict | None
    timestamp: str
    inference_time_ms: float


@lru_cache(maxsize=1)
def get_artifacts():
    return load_project_artifacts(project_dir=PROJECT_DIR)


@lru_cache(maxsize=1)
def get_schema() -> Tuple[Dict, ...]:
    return tuple(get_feature_schema())


@lru_cache(maxsize=1)
def get_rf_explainer():
    artifacts = get_artifacts()
    return shap.TreeExplainer(artifacts["bundle"]["models"]["Random Forest"])


@lru_cache(maxsize=1)
def get_ann_explainer():
    artifacts = get_artifacts()
    ann_model = artifacts["ann_model"]
    if ann_model is None:
        return None

    def predict_fn(values):
        return ann_model.predict(values, verbose=0).reshape(-1)

    return shap.KernelExplainer(predict_fn, artifacts["bundle"]["background_processed"])


def validate_patient_input(payload: Dict) -> Dict[str, object]:
    errors: Dict[str, str] = {}
    normalized: Dict[str, object] = {}

    for field in get_schema():
        name = field["name"]
        if name not in payload:
            errors[name] = "This field is required."
            continue

        value = payload[name]
        field_type = field["type"]

        if field_type == "slider_int":
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                errors[name] = "Must be an integer."
                continue
            if parsed < field["min"] or parsed > field["max"]:
                errors[name] = f"Value must be between {field['min']} and {field['max']}."
                continue
            normalized[name] = parsed
        elif field_type == "slider_float":
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                errors[name] = "Must be a number."
                continue
            if parsed < float(field["min"]) or parsed > float(field["max"]):
                errors[name] = f"Value must be between {field['min']} and {field['max']}."
                continue
            normalized[name] = round(parsed, 2)
        else:
            if value not in field["options"]:
                allowed_values = ", ".join(field["options"].keys())
                errors[name] = f"Must be one of: {allowed_values}."
                continue
            normalized[name] = value

    if errors:
        raise ValidationError(errors)

    return normalized


def _compute_patient_explanation(artifacts, patient_df: pd.DataFrame, model_name: str) -> List[Dict[str, float | str]]:
    bundle = artifacts["bundle"]
    processed_patient = artifacts["transform_input"](patient_df)
    feature_names = bundle["selected_features"]

    if model_name == "ANN" and artifacts["ann_model"] is not None:
        explainer = get_ann_explainer()
        shap_values = explainer.shap_values(processed_patient, nsamples=100)
        shap_values = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values)[0]
    else:
        explainer = get_rf_explainer()
        shap_values = explainer.shap_values(processed_patient)
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        elif getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[0, :, 1]
        else:
            shap_values = shap_values[0]

    rows = []
    for feature_name, feature_value, shap_value in zip(feature_names, processed_patient[0], shap_values):
        rows.append(
            {
                "feature": feature_name,
                "value": round(float(feature_value), 4),
                "shap_value": round(float(shap_value), 6),
                "abs_shap": round(abs(float(shap_value)), 6),
                "direction": "Increases risk" if float(shap_value) >= 0 else "Decreases risk",
            }
        )

    rows.sort(key=lambda item: item["abs_shap"], reverse=True)
    return rows


def _select_primary_model(probabilities: Dict[str, float]) -> Tuple[str, float]:
    if "ANN" in probabilities:
        return "ANN", probabilities["ANN"]
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return ranked[0][0], ranked[0][1]


def _normalize_series(values: np.ndarray) -> List[float]:
    max_abs = max(float(np.max(np.abs(values))), 1e-6)
    return [round(float(value / max_abs), 4) for value in values]


def _build_ann_visualization(artifacts, patient_df: pd.DataFrame) -> Dict | None:
    ann_model = artifacts["ann_model"]
    if ann_model is None:
        return None

    processed_patient = artifacts["transform_input"](patient_df)
    feature_names = artifacts["bundle"]["selected_features"]
    dense_layers = getattr(ann_model, "dense_layers", None)
    if not dense_layers:
        return None

    forward_pass = ann_model.forward(processed_patient)
    layer_outputs = forward_pass["dense_outputs"]

    input_normalized = _normalize_series(processed_patient[0])
    layers = [
        {
            "name": "Input",
            "nodes": [
                {
                    "id": f"input_{idx}",
                    "label": feature_name,
                    "value": round(float(processed_patient[0][idx]), 4),
                    "normalized": input_normalized[idx],
                    "kind": "input",
                }
                for idx, feature_name in enumerate(feature_names)
            ],
        }
    ]

    connections = []
    previous_indices = list(range(len(feature_names)))
    previous_key = "Input"

    for dense_index, dense_layer in enumerate(dense_layers):
        activations = np.asarray(layer_outputs[dense_index][0], dtype=np.float32)
        layer_name = "Output" if dense_index == len(dense_layers) - 1 else f"Hidden {dense_index + 1}"

        if dense_index == len(dense_layers) - 1:
            selected_indices = [0]
        else:
            top_k = min(8, len(activations))
            selected_indices = np.argsort(np.abs(activations))[-top_k:]
            selected_indices = selected_indices[np.argsort(-np.abs(activations[selected_indices]))].tolist()

        normalized_values = _normalize_series(activations[selected_indices])
        layer_nodes = []
        for rank, neuron_index in enumerate(selected_indices):
            raw_value = float(activations[neuron_index])
            node_id = f"layer_{dense_index}_{neuron_index}"
            layer_nodes.append(
                {
                    "id": node_id,
                    "label": "Risk" if layer_name == "Output" else f"N{rank + 1}",
                    "value": round(raw_value, 4),
                    "normalized": normalized_values[rank],
                    "kind": "output" if layer_name == "Output" else "hidden",
                    "neuron_index": int(neuron_index),
                }
            )
        layers.append({"name": layer_name, "nodes": layer_nodes})

        weights = np.asarray(dense_layer.kernel, dtype=np.float32)
        for neuron_index in selected_indices:
            if dense_index == 0:
                candidate_indices = previous_indices
            else:
                source_strength = np.abs(weights[previous_indices, neuron_index])
                top_sources = min(4, len(previous_indices))
                ranked_positions = np.argsort(source_strength)[-top_sources:]
                candidate_indices = [previous_indices[pos] for pos in ranked_positions]

            edge_weights = np.asarray([weights[source_index, neuron_index] for source_index in candidate_indices], dtype=np.float32)
            edge_norm = max(float(np.max(np.abs(edge_weights))), 1e-6)

            for source_index in candidate_indices:
                raw_weight = float(weights[source_index, neuron_index])
                source_id = f"input_{source_index}" if dense_index == 0 else f"layer_{dense_index - 1}_{source_index}"
                source_label = feature_names[source_index] if dense_index == 0 else f"{previous_key}-{source_index}"
                target_id = f"layer_{dense_index}_{neuron_index}"
                connections.append(
                    {
                        "id": f"{source_id}->{target_id}",
                        "from_id": source_id,
                        "to_id": target_id,
                        "from": source_label,
                        "to": "Risk" if layer_name == "Output" else f"{layer_name}-{neuron_index}",
                        "weight": round(raw_weight, 4),
                        "normalized_weight": round(raw_weight / edge_norm, 4),
                    }
                )

        previous_indices = selected_indices
        previous_key = layer_name

    input_strength = [
        {
            "feature": feature_name,
            "activation": round(float(value), 4),
        }
        for feature_name, value in zip(feature_names, processed_patient[0])
    ]

    return {
        "layers": layers,
        "connections": connections,
        "input_strength": input_strength,
        "output_probability": round(float(forward_pass["output"][0][0]), 6),
    }


def build_prediction_bundle(payload: Dict, explanation_model: str | None = None) -> PredictionBundle:
    started_at = perf_counter()
    normalized = validate_patient_input(payload)
    artifacts = get_artifacts()
    encoded_input = encode_user_input(normalized)
    probabilities = artifacts["predict_patient"](encoded_input)

    primary_model, primary_probability = _select_primary_model(probabilities)
    confidence = max(primary_probability, 1 - primary_probability)
    resolved_explanation_model = explanation_model or ("ANN" if "ANN" in probabilities else "Random Forest")
    if resolved_explanation_model == "ANN" and artifacts["ann_model"] is None:
        resolved_explanation_model = "Random Forest"

    shap_values = _compute_patient_explanation(artifacts, encoded_input, resolved_explanation_model)
    ann_visualization = _build_ann_visualization(artifacts, encoded_input)

    return PredictionBundle(
        raw_input=normalized,
        encoded_input=encoded_input,
        probabilities={name: round(float(value), 6) for name, value in probabilities.items()},
        primary_model=primary_model,
        primary_probability=round(float(primary_probability), 6),
        primary_label="High risk" if primary_probability >= 0.5 else "Low risk",
        confidence=round(float(confidence), 6),
        explanation_model=resolved_explanation_model,
        shap_values=shap_values,
        feature_importance=shap_values[:8],
        ann_visualization=ann_visualization,
        timestamp=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        inference_time_ms=round((perf_counter() - started_at) * 1000, 2),
    )


def build_prediction_response(payload: Dict, explanation_model: str | None = None) -> Dict:
    bundle = build_prediction_bundle(payload, explanation_model=explanation_model)
    artifacts = get_artifacts()
    return {
        "timestamp": bundle.timestamp,
        "input_data": bundle.raw_input,
        "encoded_input": bundle.encoded_input.iloc[0].to_dict(),
        "prediction": {
            "primary_model": bundle.primary_model,
            "label": bundle.primary_label,
            "probability": bundle.primary_probability,
            "confidence": bundle.confidence,
            "all_models": bundle.probabilities,
        },
        "explanation_model": bundle.explanation_model,
        "shap_values": bundle.shap_values,
        "feature_importance": bundle.feature_importance,
        "ann_visualization": bundle.ann_visualization,
        "available_models": list(bundle.probabilities.keys()),
        "available_explanations": ["Random Forest"],
        "inference_time_ms": bundle.inference_time_ms,
        "schema": list(get_schema()),
    }
