from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np


@dataclass
class DenseLayerWeights:
    kernel: np.ndarray
    bias: np.ndarray
    activation: str
    name: str


@dataclass
class BatchNormWeights:
    gamma: np.ndarray
    beta: np.ndarray
    moving_mean: np.ndarray
    moving_variance: np.ndarray
    epsilon: float
    name: str


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return _relu(values)
    if activation == "sigmoid":
        return _sigmoid(values)
    if activation in {"linear", None}:
        return values
    raise ValueError(f"Unsupported activation: {activation}")


class PortableANNModel:
    def __init__(self, dense_layers: List[DenseLayerWeights], batch_norm_layers: List[BatchNormWeights]):
        self.dense_layers = dense_layers
        self.batch_norm_layers = batch_norm_layers

    @classmethod
    def load(cls, keras_path: str | Path) -> "PortableANNModel":
        keras_path = Path(keras_path)
        with zipfile.ZipFile(keras_path, "r") as archive:
            config = json.loads(archive.read("config.json"))
            with tempfile.TemporaryDirectory() as temp_dir:
                archive.extract("model.weights.h5", temp_dir)
                weights_path = Path(temp_dir) / "model.weights.h5"
                dense_layers, batch_norm_layers = cls._read_weights(weights_path, config)
        return cls(dense_layers=dense_layers, batch_norm_layers=batch_norm_layers)

    @staticmethod
    def _read_weights(weights_path: Path, config: dict) -> tuple[List[DenseLayerWeights], List[BatchNormWeights]]:
        dense_activations = [
            layer["config"].get("activation", "linear")
            for layer in config["config"]["layers"]
            if layer["class_name"] == "Dense"
        ]
        batch_norm_configs = [
            layer["config"]
            for layer in config["config"]["layers"]
            if layer["class_name"] == "BatchNormalization"
        ]

        dense_layers: List[DenseLayerWeights] = []
        batch_norm_layers: List[BatchNormWeights] = []

        with h5py.File(weights_path, "r") as weights_file:
            layer_group = weights_file["layers"]
            dense_names = sorted([name for name in layer_group.keys() if name.startswith("dense")])
            batch_norm_names = sorted([name for name in layer_group.keys() if name.startswith("batch_normalization")])

            for index, name in enumerate(dense_names):
                vars_group = layer_group[name]["vars"]
                dense_layers.append(
                    DenseLayerWeights(
                        kernel=np.asarray(vars_group["0"], dtype=np.float32),
                        bias=np.asarray(vars_group["1"], dtype=np.float32),
                        activation=dense_activations[index],
                        name=name,
                    )
                )

            for index, name in enumerate(batch_norm_names):
                vars_group = layer_group[name]["vars"]
                config_item = batch_norm_configs[index]
                batch_norm_layers.append(
                    BatchNormWeights(
                        gamma=np.asarray(vars_group["0"], dtype=np.float32),
                        beta=np.asarray(vars_group["1"], dtype=np.float32),
                        moving_mean=np.asarray(vars_group["2"], dtype=np.float32),
                        moving_variance=np.asarray(vars_group["3"], dtype=np.float32),
                        epsilon=float(config_item.get("epsilon", 0.001)),
                        name=name,
                    )
                )

        return dense_layers, batch_norm_layers

    def forward(self, inputs: np.ndarray) -> dict:
        current = np.asarray(inputs, dtype=np.float32)
        dense_outputs: List[np.ndarray] = []

        for dense_index, dense_layer in enumerate(self.dense_layers):
            current = np.matmul(current, dense_layer.kernel) + dense_layer.bias
            current = _apply_activation(current, dense_layer.activation)
            dense_outputs.append(current.copy())

            if dense_index < len(self.batch_norm_layers):
                batch_norm = self.batch_norm_layers[dense_index]
                current = (
                    batch_norm.gamma
                    * (current - batch_norm.moving_mean)
                    / np.sqrt(batch_norm.moving_variance + batch_norm.epsilon)
                ) + batch_norm.beta

        return {
            "dense_outputs": dense_outputs,
            "output": dense_outputs[-1],
        }

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        forward_pass = self.forward(inputs)
        return np.asarray(forward_pass["output"], dtype=np.float32)
