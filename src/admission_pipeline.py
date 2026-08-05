"""Notebook-faithful preprocessing and MLP classification functions."""

from __future__ import annotations

import io
import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


TARGET_COL = "Admit_Chance"
ID_COL = "Serial_No"
RAW_FEATURES = [
    "GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"
]
REQUIRED_COLS = [ID_COL, *RAW_FEATURES, TARGET_COL]


class PipelineError(Exception):
    """Raised when the supplied data cannot support a notebook step."""


@dataclass
class AdmissionBundle:
    """A trained network and preprocessing objects needed for live prediction."""

    model: MLPClassifier
    scaler: MinMaxScaler
    feature_names: list[str]
    threshold: float


def load_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise PipelineError(f"Could not read this CSV: {exc}") from exc
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise PipelineError(f"Required columns are missing: {missing}")
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[REQUIRED_COLS].isna().any().any():
        bad = df[REQUIRED_COLS].columns[df[REQUIRED_COLS].isna().any()].tolist()
        raise PipelineError(f"These required columns contain missing/non-numeric values: {bad}")
    return df


def classify_target(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    data = df.copy()
    data[TARGET_COL] = (data[TARGET_COL] >= threshold).astype(int)
    return data


def prepare_data(df: pd.DataFrame, threshold: float = 0.8):
    classified = classify_target(df, threshold).drop(columns=[ID_COL])
    clean = pd.get_dummies(
        classified,
        columns=["University_Rating", "Research"],
        dtype=int,
    )
    return classified, clean


def split_and_scale(clean: pd.DataFrame, test_size=0.2, random_state=123):
    x = clean.drop(columns=[TARGET_COL])
    y = clean[TARGET_COL]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler


def train_mlp(x_train_scaled, y_train, hidden_layer_sizes=(3, 4), activation="relu",
              solver="adam", batch_size=50, learning_rate_init=0.001,
              alpha=0.0001, max_iter=200, random_state=123,
              early_stopping=False):
    model = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layer_sizes), activation=activation,
        solver=solver, batch_size=batch_size, learning_rate_init=learning_rate_init,
        alpha=alpha, max_iter=max_iter, random_state=random_state,
        early_stopping=early_stopping,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train_scaled, y_train)
    return model


def evaluate(model, x_scaled, y) -> dict:
    prediction = model.predict(x_scaled)
    probability = model.predict_proba(x_scaled)[:, 1]
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
        "confusion_matrix": confusion_matrix(y, prediction),
        "prediction": prediction,
        "probability": probability,
    }


def encode_applicant(values: dict, feature_names: list[str]) -> pd.DataFrame:
    row = {name: 0.0 for name in feature_names}
    for col in ["GRE_Score", "TOEFL_Score", "SOP", "LOR", "CGPA"]:
        if col in row:
            row[col] = float(values[col])
    university_dummy = f"University_Rating_{int(values['University_Rating'])}"
    research_dummy = f"Research_{int(values['Research'])}"
    if university_dummy in row:
        row[university_dummy] = 1.0
    if research_dummy in row:
        row[research_dummy] = 1.0
    return pd.DataFrame([row], columns=feature_names)


def predict_applicant(bundle: AdmissionBundle, values: dict) -> tuple[int, float]:
    encoded = encode_applicant(values, bundle.feature_names)
    scaled = bundle.scaler.transform(encoded)
    probability = float(bundle.model.predict_proba(scaled)[0, 1])
    return int(probability >= 0.5), probability


def pickle_bundle(bundle: AdmissionBundle) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(bundle, buffer)
    return buffer.getvalue()


def unpickle_bundle(file_bytes: bytes) -> AdmissionBundle:
    try:
        bundle = pickle.loads(file_bytes)
    except Exception as exc:
        raise PipelineError(f"Could not load this pickle file: {exc}") from exc
    if not isinstance(bundle, AdmissionBundle):
        raise PipelineError("The uploaded file is not a UCLA AdmissionBundle.")
    return bundle
