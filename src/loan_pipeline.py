"""Notebook-faithful data preparation and modelling functions."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_COL = "Loan_Approved"
ID_COL = "Loan_ID"
CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"
]
REQUIRED_COLS = [ID_COL, *CATEGORICAL_COLS[:5], "ApplicantIncome", "CoapplicantIncome",
                 "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", TARGET_COL]


class PipelineError(Exception):
    """Raised when a notebook step cannot run with the supplied data."""


@dataclass
class ModelBundle:
    """Everything required to transform raw form values and predict."""

    model_name: str
    model: object
    scaler: MinMaxScaler
    feature_names: list[str]
    raw_defaults: dict


def load_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise PipelineError(f"Could not read this CSV: {exc}") from exc
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise PipelineError(f"Required columns are missing: {missing}")
    return df


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact imputation choices demonstrated in the notebook."""
    clean = df.copy()
    categorical = ["Gender", "Married", "Dependents", "Self_Employed",
                   "Loan_Amount_Term", "Credit_History"]
    for col in categorical:
        if clean[col].dropna().empty:
            raise PipelineError(f"Cannot calculate a mode for `{col}`.")
        clean[col] = clean[col].fillna(clean[col].mode(dropna=True)[0])
    if clean["LoanAmount"].dropna().empty:
        raise PipelineError("Cannot calculate the median for `LoanAmount`.")
    clean["LoanAmount"] = clean["LoanAmount"].fillna(clean["LoanAmount"].median())
    return clean


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop Loan_ID, one-hot encode categories, and map Y/N to 1/0."""
    raw = impute_data(df).drop(columns=[ID_COL]).copy()
    processed = pd.get_dummies(raw, columns=CATEGORICAL_COLS, dtype=int)
    mapped_target = processed[TARGET_COL].map({"Y": 1, "N": 0})
    if mapped_target.isna().any():
        unexpected = processed.loc[mapped_target.isna(), TARGET_COL].drop_duplicates().tolist()
        raise PipelineError(f"`{TARGET_COL}` must contain only Y or N; found: {unexpected}")
    processed[TARGET_COL] = mapped_target.astype(int)
    return raw, processed


def split_and_scale(processed: pd.DataFrame, test_size: float = 0.2,
                    random_state: int = 42):
    x = processed.drop(columns=[TARGET_COL])
    y = processed[TARGET_COL]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled, scaler


def train_logistic(x_train_scaled, y_train, max_iter: int = 1000):
    return LogisticRegression(max_iter=max_iter, random_state=42).fit(x_train_scaled, y_train)


def train_decision_tree(x_train_scaled, y_train, criterion="gini", max_depth=None,
                        min_samples_split=2):
    return DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
        random_state=42
    ).fit(x_train_scaled, y_train)


def train_random_forest(x_train_scaled, y_train, n_estimators=100, criterion="gini",
                        max_depth=None, max_features="sqrt"):
    return RandomForestClassifier(
        n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
        max_features=max_features, random_state=42, n_jobs=-1
    ).fit(x_train_scaled, y_train)


def evaluate(model, x_test_scaled, y_test, threshold: float = 0.5) -> dict:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(x_test_scaled)[:, 1]
        prediction = (probability >= threshold).astype(int)
    else:
        prediction = model.predict(x_test_scaled)
        probability = prediction.astype(float)
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, prediction),
        "prediction": prediction,
        "probability": probability,
    }


def cross_validate(model, x_train_scaled, y_train, folds=5, stratified=False):
    cv = (StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
          if stratified else KFold(n_splits=folds, shuffle=True, random_state=42))
    return cross_val_score(model, x_train_scaled, y_train, cv=cv, scoring="accuracy")


def default_raw_values(raw: pd.DataFrame) -> dict:
    defaults = {}
    for col in CATEGORICAL_COLS:
        defaults[col] = raw[col].mode()[0]
    for col in NUMERIC_COLS:
        defaults[col] = float(pd.to_numeric(raw[col]).median())
    return defaults


def encode_application(values: dict, feature_names: list[str]) -> pd.DataFrame:
    row = {name: 0.0 for name in feature_names}
    for col in NUMERIC_COLS:
        if col in row:
            row[col] = float(values[col])
    for col in CATEGORICAL_COLS:
        dummy = f"{col}_{values[col]}"
        if dummy in row:
            row[dummy] = 1.0
    return pd.DataFrame([row], columns=feature_names)


def predict_application(bundle: ModelBundle, values: dict) -> tuple[int, float]:
    encoded = encode_application(values, bundle.feature_names)
    scaled = bundle.scaler.transform(encoded)
    probability = float(bundle.model.predict_proba(scaled)[0, 1])
    return int(probability >= 0.5), probability


def pickle_bundle(bundle: ModelBundle) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(bundle, buffer)
    return buffer.getvalue()


def unpickle_bundle(file_bytes: bytes) -> ModelBundle:
    try:
        bundle = pickle.loads(file_bytes)
    except Exception as exc:
        raise PipelineError(f"Could not load this pickle file: {exc}") from exc
    if not isinstance(bundle, ModelBundle):
        raise PipelineError("The uploaded file is not a Loan Eligibility ModelBundle.")
    return bundle
