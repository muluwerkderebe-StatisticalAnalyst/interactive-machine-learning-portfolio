"""Notebook-faithful K-Means clustering functions."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


ID_COL = "Customer_ID"
FEATURES_2D = ["Annual_Income", "Spending_Score"]
FEATURES_3D = ["Age", "Annual_Income", "Spending_Score"]
REQUIRED_COLS = [ID_COL, "Gender", *FEATURES_3D]


class PipelineError(Exception):
    """Raised when the supplied data cannot support a notebook step."""


@dataclass
class ClusterBundle:
    """A saved K-Means model plus the feature order required for prediction."""

    model: KMeans
    feature_names: list[str]


def load_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise PipelineError(f"Could not read this CSV: {exc}") from exc
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise PipelineError(f"Required columns are missing: {missing}")
    for col in FEATURES_3D:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[FEATURES_3D].isna().any().any():
        bad = df[FEATURES_3D].columns[df[FEATURES_3D].isna().any()].tolist()
        raise PipelineError(f"These clustering features contain missing/non-numeric values: {bad}")
    return df


def fit_kmeans(df: pd.DataFrame, feature_names: list[str], n_clusters: int,
               init="k-means++", n_init=10, max_iter=300, random_state=42):
    model = KMeans(
        n_clusters=n_clusters, init=init, n_init=n_init,
        max_iter=max_iter, random_state=random_state,
    )
    labels = model.fit_predict(df[feature_names])
    clustered = df.copy()
    clustered["Cluster"] = labels
    return model, clustered


def cluster_diagnostics(df: pd.DataFrame, feature_names: list[str], k_values,
                        init="k-means++", n_init=10, max_iter=300,
                        random_state=42) -> pd.DataFrame:
    rows = []
    data = df[feature_names]
    for k in k_values:
        model = KMeans(
            n_clusters=int(k), init=init, n_init=n_init,
            max_iter=max_iter, random_state=random_state,
        )
        labels = model.fit_predict(data)
        rows.append({
            "cluster": int(k),
            "WCSS_Score": float(model.inertia_),
            "Silhouette_Score": float(silhouette_score(data, labels)),
        })
    return pd.DataFrame(rows)


def cluster_profiles(clustered: pd.DataFrame) -> pd.DataFrame:
    profile = clustered.groupby("Cluster").agg(
        Customers=(ID_COL, "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual_Income", "mean"),
        Average_Spending_Score=("Spending_Score", "mean"),
        Female_Percent=("Gender", lambda s: 100 * (s.astype(str).str.lower() == "female").mean()),
    ).reset_index()
    return profile


def segment_names(profile: pd.DataFrame) -> dict[int, str]:
    income_mid = profile["Average_Income"].median()
    spending_mid = profile["Average_Spending_Score"].median()
    names = {}
    for row in profile.itertuples():
        high_income = row.Average_Income >= income_mid
        high_spend = row.Average_Spending_Score >= spending_mid
        if high_income and high_spend:
            name = "High-value customers"
        elif high_income and not high_spend:
            name = "High-income cautious spenders"
        elif not high_income and high_spend:
            name = "Enthusiastic value shoppers"
        else:
            name = "Budget-conscious customers"
        names[int(row.Cluster)] = name
    return names


def predict_cluster(bundle: ClusterBundle, values: dict) -> int:
    row = pd.DataFrame([[float(values[f]) for f in bundle.feature_names]],
                       columns=bundle.feature_names)
    return int(bundle.model.predict(row)[0])


def pickle_bundle(bundle: ClusterBundle) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(bundle, buffer)
    return buffer.getvalue()


def unpickle_bundle(file_bytes: bytes) -> ClusterBundle:
    try:
        bundle = pickle.loads(file_bytes)
    except Exception as exc:
        raise PipelineError(f"Could not load this pickle file: {exc}") from exc
    if not isinstance(bundle, ClusterBundle):
        raise PipelineError("The uploaded file is not a Mall Customer ClusterBundle.")
    return bundle
