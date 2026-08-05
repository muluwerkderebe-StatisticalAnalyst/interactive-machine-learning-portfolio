"""
src/pipeline.py

Every function here maps 1:1 to a section of Real_Estate.ipynb:

  load data & explore        -> load_data()
  x/y split + train_test_split -> split_data()
  Linear Regression model    -> train_linear_regression()
  Random Forest model        -> train_random_forest()
  MAE evaluation              -> evaluate()
  Pickle save/load            -> pickle_model() / unpickle_model()
  Single-tree visualization  -> plot_single_tree()  (the notebook's commented-out
                                 `tree.plot_tree(...)` cell, made real)

Nothing here is Streamlit-specific - it's plain pandas/sklearn, so it's easy
to test and reason about independently of the UI.
"""

from __future__ import annotations

import io
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

TARGET_COL = "price"
STRATIFY_COL = "property_type_Condo"  # notebook: stratify=x.property_type_Condo


class PipelineError(Exception):
    """Raised for any step that can't proceed given the current data/state."""


# ---------------------------------------------------------------------------
# 1. Load & explore
# ---------------------------------------------------------------------------
def load_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Could not read this CSV: {exc}") from exc

    if TARGET_COL not in df.columns:
        raise PipelineError(f"This dataset needs a `{TARGET_COL}` column (the notebook's target variable).")

    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise PipelineError(
            "All columns must already be numeric / one-hot encoded, like the notebook's `final.csv`. "
            f"Non-numeric columns found: {non_numeric}"
        )

    return df


# ---------------------------------------------------------------------------
# 2. x/y split + train_test_split (with the notebook's stratify behaviour)
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame, test_size: float = 0.2, use_stratify: bool = True, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[str]]:
    x = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    stratify_col_used = None
    stratify_arg = None
    if use_stratify and STRATIFY_COL in x.columns:
        stratify_arg = x[STRATIFY_COL]
        stratify_col_used = STRATIFY_COL

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return x_train, x_test, y_train, y_test, stratify_col_used


# ---------------------------------------------------------------------------
# 3. Linear Regression (notebook: model = LinearRegression(); lrmodel = model.fit(x_train, y_train))
# ---------------------------------------------------------------------------
def train_linear_regression(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    lrmodel = model.fit(x_train, y_train)
    return lrmodel


# ---------------------------------------------------------------------------
# 4. Random Forest (notebook: RandomForestRegressor(n_estimators=200, criterion='absolute_error'))
# ---------------------------------------------------------------------------
def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    criterion: str = "absolute_error",
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )
    rfmodel = rf.fit(x_train, y_train)
    return rfmodel


# ---------------------------------------------------------------------------
# 5. Evaluation (notebook: mean_absolute_error(pred, y))
# ---------------------------------------------------------------------------
def evaluate(model, x: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
    pred = model.predict(x)
    mae = mean_absolute_error(pred, y)
    return pred, mae


# ---------------------------------------------------------------------------
# 6. Pickle save / load (notebook: pickle.dump(...), pickle.load(...))
# ---------------------------------------------------------------------------
def pickle_model(model) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return buffer.getvalue()


def unpickle_model(file_bytes: bytes):
    try:
        return pickle.load(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Could not load this as a pickled scikit-learn model: {exc}") from exc


def predict_row(model, input_values: dict, feature_names) -> float:
    row = pd.DataFrame([[input_values.get(f, 0.0) for f in feature_names]], columns=feature_names)
    pred = model.predict(row)
    return float(np.asarray(pred).ravel()[0])


# ---------------------------------------------------------------------------
# Bonus: the notebook's commented-out single-tree visualization, made real
# ---------------------------------------------------------------------------
def plot_single_tree(rfmodel: RandomForestRegressor, feature_names, tree_index: int = 0, max_depth: int = 3):
    """Returns a matplotlib Figure visualizing one tree from the forest."""
    import matplotlib.pyplot as plt
    from sklearn import tree as sk_tree

    fig, ax = plt.subplots(figsize=(16, 8))
    sk_tree.plot_tree(
        rfmodel.estimators_[tree_index],
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Random Forest — Tree #{tree_index} (truncated to depth {max_depth})")
    return fig
