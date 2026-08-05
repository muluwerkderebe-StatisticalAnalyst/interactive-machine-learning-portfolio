"""
app.py

Interactive, Streamlit app for the Real Estate Price
Prediction project (Real_Estate.ipynb). Every tab maps to a stage of the
notebook, in the same order, using the same defaults:

  1. Explore Data        -> df.head()/tail()/shape()
  2. Train / Test Split  -> x/y split, stratify=x.property_type_Condo, test_size=0.2
  3. Linear Regression    -> LinearRegression().fit(), coef_, intercept_, MAE
  4. Random Forest        -> RandomForestRegressor(n_estimators=200, criterion='absolute_error'), MAE
  5. Pickle & Predict      -> pickle.dump/load, live prediction form

Run with:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.real_estate_pipeline import (
    PipelineError,
    evaluate,
    load_data,
    pickle_model,
    plot_single_tree,
    predict_row,
    split_data,
    train_linear_regression,
    train_random_forest,
    unpickle_model,
)

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample_data" / "final_realestateData.csv"
MAE_TARGET = 70_000

st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")
if st.session_state.get("_active_ml_page") != "real_estate":
    st.session_state.clear()
    st.session_state["_active_ml_page"] = "real_estate"
st.title(" Real Estate Price Prediction")
st.caption(
    "An interactive walkthrough of `Real_Estate.ipynb` — explore the data, split it, "
    "train a Linear Regression and a Random Forest, then pickle a model and predict."
)

# ---------------------------------------------------------------------------
# Data loading (drives every tab below via st.session_state)
# ---------------------------------------------------------------------------
st.subheader(" Load the dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV (numeric / one-hot encoded, like the notebook's `final.csv`), or leave empty to use the bundled sample",
    type=["csv"],
)

source_file = uploaded_file if uploaded_file is not None else SAMPLE_FILE
if uploaded_file is None:
    st.info(f"⬆️ No file uploaded — using the bundled sample dataset `{SAMPLE_FILE.name}`.")

try:
    df = load_data(source_file)
except PipelineError as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.get("_df_source") != getattr(source_file, "name", str(source_file)):
    # New dataset loaded -> reset everything downstream, just like re-running the notebook top to bottom.
    for key in ["x_train", "x_test", "y_train", "y_test", "stratify_col", "lrmodel", "rfmodel",
                "lr_train_mae", "lr_test_mae", "rf_train_mae", "rf_test_mae"]:
        st.session_state.pop(key, None)
    st.session_state["_df_source"] = getattr(source_file, "name", str(source_file))

st.session_state["df"] = df

tabs = st.tabs(
    [
        "1️⃣ Explore Data",
        "2️⃣ Train / Test Split",
        "3️⃣ Linear Regression",
        "4️⃣ Random Forest",
        "5️⃣ Compare Models",
        "6️⃣ Pickle & Predict",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — Explore Data  (df.head() / df.tail() / df.shape())
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("df.head()")
    st.dataframe(df.head(), width="stretch")

    st.subheader("df.tail()")
    st.dataframe(df.tail(), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("df.shape[0] — rows", f"{df.shape[0]:,}")
    c2.metric("df.shape[1] — columns", f"{df.shape[1]:,}")
    c3.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")

    with st.expander("df.describe()"):
        st.dataframe(df.describe().transpose(), width="stretch")

    with st.expander("Correlation with price"):
        corr = df.corr(numeric_only=True)["price"].drop("price").sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, len(corr) * 0.35)))
        corr.plot.barh(ax=ax, color=["#d62728" if v < 0 else "#1f77b4" for v in corr])
        ax.set_xlabel("Correlation with price")
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 2 — Train / Test Split
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Split the data")
    st.markdown(
        "Notebook: `x = df.drop('price', axis=1)`, `y = df['price']`, then "
        "`train_test_split(x, y, test_size=0.2, stratify=x.property_type_Condo)`."
    )

    c1, c2 = st.columns(2)
    test_size = c1.slider("test_size", 0.1, 0.4, 0.2, 0.05)
    use_stratify = c2.checkbox(
        "Stratify on `property_type_Condo` (matches the notebook)",
        value=True,
        help="Keeps the proportion of condos identical in train and test, exactly like the notebook's `stratify=` argument.",
    )

    if st.button("✂️ Split the data", type="primary"):
        try:
            x_train, x_test, y_train, y_test, strat_col = split_data(
                df, test_size=test_size, use_stratify=use_stratify
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not split the data: {exc}")
            st.stop()

        st.session_state["x_train"] = x_train
        st.session_state["x_test"] = x_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["stratify_col"] = strat_col
        for key in ["lrmodel", "rfmodel", "lr_train_mae", "lr_test_mae", "rf_train_mae", "rf_test_mae"]:
            st.session_state.pop(key, None)
        st.success("Data split. Head to the next tab to train a model.")

    if "x_train" in st.session_state:
        x_train = st.session_state["x_train"]
        x_test = st.session_state["x_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]

        st.markdown("**Shapes** — `x_train.shape, y_train.shape, x_test.shape, y_test.shape`")
        shape_df = pd.DataFrame(
            {
                "": ["x_train", "y_train", "x_test", "y_test"],
                "rows": [x_train.shape[0], y_train.shape[0], x_test.shape[0], y_test.shape[0]],
                "columns": [x_train.shape[1], 1, x_test.shape[1], 1],
            }
        ).set_index("")
        st.dataframe(shape_df, width="stretch")

        if st.session_state.get("stratify_col"):
            st.markdown(f"**`x_train.{st.session_state['stratify_col']}.value_counts()`**")
            c1, c2 = st.columns(2)
            c1.write("Train set")
            c1.dataframe(x_train[st.session_state["stratify_col"]].value_counts())
            c2.write("Test set")
            c2.dataframe(x_test[st.session_state["stratify_col"]].value_counts())

        st.markdown("**`x_train.head()`**")
        st.dataframe(x_train.head(), width="stretch")
    else:
        st.info("Click **Split the data** above to continue.")

# ---------------------------------------------------------------------------
# Tab 3 — Linear Regression
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Build a Linear Regression model on the training set")
    st.markdown("Notebook: `model = LinearRegression(); lrmodel = model.fit(x_train, y_train)`")

    if "x_train" not in st.session_state:
        st.warning("Split the data in the previous tab first.")
    else:
        if st.button("📈 Train Linear Regression", type="primary"):
            x_train, y_train = st.session_state["x_train"], st.session_state["y_train"]
            x_test, y_test = st.session_state["x_test"], st.session_state["y_test"]

            lrmodel = train_linear_regression(x_train, y_train)
            train_pred, train_mae = evaluate(lrmodel, x_train, y_train)
            test_pred, test_mae = evaluate(lrmodel, x_test, y_test)

            st.session_state["lrmodel"] = lrmodel
            st.session_state["lr_train_mae"] = train_mae
            st.session_state["lr_test_mae"] = test_mae
            st.session_state["lr_train_pred"] = train_pred
            st.session_state["lr_test_pred"] = test_pred
            st.success("Linear Regression trained.")

        if "lrmodel" in st.session_state:
            lrmodel = st.session_state["lrmodel"]
            x_train = st.session_state["x_train"]

            st.markdown("**`lrmodel.coef_`**")
            coef_df = pd.DataFrame(
                {"feature": x_train.columns, "coefficient": lrmodel.coef_}
            ).sort_values("coefficient", key=abs, ascending=False)
            st.dataframe(coef_df, width="stretch")

            st.metric("`lrmodel.intercept_`", f"{lrmodel.intercept_:,.2f}")

            st.markdown("**`train_pred = lrmodel.predict(x_train)`** (first 10 rows)")
            preview = pd.DataFrame(
                {
                    "actual_price": st.session_state["y_train"].head(10).values,
                    "predicted_price": st.session_state["lr_train_pred"][:10],
                }
            )
            st.dataframe(preview.style.format("{:,.0f}"), width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Train MAE", f"${st.session_state['lr_train_mae']:,.0f}")
            c2.metric(
                "Test MAE",
                f"${st.session_state['lr_test_mae']:,.0f}",
                delta=f"{'✅ under' if st.session_state['lr_test_mae'] < MAE_TARGET else '❌ over'} ${MAE_TARGET:,} target",
                delta_color="normal" if st.session_state["lr_test_mae"] < MAE_TARGET else "inverse",
            )
            if st.session_state["lr_test_mae"] >= MAE_TARGET:
                st.caption(
                    "Same conclusion as the notebook: *'Our model is still not good because we need "
                    "MAE < $70,000. Note - we have not scaled the features and not tuned the model.'*"
                )

# ---------------------------------------------------------------------------
# Tab 4 — Random Forest
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Build a Random Forest model")
    st.markdown("Notebook: `RandomForestRegressor(n_estimators=200, criterion='absolute_error')`")

    if "x_train" not in st.session_state:
        st.warning("Split the data in the **Train / Test Split** tab first.")
    else:
        c1, c2, c3 = st.columns(3)
        n_estimators = c1.slider("n_estimators", 50, 500, 200, 50)
        criterion = c2.selectbox(
            "criterion", ["absolute_error", "squared_error", "friedman_mse", "poisson"], index=0
        )
        max_depth = c3.slider("max_depth (0 = unlimited)", 0, 30, 0, 1)

        if st.button("🌲 Train Random Forest", type="primary"):
            x_train, y_train = st.session_state["x_train"], st.session_state["y_train"]
            x_test, y_test = st.session_state["x_test"], st.session_state["y_test"]

            rfmodel = train_random_forest(
                x_train, y_train,
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=None if max_depth == 0 else max_depth,
            )
            train_pred, train_mae = evaluate(rfmodel, x_train, y_train)
            test_pred, test_mae = evaluate(rfmodel, x_test, y_test)

            st.session_state["rfmodel"] = rfmodel
            st.session_state["rf_train_mae"] = train_mae
            st.session_state["rf_test_mae"] = test_mae
            st.success("Random Forest trained.")

        if "rfmodel" in st.session_state:
            rfmodel = st.session_state["rfmodel"]
            x_train = st.session_state["x_train"]

            c1, c2 = st.columns(2)
            c1.metric("Train MAE", f"${st.session_state['rf_train_mae']:,.0f}")
            c2.metric(
                "Test MAE",
                f"${st.session_state['rf_test_mae']:,.0f}",
                delta=f"{'✅ under' if st.session_state['rf_test_mae'] < MAE_TARGET else '❌ over'} ${MAE_TARGET:,} target",
                delta_color="normal" if st.session_state["rf_test_mae"] < MAE_TARGET else "inverse",
            )

            st.markdown("**Feature importance**")
            importance_df = pd.DataFrame(
                {"feature": x_train.columns, "importance": rfmodel.feature_importances_}
            ).sort_values("importance", ascending=False)
            st.bar_chart(importance_df.set_index("feature"))

            with st.expander("🌳 Visualize a single tree from the forest (the notebook's commented-out cell, made real)"):
                st.caption("Notebook reference: `tree.plot_tree(rfmodel.estimators_[2], feature_names=dtmodel.feature_names_in_)`")
                tc1, tc2 = st.columns(2)
                tree_index = tc1.slider("Tree index", 0, len(rfmodel.estimators_) - 1, 2)
                tree_depth = tc2.slider("Max depth to display", 1, 6, 3)
                fig = plot_single_tree(rfmodel, x_train.columns, tree_index=tree_index, max_depth=tree_depth)
                st.pyplot(fig)

# ---------------------------------------------------------------------------
# Tab 5 — Compare Models
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Linear Regression vs. Random Forest")
    if "lrmodel" not in st.session_state and "rfmodel" not in st.session_state:
        st.info("Train at least one model in the previous tabs to compare results here.")
    else:
        rows = []
        if "lrmodel" in st.session_state:
            rows.append(
                {"Model": "Linear Regression", "Train MAE": st.session_state["lr_train_mae"], "Test MAE": st.session_state["lr_test_mae"]}
            )
        if "rfmodel" in st.session_state:
            rows.append(
                {"Model": "Random Forest", "Train MAE": st.session_state["rf_train_mae"], "Test MAE": st.session_state["rf_test_mae"]}
            )
        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df.style.format({"Train MAE": "${:,.0f}", "Test MAE": "${:,.0f}"}), width="stretch")
        st.bar_chart(metrics_df.set_index("Model")[["Test MAE"]])

        best = metrics_df.sort_values("Test MAE").iloc[0]
        st.caption(f" Best model on the test set: **{best['Model']}** (Test MAE ${best['Test MAE']:,.0f})")

# ---------------------------------------------------------------------------
# Tab 6 — Pickle & Predict
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Pickle: save, load, and predict")
    st.markdown(
        "Notebook: `pickle.dump(model, open('RE_Model','wb'))` then "
        "`RE_Model = pickle.load(open('RE_Model','rb'))` and `RE_Model.predict([[...]])`."
    )

    trained_models = {}
    if "lrmodel" in st.session_state:
        trained_models["Linear Regression"] = st.session_state["lrmodel"]
    if "rfmodel" in st.session_state:
        trained_models["Random Forest"] = st.session_state["rfmodel"]

    if not trained_models:
        st.warning("Train a model in the Linear Regression or Random Forest tab first.")
    else:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### 💾 Save a model (`pickle.dump`)")
            choice = st.radio("Which trained model to pickle?", list(trained_models.keys()), horizontal=True)
            model_to_save = trained_models[choice]
            pkl_bytes = pickle_model(model_to_save)
            st.download_button(
                "⬇️ Download RE_Model.pkl",
                data=pkl_bytes,
                file_name="RE_Model.pkl",
                mime="application/octet-stream",
            )

        with c2:
            st.markdown("#### 📂 Load a pickled model (`pickle.load`)")
            uploaded_pkl = st.file_uploader("Upload a .pkl model", type=["pkl"], key="pkl_uploader")
            if uploaded_pkl is not None:
                try:
                    loaded_model = unpickle_model(uploaded_pkl.read())
                    st.session_state["loaded_model"] = loaded_model
                    st.success(f"Loaded `{type(loaded_model).__name__}` from the uploaded pickle.")
                except PipelineError as exc:
                    st.error(str(exc))

        st.divider()
        st.markdown("#### 🔮 Make a prediction")

        predict_source = st.radio(
            "Predict using:",
            [m for m in trained_models] + (["Uploaded pickled model"] if "loaded_model" in st.session_state else []),
            horizontal=True,
        )
        model_for_prediction = (
            st.session_state["loaded_model"] if predict_source == "Uploaded pickled model" else trained_models[predict_source]
        )

        feature_names = st.session_state["x_train"].columns.tolist()
        default_row = st.session_state["x_test"].iloc[0] if "x_test" in st.session_state else df.drop(columns=["price"]).iloc[0]

        with st.form("predict_form"):
            st.caption("Defaults are pre-filled from the first row of the test set (like `x_test.head(1)` in the notebook).")
            input_values = {}
            n_cols = 4
            cols = st.columns(n_cols)
            for i, feature in enumerate(feature_names):
                default_val = float(default_row.get(feature, 0.0))
                input_values[feature] = cols[i % n_cols].number_input(feature, value=round(default_val, 2))
            submitted = st.form_submit_button("Predict price", type="primary")

        if submitted:
            try:
                pred = predict_row(model_for_prediction, input_values, feature_names)
                st.success(f"💰 Predicted price: **${pred:,.0f}**")
                if "y_test" in st.session_state:
                    actual = float(st.session_state["y_test"].iloc[0])
                    st.caption(f"For reference, the actual price of that first test-set row was ${actual:,.0f}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
