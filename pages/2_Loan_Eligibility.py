"""Interactive, notebook-faithful Loan Eligibility classification app."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.loan_pipeline import (
    CATEGORICAL_COLS, NUMERIC_COLS, ModelBundle, PipelineError, cross_validate,
    default_raw_values, evaluate, impute_data, load_data, pickle_bundle,
    predict_application, prepare_data, split_and_scale, train_decision_tree,
    train_logistic, train_random_forest, unpickle_bundle,
)


SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample_data" / "credit_loanData.csv"
SUCCESS_TARGET = 0.76

st.set_page_config(page_title="Loan Eligibility Prediction", page_icon="🏦", layout="wide")
if st.session_state.get("_active_ml_page") != "loan_eligibility":
    st.session_state.clear()
    st.session_state["_active_ml_page"] = "loan_eligibility"
st.title("🏦 Loan Eligibility Prediction Model")
st.caption(
    "An interactive walkthrough of `Loan_Eligibility_Model_Solution.ipynb`: analyze and clean "
    "the data, encode and scale it, train three classifiers, tune models, validate performance, "
    "and predict whether a new application is eligible."
)


def reset_downstream():
    keys = [
        "imputed", "raw", "processed", "x_train", "x_test", "y_train", "y_test",
        "x_train_scaled", "x_test_scaled", "scaler", "lrmodel", "dtmodel", "rfmodel",
        "lr_metrics", "dt_metrics", "rf_metrics", "loaded_bundle",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def store_split(processed, test_size, random_state):
    values = split_and_scale(processed, test_size=test_size, random_state=random_state)
    names = ["x_train", "x_test", "y_train", "y_test", "x_train_scaled", "x_test_scaled", "scaler"]
    for name, value in zip(names, values):
        st.session_state[name] = value
    for key in ["lrmodel", "dtmodel", "rfmodel", "lr_metrics", "dt_metrics", "rf_metrics"]:
        st.session_state.pop(key, None)


def show_metrics(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("F1 score", f"{metrics['f1']:.1%}")
    status = "✅ Meets" if metrics["accuracy"] >= SUCCESS_TARGET else "❌ Below"
    st.caption(f"{status} the notebook success criterion of **76% accuracy**.")
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual: Denied", "Actual: Approved"],
        columns=["Predicted: Denied", "Predicted: Approved"],
    )
    st.dataframe(cm, width="stretch")


st.subheader("📥 Load the dataset")
uploaded = st.file_uploader(
    "Upload the original loan CSV, or leave empty to use the bundled 614-row dataset",
    type=["csv"],
)
source = uploaded if uploaded is not None else SAMPLE_FILE
if uploaded is None:
    st.info(f"No file uploaded — using `{SAMPLE_FILE.name}`.")

try:
    df = load_data(source)
except PipelineError as exc:
    st.error(str(exc))
    st.stop()

source_name = getattr(source, "name", str(source))
if st.session_state.get("_source") != source_name:
    reset_downstream()
    st.session_state["_source"] = source_name
st.session_state["df"] = df


tabs = st.tabs([
    "1️⃣ Explore Data",
    "2️⃣ Missing Values",
    "3️⃣ Prepare & Encode",
    "4️⃣ Split & Scale",
    "5️⃣ Logistic Regression",
    "6️⃣ Decision Tree",
    "7️⃣ Random Forest",
    "8️⃣ Compare & Validate",
    "9️⃣ Pickle & Predict",
])


with tabs[0]:
    st.subheader("Explore the original credit-loan dataset")
    st.dataframe(df.head(), width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Approved", f"{(df['Loan_Approved'] == 'Y').sum():,}")
    c4.metric("Denied", f"{(df['Loan_Approved'] == 'N').sum():,}")

    target_counts = df["Loan_Approved"].value_counts().rename(index={"Y": "Approved", "N": "Denied"})
    st.markdown("#### Target variable: `Loan_Approved`")
    st.bar_chart(target_counts)
    approval_rate = (df["Loan_Approved"] == "Y").mean()
    st.caption(f"Approved applications: **{approval_rate:.1%}** ({(df['Loan_Approved'] == 'Y').sum()} of {len(df)}).")

    c1, c2 = st.columns(2)
    with c1.expander("`df.describe(include='all')`", expanded=False):
        st.dataframe(df.describe(include="all").transpose(), width="stretch")
    with c2.expander("Data dictionary", expanded=False):
        dictionary = {
            "Loan_ID": "Applicant identifier", "Gender": "Male/Female", "Married": "Marital status",
            "Dependents": "Number of dependants", "Education": "Highest education level",
            "Self_Employed": "Self-employed Yes/No", "ApplicantIncome": "Monthly applicant income",
            "CoapplicantIncome": "Monthly co-applicant income", "LoanAmount": "Requested amount in $1,000s",
            "Loan_Amount_Term": "Term in months", "Credit_History": "Credit-history indicator",
            "Property_Area": "Rural/Semiurban/Urban", "Loan_Approved": "Approved Y/N",
        }
        st.dataframe(pd.DataFrame(dictionary.items(), columns=["Variable", "Meaning"]), hide_index=True)


with tabs[1]:
    st.subheader("Missing-value analysis and imputation")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing.rename("Missing values").to_frame(), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### LoanAmount distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["LoanAmount"].dropna(), bins=30, color="#1f77b4", edgecolor="white")
        ax.axvline(df["LoanAmount"].median(), color="#d62728", linestyle="--", label="Median")
        ax.set_xlabel("Loan amount ($1,000s)")
        ax.legend()
        st.pyplot(fig)
    with c2:
        st.markdown("#### Notebook imputation rules")
        st.markdown(
            "- Categorical variables: **mode**\n"
            "- `Loan_Amount_Term`: **mode** (usually 360 months)\n"
            "- `Credit_History`: **mode**\n"
            "- `LoanAmount`: **median** because of outliers"
        )

    if st.button("🧹 Impute missing values", type="primary"):
        try:
            st.session_state["imputed"] = impute_data(df)
            for key in ["raw", "processed", "x_train", "lrmodel", "dtmodel", "rfmodel"]:
                st.session_state.pop(key, None)
            st.success("Missing values were imputed using the notebook rules.")
        except PipelineError as exc:
            st.error(str(exc))

    if "imputed" in st.session_state:
        before_after = pd.DataFrame({
            "Before": df.isna().sum(), "After": st.session_state["imputed"].isna().sum()
        })
        st.dataframe(before_after, width="stretch")


with tabs[2]:
    st.subheader("Drop the ID, encode categories, and convert the target")
    st.markdown(
        "Notebook steps: drop `Loan_ID`, apply `pd.get_dummies(...)`, then replace "
        "`Loan_Approved` values with `Y → 1` and `N → 0`."
    )
    if st.button("⚙️ Prepare the dataset", type="primary"):
        try:
            raw, processed = prepare_data(df)
            st.session_state["raw"] = raw
            st.session_state["processed"] = processed
            for key in ["x_train", "lrmodel", "dtmodel", "rfmodel"]:
                st.session_state.pop(key, None)
            st.success("Dataset prepared and encoded.")
        except PipelineError as exc:
            st.error(str(exc))

    if "processed" in st.session_state:
        processed = st.session_state["processed"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Processed rows", processed.shape[0])
        c2.metric("Processed columns", processed.shape[1])
        c3.metric("Missing values", int(processed.isna().sum().sum()))
        st.dataframe(processed.head(), width="stretch")
        st.download_button(
            "⬇️ Download Processed_Credit_Dataset.csv",
            processed.to_csv(index=False).encode("utf-8"),
            "Processed_Credit_Dataset.csv", "text/csv",
        )
    else:
        st.info("Click **Prepare the dataset** to continue.")


with tabs[3]:
    st.subheader("Stratified train/test split and Min–Max normalization")
    if "processed" not in st.session_state:
        st.warning("Prepare the dataset in the previous tab first.")
    else:
        c1, c2 = st.columns(2)
        test_size = c1.slider("Test-set size", 0.10, 0.40, 0.20, 0.05)
        random_state = c2.number_input("Random state", 0, 10_000, 42)
        if st.button("✂️ Split and scale", type="primary"):
            store_split(st.session_state["processed"], test_size, int(random_state))
            st.success("Data split with `stratify=y`; scaler fitted only on training data.")

        if "x_train" in st.session_state:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("x_train", str(st.session_state["x_train"].shape))
            c2.metric("x_test", str(st.session_state["x_test"].shape))
            c3.metric("y_train", str(st.session_state["y_train"].shape))
            c4.metric("y_test", str(st.session_state["y_test"].shape))
            proportions = pd.DataFrame({
                "Train": st.session_state["y_train"].value_counts(normalize=True),
                "Test": st.session_state["y_test"].value_counts(normalize=True),
            }).rename(index={0: "Denied", 1: "Approved"})
            st.markdown("#### Class proportions preserved by stratification")
            st.dataframe(proportions.style.format("{:.1%}"), width="stretch")


with tabs[4]:
    st.subheader("Logistic Regression")
    if "x_train" not in st.session_state:
        st.warning("Complete **Split & Scale** first.")
    else:
        threshold = st.slider("Approval probability threshold", 0.30, 0.90, 0.50, 0.05)
        if st.button("📈 Train Logistic Regression", type="primary"):
            model = train_logistic(st.session_state["x_train_scaled"], st.session_state["y_train"])
            st.session_state["lrmodel"] = model
            st.success("Logistic Regression trained.")
        if "lrmodel" in st.session_state:
            metrics = evaluate(st.session_state["lrmodel"], st.session_state["x_test_scaled"],
                               st.session_state["y_test"], threshold)
            st.session_state["lr_metrics"] = metrics
            show_metrics(metrics)
            preview = pd.DataFrame({
                "Actual": st.session_state["y_test"].to_numpy()[:15],
                "Probability approved": metrics["probability"][:15],
                "Predicted": metrics["prediction"][:15],
            })
            st.dataframe(preview.style.format({"Probability approved": "{:.1%}"}), width="stretch")
            if threshold != 0.5:
                st.info("This reproduces the notebook's threshold experiment; increasing the threshold changes approvals and may reduce accuracy.")


with tabs[5]:
    st.subheader("Decision Tree Classifier")
    if "x_train" not in st.session_state:
        st.warning("Complete **Split & Scale** first.")
    else:
        c1, c2, c3 = st.columns(3)
        criterion = c1.selectbox("Criterion", ["gini", "entropy", "log_loss"], key="dt_criterion")
        depth = c2.slider("Max depth (0 = unlimited)", 0, 20, 0, key="dt_depth")
        min_split = c3.slider("Minimum samples to split", 2, 20, 2)
        if st.button("🌳 Train Decision Tree", type="primary"):
            model = train_decision_tree(
                st.session_state["x_train_scaled"], st.session_state["y_train"],
                criterion=criterion, max_depth=None if depth == 0 else depth,
                min_samples_split=min_split,
            )
            st.session_state["dtmodel"] = model
            st.session_state["dt_metrics"] = evaluate(
                model, st.session_state["x_test_scaled"], st.session_state["y_test"]
            )
            st.success("Decision Tree trained.")
        if "dtmodel" in st.session_state:
            show_metrics(st.session_state["dt_metrics"])
            params = st.session_state["dtmodel"].get_params()
            with st.expander("Tunable hyperparameters (`dtmodel.get_params()`)"):
                st.json(params)


with tabs[6]:
    st.subheader("Random Forest and hyperparameter tuning")
    if "x_train" not in st.session_state:
        st.warning("Complete **Split & Scale** first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        trees = c1.slider("n_estimators", 2, 500, 100, 2)
        depth = c2.slider("max_depth (0 = unlimited)", 0, 20, 0, key="rf_depth")
        feature_option = c3.selectbox("max_features", ["sqrt", "log2", "0.33", "10"])
        criterion = c4.selectbox("criterion", ["gini", "entropy", "log_loss"], key="rf_criterion")
        max_features = float(feature_option) if feature_option == "0.33" else (
            int(feature_option) if feature_option == "10" else feature_option
        )
        if st.button("🌲 Train Random Forest", type="primary"):
            model = train_random_forest(
                st.session_state["x_train_scaled"], st.session_state["y_train"],
                n_estimators=trees, criterion=criterion,
                max_depth=None if depth == 0 else depth, max_features=max_features,
            )
            st.session_state["rfmodel"] = model
            st.session_state["rf_metrics"] = evaluate(
                model, st.session_state["x_test_scaled"], st.session_state["y_test"]
            )
            st.success("Random Forest trained.")
        if "rfmodel" in st.session_state:
            show_metrics(st.session_state["rf_metrics"])
            importance = pd.DataFrame({
                "Feature": st.session_state["x_train"].columns,
                "Importance": st.session_state["rfmodel"].feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.markdown("#### Feature importance")
            st.bar_chart(importance.set_index("Feature"))
            st.dataframe(importance, width="stretch", hide_index=True)


with tabs[7]:
    st.subheader("Model comparison and cross-validation")
    trained = []
    for label, key, metric_key in [
        ("Logistic Regression", "lrmodel", "lr_metrics"),
        ("Decision Tree", "dtmodel", "dt_metrics"),
        ("Random Forest", "rfmodel", "rf_metrics"),
    ]:
        if key in st.session_state:
            m = st.session_state.get(metric_key) or evaluate(
                st.session_state[key], st.session_state["x_test_scaled"], st.session_state["y_test"]
            )
            trained.append({"Model": label, "Accuracy": m["accuracy"], "Precision": m["precision"],
                            "Recall": m["recall"], "F1": m["f1"]})
    if not trained:
        st.info("Train at least one model in the previous tabs.")
    else:
        comparison = pd.DataFrame(trained).sort_values("Accuracy", ascending=False)
        st.dataframe(comparison.style.format({c: "{:.1%}" for c in ["Accuracy", "Precision", "Recall", "F1"]}),
                     width="stretch", hide_index=True)
        st.bar_chart(comparison.set_index("Model")[["Accuracy", "F1"]])
        best = comparison.iloc[0]
        st.success(f"Best test-set accuracy: **{best['Model']} — {best['Accuracy']:.1%}**")

        st.markdown("#### Cross-validation")
        c1, c2 = st.columns(2)
        folds = c1.slider("Number of folds", 3, 10, 5)
        stratified = c2.checkbox("Use StratifiedKFold", value=False,
                                 help="Preserves class proportions in every fold.")
        if st.button("🔁 Run cross-validation"):
            cv_rows = []
            for label, key in [("Logistic Regression", "lrmodel"), ("Decision Tree", "dtmodel"),
                               ("Random Forest", "rfmodel")]:
                if key in st.session_state:
                    scores = cross_validate(st.session_state[key], st.session_state["x_train_scaled"],
                                            st.session_state["y_train"], folds, stratified)
                    cv_rows.append({"Model": label, "Fold scores": ", ".join(f"{s:.3f}" for s in scores),
                                    "Mean accuracy": scores.mean(), "Std. deviation": scores.std()})
            cv_df = pd.DataFrame(cv_rows)
            st.dataframe(cv_df.style.format({"Mean accuracy": "{:.1%}", "Std. deviation": "{:.3f}"}),
                         width="stretch", hide_index=True)


with tabs[8]:
    st.subheader("Save, load, and make a live loan-eligibility prediction")
    models = {}
    for label, key in [("Logistic Regression", "lrmodel"), ("Decision Tree", "dtmodel"),
                       ("Random Forest", "rfmodel")]:
        if key in st.session_state:
            models[label] = st.session_state[key]

    if not models and "loaded_bundle" not in st.session_state:
        st.warning("Train a model first, or upload a previously saved model bundle.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 💾 Save a trained model")
            if models:
                choice = st.selectbox("Model to save", list(models))
                bundle = ModelBundle(
                    choice, models[choice], st.session_state["scaler"],
                    st.session_state["x_train"].columns.tolist(),
                    default_raw_values(st.session_state["raw"]),
                )
                st.download_button("⬇️ Download Loan_Eligibility_Model.pkl", pickle_bundle(bundle),
                                   "Loan_Eligibility_Model.pkl", "application/octet-stream")
        with c2:
            st.markdown("#### 📂 Load a saved model")
            uploaded_model = st.file_uploader("Upload a .pkl model bundle", type=["pkl"], key="model_upload")
            if uploaded_model is not None:
                try:
                    st.session_state["loaded_bundle"] = unpickle_bundle(uploaded_model.read())
                    st.success(f"Loaded {st.session_state['loaded_bundle'].model_name}.")
                except PipelineError as exc:
                    st.error(str(exc))

        available = []
        if models:
            available.extend(models.keys())
        if "loaded_bundle" in st.session_state:
            available.append("Uploaded model bundle")
        selected = st.radio("Predict using", available, horizontal=True)
        if selected == "Uploaded model bundle":
            active_bundle = st.session_state["loaded_bundle"]
        else:
            active_bundle = ModelBundle(
                selected, models[selected], st.session_state["scaler"],
                st.session_state["x_train"].columns.tolist(),
                default_raw_values(st.session_state["raw"]),
            )

        defaults = active_bundle.raw_defaults
        with st.form("loan_application"):
            st.markdown("#### 🔮 Applicant information")
            a, b, c = st.columns(3)
            values = {
                "Gender": a.selectbox("Gender", ["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1),
                "Married": b.selectbox("Married", ["Yes", "No"], index=0 if defaults["Married"] == "Yes" else 1),
                "Dependents": c.selectbox("Dependents", ["0", "1", "2", "3+"],
                                          index=["0", "1", "2", "3+"].index(str(defaults["Dependents"]))),
                "Education": a.selectbox("Education", ["Graduate", "Not Graduate"]),
                "Self_Employed": b.selectbox("Self-employed", ["No", "Yes"]),
                "Property_Area": c.selectbox("Property area", ["Rural", "Semiurban", "Urban"]),
                "ApplicantIncome": a.number_input("Applicant monthly income", min_value=0.0,
                                                   value=float(defaults["ApplicantIncome"]), step=100.0),
                "CoapplicantIncome": b.number_input("Co-applicant monthly income", min_value=0.0,
                                                     value=float(defaults["CoapplicantIncome"]), step=100.0),
                "LoanAmount": c.number_input("Loan amount ($1,000s)", min_value=1.0,
                                             value=float(defaults["LoanAmount"]), step=1.0),
                "Loan_Amount_Term": a.selectbox("Loan term (months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8),
                "Credit_History": b.selectbox("Credit history", [1.0, 0.0],
                                              format_func=lambda x: "Good / available" if x == 1 else "No / poor"),
            }
            submitted = st.form_submit_button("Check eligibility", type="primary")
        if submitted:
            prediction, probability = predict_application(active_bundle, values)
            if prediction == 1:
                st.success(f"✅ Model prediction: **Eligible / likely approved** — probability {probability:.1%}")
            else:
                st.error(f"❌ Model prediction: **Not eligible / likely denied** — approval probability {probability:.1%}")
            st.caption("This is an educational model prediction, not a lending decision or financial advice.")
