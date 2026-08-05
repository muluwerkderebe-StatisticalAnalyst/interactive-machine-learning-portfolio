"""Interactive, notebook-faithful UCLA admission neural-network app."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.admission_pipeline import (
    AdmissionBundle, PipelineError, classify_target, evaluate, load_data,
    pickle_bundle, predict_applicant, prepare_data, split_and_scale,
    train_mlp, unpickle_bundle,
)


SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample_data" / "Admission_NNData.csv"
SUCCESS_TARGET = 0.90

st.set_page_config(page_title="UCLA Admission Neural Network", page_icon="🎓", layout="wide")
if st.session_state.get("_active_ml_page") != "ucla_admission":
    st.session_state.clear()
    st.session_state["_active_ml_page"] = "ucla_admission"
st.title("🎓 Predicting Chances of Admission at UCLA")
st.caption(
    "An interactive walkthrough of `UCLA_Neural_Networks_Solution.ipynb`: convert admission "
    "chance into a classification target, prepare and scale the data, build and tune an MLP "
    "neural network, evaluate performance, save the model, and predict a new applicant."
)


def reset_downstream():
    for key in [
        "classified", "prepared", "clean", "x_train", "x_test", "y_train", "y_test",
        "x_train_scaled", "x_test_scaled", "scaler", "original_model", "original_train_metrics",
        "original_test_metrics", "tuned_model", "tuned_train_metrics", "tuned_test_metrics",
        "loaded_bundle",
    ]:
        st.session_state.pop(key, None)


def store_split(clean, test_size, random_state):
    values = split_and_scale(clean, test_size=test_size, random_state=random_state)
    names = ["x_train", "x_test", "y_train", "y_test", "x_train_scaled", "x_test_scaled", "scaler"]
    for name, value in zip(names, values):
        st.session_state[name] = value
    for key in ["original_model", "original_train_metrics", "original_test_metrics",
                "tuned_model", "tuned_train_metrics", "tuned_test_metrics"]:
        st.session_state.pop(key, None)


def show_metrics(metrics, prefix=""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric(f"{prefix}Precision", f"{metrics['precision']:.1%}")
    c3.metric(f"{prefix}Recall", f"{metrics['recall']:.1%}")
    c4.metric(f"{prefix}F1 score", f"{metrics['f1']:.1%}")


def show_confusion_matrix(metrics):
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual: Not admitted", "Actual: Admitted"],
        columns=["Predicted: Not admitted", "Predicted: Admitted"],
    )
    st.dataframe(cm, width="stretch")


def plot_loss(model, title="Neural Network Loss Curve"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(model.loss_curve_, color="#1f77b4", linewidth=2, label="Training loss")
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


st.subheader(" Load the admission dataset")
uploaded = st.file_uploader(
    "Upload an admission CSV, or leave empty to use the bundled 500-applicant dataset",
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
    st.session_state["threshold"] = 0.8
st.session_state["df"] = df


tabs = st.tabs([
    "1️⃣ Explore Data",
    "2️⃣ Create Target",
    "3️⃣ Visualize Patterns",
    "4️⃣ Prepare & Encode",
    "5️⃣ Split & Scale",
    "6️⃣ Build Neural Network",
    "7️⃣ Tune & Compare",
    "8️⃣ Save & Predict",
])


with tabs[0]:
    st.subheader("Explore the original UCLA admission dataset")
    st.dataframe(df.head(), width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{df.shape[0]:,}")
    c2.metric("Original columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Average admission chance", f"{df['Admit_Chance'].mean():.1%}")
    st.dataframe(df.describe().transpose(), width="stretch")
    with st.expander("Data dictionary"):
        dictionary = [
            ("GRE_Score", "GRE score out of 340"),
            ("TOEFL_Score", "TOEFL score out of 120"),
            ("University_Rating", "Bachelor's university rating from 1 to 5"),
            ("SOP", "Statement of Purpose strength from 1 to 5"),
            ("LOR", "Letter of Recommendation strength from 1 to 5"),
            ("CGPA", "Undergraduate GPA out of 10"),
            ("Research", "Research experience: 0 or 1"),
            ("Admit_Chance", "Original admission chance from 0 to 1"),
        ]
        st.dataframe(pd.DataFrame(dictionary, columns=["Variable", "Meaning"]),
                     hide_index=True, width="stretch")


with tabs[1]:
    st.subheader("Convert admission chance into a classification target")
    st.markdown(
        "The notebook uses a threshold of **80%**: `Admit_Chance ≥ 0.8 → 1 (admitted)`, "
        "otherwise `0 (not admitted)`."
    )
    threshold = st.slider("Admission-chance threshold", 0.50, 0.95, 0.80, 0.01)
    if st.button("🏷️ Create classification target", type="primary"):
        st.session_state["threshold"] = threshold
        st.session_state["classified"] = classify_target(df, threshold)
        for key in ["prepared", "clean", "x_train", "original_model", "tuned_model"]:
            st.session_state.pop(key, None)
        st.success(f"Target created using threshold {threshold:.0%}.")
    classified_preview = st.session_state.get("classified", classify_target(df, threshold))
    counts = classified_preview["Admit_Chance"].value_counts().sort_index().rename(
        index={0: "Not admitted", 1: "Admitted"}
    )
    c1, c2 = st.columns(2)
    c1.metric("Not admitted", int(counts.get("Not admitted", 0)))
    c2.metric("Admitted", int(counts.get("Admitted", 0)))
    st.bar_chart(counts)
    st.caption(f"Positive-class rate: **{classified_preview['Admit_Chance'].mean():.1%}**.")


with tabs[2]:
    st.subheader("Correlations and admission patterns")
    classified = st.session_state.get("classified", classify_target(df, st.session_state.get("threshold", 0.8)))
    corr = classified.drop(columns=["Serial_No"]).corr()
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), width="stretch")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    admitted = classified[classified["Admit_Chance"] == 1]
    not_admitted = classified[classified["Admit_Chance"] == 0]
    ax.scatter(not_admitted["GRE_Score"], not_admitted["TOEFL_Score"],
               alpha=0.65, label="Not admitted", color="#1f77b4")
    ax.scatter(admitted["GRE_Score"], admitted["TOEFL_Score"],
               alpha=0.75, label="Admitted", color="#ff7f0e")
    ax.set_xlabel("GRE Score")
    ax.set_ylabel("TOEFL Score")
    ax.set_title("GRE and TOEFL Scores by Admission Class")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    st.caption(
        "The notebook observes a strong linear relationship between GRE and TOEFL scores. "
        "Most applicants classified as admitted have GRE scores above roughly 320 and TOEFL scores above 105."
    )


with tabs[3]:
    st.subheader("Drop Serial_No and one-hot encode categorical variables")
    st.markdown(
        "Although `University_Rating` and `Research` are stored as numbers, the notebook treats "
        "them as categories and creates dummy variables."
    )
    if st.button("⚙️ Prepare and encode data", type="primary"):
        threshold = st.session_state.get("threshold", 0.8)
        prepared, clean = prepare_data(df, threshold)
        st.session_state["prepared"] = prepared
        st.session_state["clean"] = clean
        for key in ["x_train", "original_model", "tuned_model"]:
            st.session_state.pop(key, None)
        st.success("Serial number removed and categorical features encoded.")
    if "clean" in st.session_state:
        clean = st.session_state["clean"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", clean.shape[0])
        c2.metric("Encoded columns", clean.shape[1])
        c3.metric("Missing values", int(clean.isna().sum().sum()))
        st.dataframe(clean.head(), width="stretch")
        st.download_button("⬇️ Download processed admission data",
                           clean.to_csv(index=False).encode("utf-8"),
                           "Processed_Admission_Dataset.csv", "text/csv")
    else:
        st.info("Click **Prepare and encode data** to continue.")


with tabs[4]:
    st.subheader("Stratified split and leakage-safe Min–Max scaling")
    if "clean" not in st.session_state:
        st.warning("Complete **Prepare & Encode** first.")
    else:
        c1, c2 = st.columns(2)
        test_size = c1.slider("Test-set size", 0.10, 0.40, 0.20, 0.05)
        random_state = c2.number_input("Random state", 0, 10_000, 123)
        if st.button("✂️ Split and scale", type="primary"):
            store_split(st.session_state["clean"], test_size, int(random_state))
            st.success("Data split with `stratify=y`; MinMaxScaler fitted only on training data.")
        if "x_train" in st.session_state:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("x_train", str(st.session_state["x_train"].shape))
            c2.metric("x_test", str(st.session_state["x_test"].shape))
            c3.metric("y_train", str(st.session_state["y_train"].shape))
            c4.metric("y_test", str(st.session_state["y_test"].shape))
            scaled = pd.DataFrame(st.session_state["x_train_scaled"],
                                  columns=st.session_state["x_train"].columns)
            st.markdown("#### Scaled training data")
            st.dataframe(scaled.head(), width="stretch")
            with st.expander("Scaler minimum and maximum values"):
                scale_info = pd.DataFrame({
                    "Feature": st.session_state["x_train"].columns,
                    "Training minimum": st.session_state["scaler"].data_min_,
                    "Training maximum": st.session_state["scaler"].data_max_,
                })
                st.dataframe(scale_info, width="stretch", hide_index=True)
            fig, axes = plt.subplots(2, 2, figsize=(9, 6))
            axes[0, 0].hist(st.session_state["x_train"]["GRE_Score"], bins=20, color="#1f77b4")
            axes[0, 0].set_title("GRE before scaling")
            axes[0, 1].hist(scaled["GRE_Score"], bins=20, color="#2ca02c")
            axes[0, 1].set_title("GRE after scaling")
            axes[1, 0].hist(st.session_state["x_train"]["TOEFL_Score"], bins=20, color="#1f77b4")
            axes[1, 0].set_title("TOEFL before scaling")
            axes[1, 1].hist(scaled["TOEFL_Score"], bins=20, color="#2ca02c")
            axes[1, 1].set_title("TOEFL after scaling")
            fig.tight_layout()
            st.pyplot(fig)


with tabs[5]:
    st.subheader("Build the notebook neural network")
    st.markdown(
        "Notebook model: `MLPClassifier(hidden_layer_sizes=(3,4), batch_size=50, "
        "max_iter=200, random_state=123)`. The tuple `(3,4)` creates **two hidden layers** "
        "with 3 and 4 neurons, even though the notebook text says one hidden layer."
    )
    if "x_train" not in st.session_state:
        st.warning("Complete **Split & Scale** first.")
    else:
        if st.button(" Train original neural network", type="primary"):
            model = train_mlp(st.session_state["x_train_scaled"], st.session_state["y_train"])
            st.session_state["original_model"] = model
            st.session_state["original_train_metrics"] = evaluate(
                model, st.session_state["x_train_scaled"], st.session_state["y_train"]
            )
            st.session_state["original_test_metrics"] = evaluate(
                model, st.session_state["x_test_scaled"], st.session_state["y_test"]
            )
            st.success("Original MLP trained.")
        if "original_model" in st.session_state:
            st.markdown("#### Training performance")
            show_metrics(st.session_state["original_train_metrics"], "Train ")
            st.markdown("#### Testing performance")
            show_metrics(st.session_state["original_test_metrics"], "Test ")
            test_accuracy = st.session_state["original_test_metrics"]["accuracy"]
            if test_accuracy >= SUCCESS_TARGET:
                st.success(f"The model meets the 90% success criterion ({test_accuracy:.1%}).")
            else:
                st.warning(f"The model is below the 90% success criterion ({test_accuracy:.1%}); tune it in the next tab.")
            show_confusion_matrix(st.session_state["original_test_metrics"])
            st.pyplot(plot_loss(st.session_state["original_model"]))
            architecture = pd.DataFrame({
                "Layer": ["Input", "Hidden 1", "Hidden 2", "Output"],
                "Neurons": [st.session_state["x_train"].shape[1], 3, 4, 1],
                "Purpose": ["Encoded applicant features", "Learn patterns", "Refine patterns", "Admission class"],
            })
            st.dataframe(architecture, width="stretch", hide_index=True)


with tabs[6]:
    st.subheader("Tune and compare neural-network architectures")
    if "x_train" not in st.session_state:
        st.warning("Complete **Split & Scale** first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        architecture_text = c1.text_input("Hidden layers", "3,4",
                                          help="Comma-separated neurons, e.g. 3,4 or 16,8.")
        activation = c2.selectbox("Activation", ["relu", "tanh", "logistic", "identity"])
        batch_size = c3.slider("Batch size", 10, 200, 50, 10)
        max_iter = c4.slider("Maximum iterations", 50, 1000, 200, 50)
        c1, c2, c3 = st.columns(3)
        learning_rate = c1.number_input("Learning rate", 0.0001, 0.1000, 0.0010, 0.0001,
                                        format="%.4f")
        alpha = c2.number_input("L2 regularization (alpha)", 0.0000, 1.0000, 0.0001, 0.0001,
                                format="%.4f")
        early_stopping = c3.checkbox("Early stopping", value=False)
        if st.button("🛠️ Train tuned neural network", type="primary"):
            try:
                hidden_layers = tuple(int(x.strip()) for x in architecture_text.split(",") if x.strip())
                if not hidden_layers or any(x <= 0 for x in hidden_layers):
                    raise ValueError
                model = train_mlp(
                    st.session_state["x_train_scaled"], st.session_state["y_train"],
                    hidden_layer_sizes=hidden_layers, activation=activation,
                    batch_size=batch_size, max_iter=max_iter,
                    learning_rate_init=learning_rate, alpha=alpha,
                    early_stopping=early_stopping,
                )
                st.session_state["tuned_model"] = model
                st.session_state["tuned_train_metrics"] = evaluate(
                    model, st.session_state["x_train_scaled"], st.session_state["y_train"]
                )
                st.session_state["tuned_test_metrics"] = evaluate(
                    model, st.session_state["x_test_scaled"], st.session_state["y_test"]
                )
                st.success("Tuned MLP trained.")
            except ValueError:
                st.error("Enter positive whole numbers separated by commas, such as `3,4` or `16,8`.")
        if "tuned_model" in st.session_state:
            show_metrics(st.session_state["tuned_test_metrics"], "Test ")
            show_confusion_matrix(st.session_state["tuned_test_metrics"])
            st.pyplot(plot_loss(st.session_state["tuned_model"], "Tuned Neural Network Loss Curve"))

        rows = []
        if "original_model" in st.session_state:
            rows.append({"Model": "Original MLP (3,4; ReLU)",
                         "Train Accuracy": st.session_state["original_train_metrics"]["accuracy"],
                         "Test Accuracy": st.session_state["original_test_metrics"]["accuracy"],
                         "Test F1": st.session_state["original_test_metrics"]["f1"]})
        if "tuned_model" in st.session_state:
            rows.append({"Model": "Tuned MLP",
                         "Train Accuracy": st.session_state["tuned_train_metrics"]["accuracy"],
                         "Test Accuracy": st.session_state["tuned_test_metrics"]["accuracy"],
                         "Test F1": st.session_state["tuned_test_metrics"]["f1"]})
        if rows:
            comparison = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)
            st.markdown("#### Model comparison")
            st.dataframe(comparison.style.format({"Train Accuracy": "{:.1%}", "Test Accuracy": "{:.1%}",
                                                  "Test F1": "{:.1%}"}),
                         width="stretch", hide_index=True)


with tabs[7]:
    st.subheader("Save a model and predict a new applicant")
    models = {}
    if "original_model" in st.session_state:
        models["Original MLP"] = st.session_state["original_model"]
    if "tuned_model" in st.session_state:
        models["Tuned MLP"] = st.session_state["tuned_model"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💾 Save a trained model")
        if models:
            save_choice = st.selectbox("Model to save", list(models))
            bundle = AdmissionBundle(
                models[save_choice], st.session_state["scaler"],
                st.session_state["x_train"].columns.tolist(),
                st.session_state.get("threshold", 0.8),
            )
            st.download_button("⬇️ Download UCLA_Admission_MLP.pkl", pickle_bundle(bundle),
                               "UCLA_Admission_MLP.pkl", "application/octet-stream")
        else:
            st.info("Train a neural network before saving it.")
    with c2:
        st.markdown("#### 📂 Load a saved model")
        uploaded_model = st.file_uploader("Upload a .pkl admission bundle", type=["pkl"], key="pkl_upload")
        if uploaded_model is not None:
            try:
                st.session_state["loaded_bundle"] = unpickle_bundle(uploaded_model.read())
                st.success("Admission neural-network model loaded.")
            except PipelineError as exc:
                st.error(str(exc))

    prediction_models = {}
    for name, model in models.items():
        prediction_models[name] = AdmissionBundle(
            model, st.session_state["scaler"], st.session_state["x_train"].columns.tolist(),
            st.session_state.get("threshold", 0.8),
        )
    if "loaded_bundle" in st.session_state:
        prediction_models["Uploaded model"] = st.session_state["loaded_bundle"]

    if prediction_models:
        selected = st.radio("Predict using", list(prediction_models), horizontal=True)
        bundle = prediction_models[selected]
        with st.form("applicant_form"):
            st.markdown("#### 👤 Applicant profile")
            c1, c2, c3 = st.columns(3)
            values = {
                "GRE_Score": c1.slider("GRE score", 260, 340, 317),
                "TOEFL_Score": c2.slider("TOEFL score", 80, 120, 107),
                "University_Rating": c3.select_slider("University rating", options=[1, 2, 3, 4, 5], value=3),
                "SOP": c1.select_slider("SOP strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.5),
                "LOR": c2.select_slider("LOR strength", options=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.5),
                "CGPA": c3.slider("CGPA", 6.0, 10.0, 8.6, 0.01),
                "Research": c1.selectbox("Research experience", [1, 0],
                                         format_func=lambda x: "Yes" if x == 1 else "No"),
            }
            submitted = st.form_submit_button("Predict admission classification", type="primary")
        if submitted:
            prediction, probability = predict_applicant(bundle, values)
            if prediction == 1:
                st.success(f"✅ Model classification: **Likely admitted** — probability {probability:.1%}")
            else:
                st.error(f"❌ Model classification: **Not likely admitted** — admission probability {probability:.1%}")
            st.caption(
                f"The model predicts the notebook's binary class created from the original "
                f"Admit_Chance threshold of {bundle.threshold:.0%}. This is an educational estimate, not a UCLA decision."
            )
