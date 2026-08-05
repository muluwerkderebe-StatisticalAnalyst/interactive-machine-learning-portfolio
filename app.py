"""Landing dashboard for the four-page machine-learning portfolio."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Interactive Machine Learning Applications",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if st.session_state.get("_active_ml_page") != "home":
    st.session_state.clear()
    st.session_state["_active_ml_page"] = "home"

st.markdown(
    """
    <style>
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 1.2rem;
        background: linear-gradient(125deg, #0b1f3a 0%, #123f66 58%, #0f766e 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 28px rgba(11,31,58,.18);
    }
    .hero h1 { margin: 0 0 .55rem 0; font-size: 2.45rem; }
    .hero p { margin: 0; font-size: 1.08rem; opacity: .92; max-width: 900px; }
    .project-card {
        min-height: 180px;
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(49, 68, 89, .18);
        border-radius: 1rem;
        background: rgba(255,255,255,.72);
        box-shadow: 0 5px 18px rgba(11,31,58,.07);
        margin-bottom: .75rem;
    }
    .project-card h3 { margin-top: 0; color: #123f66; }
    .project-card p { min-height: 66px; }
    .tag {
        display: inline-block; padding: .18rem .55rem; margin: .12rem .12rem .12rem 0;
        border-radius: 999px; background: #e7f5f3; color: #0f5f59; font-size: .78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <h1>Interactive Machine Learning Applications</h1>
      <p>Four complete, machine-learning applications in one Streamlit platform covering regression, classification, unsupervised clustering, and neural networks.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

datasets = {
    "Real Estate": ROOT / "sample_data" / "final_realestateData.csv",
    "Loan Eligibility": ROOT / "sample_data" / "credit_loanData.csv",
    "Mall Customers": ROOT / "sample_data" / "mall_customers_clusteringData.csv",
    "UCLA Admission": ROOT / "sample_data" / "Admission_NNData.csv",
}

c1, c2, c3, c4 = st.columns(4)
metrics = []
for name, path in datasets.items():
    frame = pd.read_csv(path)
    metrics.append((name, len(frame), len(frame.columns)))
for col, (name, rows, columns) in zip([c1, c2, c3, c4], metrics):
    col.metric(name, f"{rows:,} records", f"{columns} variables", delta_color="off")

st.markdown("## Choose a machine-learning application")

left, right = st.columns(2)
with left:
    st.markdown(
        """
        <div class="project-card">
          <h3> Real Estate Price Prediction</h3>
          <p>Explore property data, create a stratified split, train Linear Regression and Random Forest models, compare MAE, save a model, and predict a property price.</p>
          <span class="tag">Regression</span><span class="tag">Random Forest</span><span class="tag">Model Persistence</span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.page_link("pages/1_Real_Estate_Prediction.py", label="Open Real Estate Application", icon="🏠")

    st.markdown(
        """
        <div class="project-card">
          <h3> Mall Customer Segmentation</h3>
          <p>Build K-Means segments, evaluate K with Elbow and Silhouette methods, visualize 2D/3D clusters, profile customer groups, and segment a new customer.</p>
          <span class="tag">Unsupervised Learning</span><span class="tag">K-Means</span><span class="tag">Segmentation</span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.page_link("pages/3_Mall_Customer_Clustering.py", label="Open Customer Clustering Application", icon="🛍️")

with right:
    st.markdown(
        """
        <div class="project-card">
          <h3> Loan Eligibility Prediction</h3>
          <p>Impute and encode credit data, train Logistic Regression, Decision Tree and Random Forest classifiers, compare metrics, cross-validate, and score a new loan application.</p>
          <span class="tag">Classification</span><span class="tag">Cross-Validation</span><span class="tag">Feature Importance</span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.page_link("pages/2_Loan_Eligibility.py", label="Open Loan Eligibility Application", icon="🏦")

    st.markdown(
        """
        <div class="project-card">
          <h3> UCLA Admission Neural Network</h3>
          <p>Prepare admission data, prevent scaling leakage, build and tune a multilayer perceptron, inspect its loss curve and confusion matrix, and predict an applicant's class.</p>
          <span class="tag">Neural Network</span><span class="tag">MLPClassifier</span><span class="tag">Hyperparameter Tuning</span>
        </div>
        """, unsafe_allow_html=True,
    )
    st.page_link("pages/4_UCLA_Admission_Neural_Network.py", label="Open UCLA Neural Network Application", icon="🎓")

st.divider()
st.markdown("## Platform coverage")
coverage = pd.DataFrame([
    ["Real Estate", "Regression", "Price", "Linear Regression; Random Forest", "MAE"],
    ["Loan Eligibility", "Classification", "Approved / Denied", "Logistic Regression; Decision Tree; Random Forest", "Accuracy, Precision, Recall, F1"],
    ["Mall Customers", "Clustering", "Customer segment", "K-Means", "WCSS, Silhouette"],
    ["UCLA Admission", "Neural-network classification", "Admission class", "MLPClassifier", "Accuracy, Precision, Recall, F1"],
], columns=["Application", "ML Task", "Output", "Models", "Evaluation"])
st.dataframe(coverage, width="stretch", hide_index=True)

with st.expander("How to use this platform"):
    st.markdown(
        """
        1. Select a page from the sidebar or one of the links above.
        2. Use the bundled dataset or upload a CSV with the same schema.
        3. Complete the tabs from left to right, following the original notebook sequence.
        4. Train and compare models interactively.
        5. Download processed datasets and serialized models where available.
        6. Use the final tab on each page for a live prediction or cluster assignment.
        """
    )

st.caption("Educational machine-learning portfolio. Predictions are demonstrations and should not be treated as financial, lending, or university decisions.")

