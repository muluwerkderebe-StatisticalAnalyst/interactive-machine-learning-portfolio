"""Interactive, notebook-faithful Mall Customer Segmentation app."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.plotting import scatter_matrix

from src.clustering_pipeline import (
    FEATURES_2D, FEATURES_3D, ClusterBundle, PipelineError, cluster_diagnostics,
    cluster_profiles, fit_kmeans, load_data, pickle_bundle, predict_cluster,
    segment_names, unpickle_bundle,
)


SAMPLE_FILE = Path(__file__).resolve().parent.parent / "sample_data" / "mall_customers_clusteringData.csv"

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")
if st.session_state.get("_active_ml_page") != "mall_clustering":
    st.session_state.clear()
    st.session_state["_active_ml_page"] = "mall_clustering"
st.title("🛍️ Mall Customer Segmentation Model")
st.caption(
    "An interactive walkthrough of `Unsupervised_Clustering_Solution.ipynb`: explore customer "
    "behaviour, create K-Means segments, evaluate K with Elbow and Silhouette methods, tune "
    "hyperparameters, profile the clusters, and assign new customers to segments."
)


def reset_results():
    for key in [
        "model_2d", "clustered_2d", "diagnostics_2d", "model_3d", "clustered_3d",
        "diagnostics_3d", "tuned_model", "tuned_clustered", "loaded_bundle",
    ]:
        st.session_state.pop(key, None)


def plot_clusters_2d(clustered, model, title):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(
        clustered["Annual_Income"], clustered["Spending_Score"],
        c=clustered["Cluster"], cmap="tab10", s=48, alpha=0.82,
        edgecolor="white", linewidth=0.4,
    )
    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="X", s=220,
               label="Cluster centres")
    ax.set_xlabel("Annual income ($1,000s)")
    ax.set_ylabel("Spending score")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def show_diagnostic_charts(diagnostics, title_suffix=""):
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(diagnostics["cluster"], diagnostics["WCSS_Score"], marker="o", color="#1f77b4")
        ax.set_title(f"Elbow Plot{title_suffix}")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("WCSS")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(diagnostics["cluster"], diagnostics["Silhouette_Score"], marker="o", color="#d62728")
        best = diagnostics.loc[diagnostics["Silhouette_Score"].idxmax()]
        ax.scatter([best["cluster"]], [best["Silhouette_Score"]], color="black", s=90, zorder=5)
        ax.set_title(f"Silhouette Plot{title_suffix}")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette score")
        ax.grid(alpha=0.3)
        st.pyplot(fig)


st.subheader("📥 Load the mall-customer dataset")
uploaded = st.file_uploader(
    "Upload a mall-customer CSV, or leave empty to use the bundled 200-customer dataset",
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
    reset_results()
    st.session_state["_source"] = source_name
st.session_state["df"] = df


tabs = st.tabs([
    "1️⃣ Explore Data",
    "2️⃣ Relationships",
    "3️⃣ Two-Feature K-Means",
    "4️⃣ Elbow & Silhouette",
    "5️⃣ Three-Feature Model",
    "6️⃣ Tune K-Means",
    "7️⃣ Profile Segments",
    "8️⃣ Save & Segment",
])


with tabs[0]:
    st.subheader("Explore the original customer dataset")
    st.dataframe(df.head(), width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Average age", f"{df['Age'].mean():.1f}")
    c4.metric("Average spending score", f"{df['Spending_Score'].mean():.1f}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Summary statistics")
        st.dataframe(df.describe().transpose(), width="stretch")
    with c2:
        st.markdown("#### Gender distribution")
        st.bar_chart(df["Gender"].value_counts())
        st.markdown("#### Missing values")
        st.dataframe(df.isna().sum().rename("Missing").to_frame(), width="stretch")
    with st.expander("Data dictionary"):
        dictionary = [
            ("Customer_ID", "Unique customer identifier"),
            ("Gender", "Customer gender"),
            ("Age", "Customer age"),
            ("Annual_Income", "Annual income in $1,000s"),
            ("Spending_Score", "Mall-assigned spending score from 1 to 100"),
        ]
        st.dataframe(pd.DataFrame(dictionary, columns=["Variable", "Meaning"]), hide_index=True,
                     width="stretch")


with tabs[1]:
    st.subheader("Correlation and pairwise relationships")
    method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)
    corr = df.select_dtypes("number").corr(method=method.lower())
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1), width="stretch")
    st.caption(
        "The notebook observes that spending tends to be higher among many customers aged 20–40 "
        "and relatively lower beyond age 40. Customer_ID is an identifier and is not used for clustering."
    )
    if st.checkbox("Show pairwise scatter-matrix", value=True):
        fig = plt.figure(figsize=(9, 9))
        scatter_matrix(
            df[["Age", "Annual_Income", "Spending_Score"]], figsize=(9, 9),
            diagonal="hist", color="#1f77b4", alpha=0.65, ax=None,
        )
        st.pyplot(plt.gcf())
        plt.close("all")


with tabs[2]:
    st.subheader("K-Means using Annual Income and Spending Score")
    st.markdown(
        "This recreates the notebook's visual models with **3 clusters** and then **5 clusters**. "
        "Choose either value—or experiment with another K."
    )
    c1, c2 = st.columns(2)
    n_clusters = c1.slider("Number of clusters (K)", 2, 10, 5)
    seed = c2.number_input("Random state", 0, 10_000, 42, key="basic_seed")
    if st.button("🎯 Fit two-feature K-Means", type="primary"):
        model, clustered = fit_kmeans(
            df, FEATURES_2D, n_clusters=n_clusters, random_state=int(seed)
        )
        st.session_state["model_2d"] = model
        st.session_state["clustered_2d"] = clustered
        st.success(f"K-Means fitted with K={n_clusters}.")

    if "model_2d" in st.session_state:
        model = st.session_state["model_2d"]
        clustered = st.session_state["clustered_2d"]
        st.pyplot(plot_clusters_2d(clustered, model, "Mall Customer Segments"))
        c1, c2 = st.columns(2)
        with c1:
            centers = pd.DataFrame(model.cluster_centers_, columns=FEATURES_2D)
            centers.index.name = "Cluster"
            st.markdown("#### Cluster centres")
            st.dataframe(centers, width="stretch")
        with c2:
            st.markdown("#### Customers per cluster")
            st.dataframe(clustered["Cluster"].value_counts().sort_index().rename("Customers").to_frame(),
                         width="stretch")
        st.download_button(
            "⬇️ Download customers with cluster labels",
            clustered.to_csv(index=False).encode("utf-8"),
            "Mall_Customers_with_Clusters.csv", "text/csv",
        )


with tabs[3]:
    st.subheader("Find the optimal K with Elbow and Silhouette methods")
    st.markdown(
        "The notebook tests K from **3 to 8**. WCSS should decline, while a Silhouette score "
        "closer to +1 indicates better-separated clusters."
    )
    c1, c2 = st.columns(2)
    min_k = c1.slider("Minimum K", 2, 8, 3)
    max_k = c2.slider("Maximum K", min_k + 1, 12, max(8, min_k + 1))
    if st.button("📐 Calculate two-feature diagnostics", type="primary"):
        diagnostics = cluster_diagnostics(df, FEATURES_2D, range(min_k, max_k + 1))
        st.session_state["diagnostics_2d"] = diagnostics
    if "diagnostics_2d" in st.session_state:
        diagnostics = st.session_state["diagnostics_2d"]
        st.dataframe(diagnostics.style.format({"WCSS_Score": "{:,.2f}", "Silhouette_Score": "{:.4f}"}),
                     width="stretch", hide_index=True)
        show_diagnostic_charts(diagnostics)
        best = diagnostics.loc[diagnostics["Silhouette_Score"].idxmax()]
        st.success(
            f"Best Silhouette result: **K={int(best['cluster'])}** with score "
            f"**{best['Silhouette_Score']:.4f}**. The notebook concludes that K=5 is optimal for two features."
        )


with tabs[4]:
    st.subheader("Use Age, Annual Income, and Spending Score")
    st.markdown(
        "With three features the clusters cannot be represented fully on a flat 2D chart, so the "
        "notebook compares Elbow and Silhouette scores and concludes that **K=6** is optimal."
    )
    if st.button(" Analyze K=3 through K=8 with all three features", type="primary"):
        st.session_state["diagnostics_3d"] = cluster_diagnostics(df, FEATURES_3D, range(3, 9))
    if "diagnostics_3d" in st.session_state:
        diagnostics = st.session_state["diagnostics_3d"]
        st.dataframe(diagnostics.style.format({"WCSS_Score": "{:,.2f}", "Silhouette_Score": "{:.4f}"}),
                     width="stretch", hide_index=True)
        show_diagnostic_charts(diagnostics, " — Three Features")
        best = diagnostics.loc[diagnostics["Silhouette_Score"].idxmax()]
        st.info(f"Highest three-feature Silhouette score occurs at **K={int(best['cluster'])}**.")

    k3 = st.slider("Fit a three-feature model with K", 2, 10, 6)
    if st.button(" Fit three-feature K-Means"):
        model, clustered = fit_kmeans(df, FEATURES_3D, k3)
        st.session_state["model_3d"] = model
        st.session_state["clustered_3d"] = clustered
    if "model_3d" in st.session_state:
        model = st.session_state["model_3d"]
        clustered = st.session_state["clustered_3d"]
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
        points = ax.scatter(clustered["Age"], clustered["Annual_Income"],
                            clustered["Spending_Score"], c=clustered["Cluster"],
                            cmap="tab10", s=45, alpha=0.8)
        centers = model.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="black", marker="X", s=180)
        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Income")
        ax.set_zlabel("Spending Score")
        ax.set_title("Three-Feature Customer Segments")
        fig.colorbar(points, ax=ax, label="Cluster", shrink=0.7)
        st.pyplot(fig)


with tabs[5]:
    st.subheader("Tune K-Means hyperparameters")
    st.markdown(
        "The notebook exercise uses `init='k-means++'`, `n_init='auto'`, and `max_iter=10`. "
        "This tab makes those settings interactive."
    )
    c1, c2, c3, c4 = st.columns(4)
    tuned_k = c1.slider("n_clusters", 2, 10, 5, key="tuned_k")
    init = c2.selectbox("init", ["k-means++", "random"])
    n_init = c3.slider("n_init", 1, 30, 10)
    max_iter = c4.slider("max_iter", 10, 500, 300, 10)
    feature_set = st.radio(
        "Features", ["Income + Spending (visual)", "Age + Income + Spending"], horizontal=True
    )
    selected_features = FEATURES_2D if feature_set.startswith("Income") else FEATURES_3D
    if st.button("🛠️ Train tuned model", type="primary"):
        model, clustered = fit_kmeans(
            df, selected_features, tuned_k, init=init, n_init=n_init, max_iter=max_iter
        )
        st.session_state["tuned_model"] = model
        st.session_state["tuned_clustered"] = clustered
        st.session_state["tuned_features"] = selected_features
        score = cluster_diagnostics(
            df, selected_features, [tuned_k], init=init, n_init=n_init, max_iter=max_iter
        ).iloc[0]
        st.session_state["tuned_score"] = score
        st.success("Tuned K-Means model trained.")
    if "tuned_model" in st.session_state:
        score = st.session_state["tuned_score"]
        c1, c2 = st.columns(2)
        c1.metric("WCSS", f"{score['WCSS_Score']:,.2f}")
        c2.metric("Silhouette score", f"{score['Silhouette_Score']:.4f}")
        if st.session_state["tuned_features"] == FEATURES_2D:
            st.pyplot(plot_clusters_2d(st.session_state["tuned_clustered"],
                                       st.session_state["tuned_model"], "Tuned K-Means Segments"))
        st.markdown("**`fit_predict(X)` is equivalent to calling `fit(X)` and then reading or predicting the labels.**")


with tabs[6]:
    st.subheader("Profile and interpret customer segments")
    available = {}
    if "model_2d" in st.session_state:
        available["Two-feature model"] = (st.session_state["model_2d"], st.session_state["clustered_2d"], FEATURES_2D)
    if "model_3d" in st.session_state:
        available["Three-feature model"] = (st.session_state["model_3d"], st.session_state["clustered_3d"], FEATURES_3D)
    if "tuned_model" in st.session_state:
        available["Tuned model"] = (st.session_state["tuned_model"], st.session_state["tuned_clustered"], st.session_state["tuned_features"])
    if not available:
        st.info("Train at least one K-Means model in the previous tabs.")
    else:
        selected = st.selectbox("Model to profile", list(available))
        model, clustered, features = available[selected]
        profile = cluster_profiles(clustered)
        names = segment_names(profile)
        profile.insert(1, "Suggested_Segment", profile["Cluster"].map(names))
        st.dataframe(
            profile.style.format({"Average_Age": "{:.1f}", "Average_Income": "{:.1f}",
                                  "Average_Spending_Score": "{:.1f}", "Female_Percent": "{:.1f}%"}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Segment names are business-friendly interpretations based on each cluster's average "
            "income and spending score; cluster numbers themselves have no natural ranking."
        )
        st.bar_chart(profile.set_index("Suggested_Segment")[["Customers"]])


with tabs[7]:
    st.subheader("Save a model and assign a new customer to a segment")
    models = {}
    if "model_2d" in st.session_state:
        models["Two-feature model"] = ClusterBundle(st.session_state["model_2d"], FEATURES_2D)
    if "model_3d" in st.session_state:
        models["Three-feature model"] = ClusterBundle(st.session_state["model_3d"], FEATURES_3D)
    if "tuned_model" in st.session_state:
        models["Tuned model"] = ClusterBundle(st.session_state["tuned_model"], st.session_state["tuned_features"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💾 Save a trained model")
        if models:
            save_choice = st.selectbox("Model to save", list(models), key="save_choice")
            st.download_button(
                "⬇️ Download Mall_Customer_KMeans.pkl",
                pickle_bundle(models[save_choice]), "Mall_Customer_KMeans.pkl",
                "application/octet-stream",
            )
        else:
            st.info("Train a model before saving it.")
    with c2:
        st.markdown("####  Load a saved model")
        uploaded_model = st.file_uploader("Upload a .pkl cluster bundle", type=["pkl"], key="pkl_upload")
        if uploaded_model is not None:
            try:
                st.session_state["loaded_bundle"] = unpickle_bundle(uploaded_model.read())
                st.success("Saved K-Means model loaded.")
            except PipelineError as exc:
                st.error(str(exc))

    prediction_models = dict(models)
    if "loaded_bundle" in st.session_state:
        prediction_models["Uploaded model"] = st.session_state["loaded_bundle"]
    if prediction_models:
        selected = st.radio("Assign using", list(prediction_models), horizontal=True)
        bundle = prediction_models[selected]
        with st.form("new_customer"):
            st.markdown("#### 👤 New customer information")
            values = {}
            cols = st.columns(len(bundle.feature_names))
            defaults = {"Age": float(df["Age"].median()),
                        "Annual_Income": float(df["Annual_Income"].median()),
                        "Spending_Score": float(df["Spending_Score"].median())}
            for col, feature in zip(cols, bundle.feature_names):
                values[feature] = col.number_input(feature.replace("_", " "), min_value=0.0,
                                                   value=defaults[feature], step=1.0)
            submitted = st.form_submit_button("Assign customer segment", type="primary")
        if submitted:
            cluster = predict_cluster(bundle, values)
            st.success(f"The new customer belongs to **Cluster {cluster}**.")
            st.caption("Cluster labels are model-specific and should be interpreted using the segment profile table.")
