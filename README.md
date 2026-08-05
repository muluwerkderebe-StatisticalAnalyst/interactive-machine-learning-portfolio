# Interactive Machine Learning Portfolio — Four-Page Streamlit App

This full-scale multipage Streamlit application combines four notebook-faithful
machine-learning projects into one platform:

<a href="https://interactive-machine-learning-portfolio-fwya8x8sb7jckck63w9epu.streamlit.app" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Open-Streamlit%20Machine%20Learning%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Open Streamlit Machine Learning App">
</a>

1. Real Estate Price Prediction
2. Loan Eligibility Prediction
3. Mall Customer Segmentation
4. UCLA Admission Neural Network

Each project keeps its own dataset, pipeline, interactive training workflow,
model evaluation, downloadable outputs, and live prediction or segmentation form.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the landing dashboard. Use the sidebar to navigate among
the four ML applications.

## Application pages

| Page | ML task | Main models and methods | Final interaction |
|---|---|---|---|
| Real Estate | Regression | Linear Regression, Random Forest, MAE | Predict property price |
| Loan Eligibility | Classification | Logistic Regression, Decision Tree, Random Forest, CV | Predict eligibility |
| Mall Customers | Unsupervised clustering | K-Means, Elbow, Silhouette, profiling | Assign customer segment |
| UCLA Admission | Neural-network classification | MLPClassifier, MinMax scaling, tuning | Predict admission class |

## Project layout

```text
ML_Portfolio_Streamlit_App/
├── app.py
├── pages/
│   ├── 1_Real_Estate_Prediction.py
│   ├── 2_Loan_Eligibility.py
│   ├── 3_Mall_Customer_Clustering.py
│   └── 4_UCLA_Admission_Neural_Network.py
├── src/
│   ├── real_estate_pipeline.py
│   ├── loan_pipeline.py
│   ├── clustering_pipeline.py
│   └── admission_pipeline.py
├── sample_data/
│   ├── final_realestateData.csv
│   ├── credit_loanData.csv
│   ├── mall_customers_clusteringData.csv
│   └── Admission_NNData.csv
├── notebooks/
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Deploy on Streamlit Community Cloud

1. Upload this project folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and choose the repository.
4. Set the main file path to `app.py`.
5. Choose a supported Python version such as Python 3.12.
6. Deploy the app. Streamlit installs the packages from `requirements.txt`.

## Design and reliability notes

- Each page imports its own dedicated pipeline module.
- Session state is isolated when users switch projects, preventing models or
  datasets from one page from leaking into another page.
- All bundled datasets are addressed with project-relative paths, so local and
  cloud execution use the same structure.
- All four applications support uploaded replacement datasets with schema checks.
- TensorFlow is not required; the UCLA project follows the notebook and uses
  scikit-learn's `MLPClassifier`.
- Predictions are educational demonstrations, not financial, lending, or
  university decisions.
