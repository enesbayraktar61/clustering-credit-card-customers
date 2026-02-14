import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(page_title="clustering_credit_card_customers", layout="centered")

st.title("💳 clustering_credit_card_customers")
st.write("Enter customer financial behavior metrics to assign a cluster (KMeans).")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_credit_card.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_credit_card.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_list.json")

# Load artifacts
kmeans = joblib.load(KMEANS_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    features = json.load(f)

# IMPORTANT: These features were log-transformed during training (np.log1p)
LOG_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]

# Cluster labels (based on your cluster profiling interpretation)
cluster_labels = {
    0: "Cash Advance Heavy Users",
    1: "Low Activity Conservative Users",
    2: "High Value Premium Customers",
    3: "Installment / Revolving Users",
    4: "Balanced Mid-Level Customers",
}

cluster_descriptions = {
    0: "High cash advance usage and frequent cash advance transactions. May indicate liquidity dependence and higher risk.",
    1: "Low overall activity with conservative behavior. Often low balances and spending, sometimes higher full-payment tendency.",
    2: "High-value customers with strong purchasing activity, higher credit limits, and higher payment volumes.",
    3: "Installment-oriented users with frequent installment purchase behavior and revolving tendencies.",
    4: "Moderate and balanced behavior across multiple financial activity indicators.",
}

st.subheader("Input Features")

st.info(
    "Tip: If you're unsure about values, keep defaults and adjust gradually. "
    "Frequencies and ratios are usually between 0 and 1."
)

# Defaults (reasonable starting points)
defaults = {
    "BALANCE": 1000.0,
    "BALANCE_FREQUENCY": 0.9,
    "PURCHASES": 800.0,
    "ONEOFF_PURCHASES": 300.0,
    "INSTALLMENTS_PURCHASES": 300.0,
    "CASH_ADVANCE": 100.0,
    "PURCHASES_FREQUENCY": 0.5,
    "ONEOFF_PURCHASES_FREQUENCY": 0.2,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.3,
    "CASH_ADVANCE_FREQUENCY": 0.1,
    "CASH_ADVANCE_TRX": 2,
    "PURCHASES_TRX": 10,
    "CREDIT_LIMIT": 3000.0,
    "PAYMENTS": 900.0,
    "MINIMUM_PAYMENTS": 200.0,
    "PRC_FULL_PAYMENT": 0.2,
    "TENURE": 12,
}

# Simple input helpers
def bounded_float(name, default, min_v=0.0, max_v=None, step=0.1):
    val = st.number_input(name, min_value=min_v, value=float(default), step=float(step))
    if max_v is not None:
        val = min(val, max_v)
    return float(val)

user_input = {}

for feat in features:
    if feat in ["CASH_ADVANCE_TRX", "PURCHASES_TRX", "TENURE"]:
        user_input[feat] = st.number_input(
            feat, min_value=0, value=int(defaults.get(feat, 0)), step=1
        )
    elif feat in [
        "BALANCE_FREQUENCY",
        "PURCHASES_FREQUENCY",
        "ONEOFF_PURCHASES_FREQUENCY",
        "PURCHASES_INSTALLMENTS_FREQUENCY",
        "CASH_ADVANCE_FREQUENCY",
        "PRC_FULL_PAYMENT",
    ]:
        # Frequencies/ratios typically between 0 and 1
        user_input[feat] = bounded_float(feat, defaults.get(feat, 0.0), min_v=0.0, max_v=1.0, step=0.01)
    elif feat == "CREDIT_LIMIT":
        user_input[feat] = bounded_float(feat, defaults.get(feat, 3000.0), min_v=0.0, step=100.0)
    else:
        user_input[feat] = bounded_float(feat, defaults.get(feat, 0.0), min_v=0.0, step=10.0)

input_df = pd.DataFrame([user_input], columns=features)

if st.button("Assign Cluster"):
    # Apply the SAME log transformation used in training
    input_transformed = input_df.copy()
    for col in LOG_FEATURES:
        if col in input_transformed.columns:
            input_transformed[col] = np.log1p(input_transformed[col])

    # Scale + predict
    X_scaled = scaler.transform(input_transformed.values)
    cluster = int(kmeans.predict(X_scaled)[0])

    st.success(f"Assigned Cluster: **{cluster}**")
    st.subheader("Customer Segment")
    st.write(f"**{cluster_labels.get(cluster, 'Unknown Segment')}**")
    st.write(cluster_descriptions.get(cluster, ""))

    with st.expander("Show input values"):
        st.dataframe(input_df)

st.caption("Unsupervised Learning – KMeans Credit Card Customer Segmentation (k=5)")


  