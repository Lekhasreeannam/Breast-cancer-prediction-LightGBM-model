import streamlit as st
import pandas as pd
import pickle

# ==============================
# Load Saved Model
# ==============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Prediction System")

st.markdown("""
This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained **LightGBM Classifier**.
""")

st.markdown("---")

# ==============================
# User Input
# ==============================
st.subheader("Enter Tumor Measurements")

col1, col2 = st.columns(2)

user_input = {}

for i, feature in enumerate(feature_names):

    with (col1 if i % 2 == 0 else col2):

        user_input[feature] = st.number_input(
            label=feature.replace("_", " ").title(),
            value=0.0,
            step=0.01,
            format="%.4f"
        )

st.markdown("---")

# ==============================
# Prediction
# ==============================
if st.button("Predict"):

    input_df = pd.DataFrame([user_input])

    # Keep the same feature order
    input_df = input_df[feature_names]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    # Change these labels only if your target encoding was different
    if prediction == 1:
        st.error("🔴 Malignant Tumor")
    else:
        st.success("🟢 Benign Tumor")


st.markdown("---")

st.info(
    "Model: LightGBM Classifier | Dataset: Breast Cancer Wisconsin Dataset"
)