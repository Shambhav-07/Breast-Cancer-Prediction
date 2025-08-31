import joblib
import streamlit as st
import numpy as np

# Load reduced trained model
model = joblib.load("cancer_model_reduced.pkl")

st.title("🩺 Breast Cancer Prediction App")
st.write("Enter the tumor measurements below or pick a sample case to pre-fill values (you can still edit).")

# Define your 5 best features
feature_names = [
    "area_worst",
    "concave points_worst",
    "concave points_mean",
    "radius_worst",
    "perimeter_worst"
]

# Sample test cases (B = 0, M = 1)
sample_data = {
    "Benign Example": [880.8, 0.089, 0.050, 12.45, 82.0],
    "Malignant Example": [2010.0, 0.265, 0.180, 25.38, 184.6],
}

# Sidebar: pick sample
st.sidebar.header("🔬 Sample Test Data")
choice = st.sidebar.selectbox("Choose a sample case", ["Manual Input"] + list(sample_data.keys()))

# Pre-fill values
if choice == "Manual Input":
    default_values = [0.0] * len(feature_names)
else:
    default_values = sample_data[choice]

# Editable input fields (pre-filled but can be changed)
inputs = []
for i, feature in enumerate(feature_names):
    value = st.text_input(f"{feature}", value=str(default_values[i]))
    try:
        value = float(value)
    except:
        value = 0.0
    inputs.append(value)

# Prediction button
if st.button("🔍 Predict"):
    features = np.array([inputs])  # Convert to correct shape
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("🚨 The tumor is **Malignant (M)**")
    else:
        st.success("✅ The tumor is **Benign (B)**")
