import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler


MODEL_NAME = "Tuned Gradient Boosting"
with open(r"C:\Users\DELL\OneDrive\Desktop\steamlit\model_features.pkl", "rb") as f:
    SELECTED_FEATURES = pickle.load(f)

MODEL_PATH = "best_gb_model.pkl"      
SCALER_PATH = "scaler.pkl"

st.set_page_config(page_title="Si Content Predictor (Gradient Boosting)", layout="wide")

st.title("Hot Metal Silicon (Si) Content Prediction")
st.markdown(f"""
This app predicts **Silicon (Si) content (%)** in Hot Metal using the  
**{MODEL_NAME}** model — your best performing model.  

Only features with **|correlation| ≥ 0.45** with Si are accepted as input.
""")

# ───────────────────────────────────────────────────────────────
#   Load Model & Scaler
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}\nPlease save the trained model first.")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found: {SCALER_PATH}")
        st.stop()
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_model_and_scaler()

st.success(f"✅ Loaded **{MODEL_NAME}** model successfully!")

# ───────────────────────────────────────────────────────────────
#   Input Section
# ───────────────────────────────────────────────────────────────
st.subheader("Enter Input Parameters")

# Layout: 3 columns for better UX
cols = st.columns(3)

input_data = {}
for i, feature in enumerate(SELECTED_FEATURES):
    with cols[i % 3]:
        # You can adjust default/step/min/max based on your data ranges
        value = st.number_input(
            label=f"{feature}",
            value=0.0,
            step=0.01,
            format="%.4f",
            key=f"input_{feature}"
        )
        input_data[feature] = value

# Create single-row DataFrame from inputs
input_df = pd.DataFrame([input_data])

# ───────────────────────────────────────────────────────────────
#   Predict Button
# ───────────────────────────────────────────────────────────────
if st.button("Predict Si Content", type="primary"):
    try:
        # Scale the input using the same scaler as training
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        si_prediction = model.predict(input_scaled)[0]
        
        # Display result prominently
        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric(
            label="Predicted Silicon Content (%)",
            value=f"{si_prediction:.4f}",
            delta=None
        )
        
        st.info(f"**Model used:** {MODEL_NAME} (Tuned Gradient Boosting)")
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Blast Furnace Si Prediction App | Built with Tuned Gradient Boosting")