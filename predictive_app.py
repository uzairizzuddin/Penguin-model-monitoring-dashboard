import streamlit as st
import joblib
import time
import pandas as pd
from log_utils import log_prediction

# Page Config
st.set_page_config(page_title="Penguin Predictor", page_icon="🐧")

st.title("🐧 Penguin Species Predictor")
st.write("Enter physical measurements of the penguin to predict its species.")

# 1. Sidebar: Model Selection
model_version = st.sidebar.selectbox(
    "Select Model Version", 
    ["v1 (Logistic Regression)", "v2 (Random Forest)"]
)

# Load the selected model
try:
    if model_version == "v1 (Logistic Regression)":
        model = joblib.load('model_v1.pkl')
        version_tag = "v1"
    else:
        model = joblib.load('model_v2.pkl')
        version_tag = "v2"
except FileNotFoundError:
    st.error("Model files not found. Please run train_model_v1.py and train_model_v2.py first.")
    st.stop()

# 2. Input Form
with st.form("prediction_form"):
    st.subheader("Input Features")
    col1, col2 = st.columns(2)
    with col1:
        bill_length = st.number_input("Bill Length (mm)", min_value=0.0, value=45.0)
        bill_depth = st.number_input("Bill Depth (mm)", min_value=0.0, value=15.0)
    with col2:
        flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0, value=200.0)
        body_mass = st.number_input("Body Mass (g)", min_value=0.0, value=4000.0)
    
    submit_button = st.form_submit_button("Predict Species")

# 3. Prediction Logic
if submit_button:
    # Start Timer for Latency Calculation
    start_time = time.time()
    
    # Make Prediction
    input_data = [[bill_length, bill_depth, flipper_length, body_mass]]
    prediction = model.predict(input_data)[0]
    
    # End Timer
    end_time = time.time()
    latency = end_time - start_time
    
    # Show Result
    st.success(f"Predicted Species: **{prediction}**")
    st.info(f"Prediction Time: {latency:.4f} seconds")
    
    # Store data in session state for the feedback loop
    st.session_state['last_prediction'] = {
        'version': version_tag,
        'input': [bill_length, bill_depth, flipper_length, body_mass],
        'pred': prediction,
        'lat': latency
    }

# 4. Feedback Loop
if 'last_prediction' in st.session_state:
    st.divider()
    st.write("### Help us improve!")
    st.write("Was this prediction correct or helpful?")
    
    feedback = st.slider("Rate from 1 (Poor) to 5 (Excellent)", 1, 5, 3)
    
    if st.button("Submit Feedback"):
        data = st.session_state['last_prediction']
        
        # Save to CSV 
        log_prediction(
            data['version'], 
            data['input'], 
            data['pred'], 
            data['lat'], 
            feedback
        )
        st.success("Feedback recorded! Check the Monitoring Dashboard.")
        
        del st.session_state['last_prediction']