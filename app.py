import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="AI Impact Predictor", page_icon="🎓", layout="centered")

# --- LOAD ANIMATIONS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_study = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_DMgA0w.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure 'Model.pkl' is uploaded to your GitHub repo in the same folder
    with open('Model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'Model.pkl' not found. Please upload it to your GitHub repository.")
    st.stop()

# --- UI FRONTEND ---
st.title("🎓 Student AI Impact Analyzer")
st_lottie(lottie_study, height=200)

st.write("Enter the student details below to predict the outcome using the KNN Model.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        edu_level = st.selectbox("Education Level", ["School", "Undergraduate", "Postgraduate"])
        city = st.text_input("City", value="New York")
        
    with col2:
        ai_tool = st.selectbox("AI Tool Used", ["ChatGPT", "Gemini", "Claude", "Other"])
        hours = st.slider("Daily Usage Hours", 0, 24, 2)
        purpose = st.selectbox("Purpose", ["Study", "Research", "Coding", "General"])
        grades_impact = st.selectbox("Current Impact on Grades", ["Positive", "Neutral", "Negative"])

    submit = st.form_submit_button("Predict Outcome")

if submit:
    # Note: You may need to LabelEncode these inputs to match your model training!
    # This is a placeholder for the 8 features identified in your .pkl file
    features = np.array([[age, 0, 0, 0, 0, hours, 0, 0]]) # Replace 0s with encoded values
    
    prediction = model.predict(features)
    
    st.balloons()
    st.success(f"### Predicted Result: {prediction[0]}")
    st.info("Note: Categorical data (City, Gender, etc.) must be encoded as they were during training.")
