import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Config
st.set_page_config(page_title="AI Performance Predictor", page_icon="📈", layout="centered")

# --- ANIMATION LOADER ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3v8p.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure the filename 'Model.pkl' matches your GitHub file exactly 
    with open('Model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI FRONTEND ---
st.title("🎓 Student AI Impact Analyzer")
if lottie_coding:
    st_lottie(lottie_coding, height=200)

st.markdown("### Predict Academic Outcomes based on AI Tool Usage")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 10, 60, 20)
        # Numerical encoding is required for KNN models [cite: 37, 92]
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        edu = st.selectbox("Education Level", [0, 1, 2], format_func=lambda x: ["School", "UG", "PG"][x])
        city = st.number_input("City (Encoded ID)", 0, 100, 0)
        
    with col2:
        ai_tool = st.selectbox("AI Tool", [0, 1, 2], format_func=lambda x: ["ChatGPT", "Gemini", "Other"][x])
        hours = st.slider("Daily Usage Hours", 0, 24, 2)
        purpose = st.selectbox("Purpose", [0, 1], format_func=lambda x: "Academic" if x==0 else "Personal")
        impact = st.selectbox("Current Impact", [0, 1], format_func=lambda x: "Neutral/Negative" if x==0 else "Positive")

    submit = st.form_submit_button("✨ Generate Prediction")

if submit:
    # Prepare input for the 8-feature model [cite: 1]
    input_data = np.array([[age, gender, edu, city, ai_tool, hours, purpose, impact]])
    
    prediction = model.predict(input_data)
    
    st.balloons()
    st.success(f"### Predicted Result: {prediction[0]}")
    st.info("The model classifies outcomes based on the KNN algorithm.")
