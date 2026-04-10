import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="AI Impact Predictor", page_icon="🤖", layout="wide")

# Custom CSS for an attractive UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #4CAF50; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- ANIMATIONS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_ai = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_m6cu96ze.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Ensure 'Model.pkl' is in your GitHub repo folder
    with open('Model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# --- FRONTEND ---
st.title("🎓 Student AI-Usage Impact Predictor")
st.write("Analyze how AI tools are shaping academic outcomes.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st_lottie(lottie_ai, height=300)

with col_right:
    with st.expander("📝 Enter Student Details", expanded=True):
        age = st.number_input("Age", 10, 60, 21)
        # Note: These need to be encoded to numbers (0, 1, 2...) 
        # based on how you trained your model.
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        edu = st.selectbox("Education", [0, 1, 2], format_func=lambda x: ["School", "UG", "PG"][x])
        city = st.number_input("City Code (e.g., 0-10)", 0, 10, 0)
        tool = st.selectbox("AI Tool", [0, 1, 2], format_func=lambda x: ["ChatGPT", "Gemini", "Other"][x])
        hours = st.slider("Daily Usage Hours", 0, 12, 2)
        purpose = st.selectbox("Purpose", [0, 1], format_func=lambda x: "Study" if x==0 else "Work")
        impact = st.selectbox("Current Impact", [0, 1], format_func=lambda x: "Neutral" if x==0 else "Positive")

    if st.button("✨ Predict Impact"):
        # The model expects 8 features 
        input_data = np.array([[age, gender, edu, city, tool, hours, purpose, impact]])
        prediction = model.predict(input_data)
        
        st.balloons()
        st.success(f"### Predicted Category: {prediction[0]}")
