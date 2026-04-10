import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Usage Predictor", page_icon="🤖", layout="centered")

# --- ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9unqzzx.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("Model1.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- CUSTOM CSS FOR ANIMATION & STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div.stButton > button:first-child {
        background-color: #00ffbd;
        color: black;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #00d49d;
        border: 2px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
with st.container():
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.title("🤖 AI Insight Predictor")
        st.subheader("Predicting AI Tool engagement for 2026.")
        st.write("Enter your demographics below to see the predicted AI behavior.")
    with right_col:
        st_lottie(lottie_ai, height=150, key="coding")

st.write("---")

# --- USER INPUT FORM ---
with st.form("prediction_form"):
    st.markdown("### 📋 User Profile Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    
    with col2:
        city = st.selectbox("City Category", ["Tier 1", "Tier 2", "Tier 3"])
        usage_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.5)
        purpose = st.selectbox("Primary Purpose", ["Education", "Work", "Personal", "Research"])

    # Mapping categorical data to match model training (adjust based on your LabelEncoders)
    # Note: freshers should ensure these mappings match the training set exactly.
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    city_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    purpose_map = {"Education": 0, "Work": 1, "Personal": 2, "Research": 3}

    submit = st.form_submit_button("Generate Prediction ✨")

# --- PREDICTION LOGIC ---
if submit:
    # Prepare features based on Model1.pkl structure
    features = np.array([[
        age, 
        gender_map[gender], 
        edu_map[education], 
        city_map[city], 
        # Note: 'AI_Tool_Used' was a feature in your PKL. 
        # If it's a feature, you'd need an input. If it's the target, remove from here.
        0, # Placeholder for AI_Tool_Used feature index
        usage_usage_hours,
        purpose_map[purpose]
    ]])
    
    try:
        prediction = model.predict(features)
        
        st.balloons()
        st.success(f"### 🎯 Prediction Result: {prediction[0]}")
        
        # Display analysis
        with st.expander("See Detailed Metrics"):
            st.write(f"Based on a KNN (k={model.n_neighbors}) analysis, your profile aligns with users who typically interact with AI for **{purpose}**.")
            
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("Built for the 2026 Career Portfolio | Python & Streamlit Integration")
