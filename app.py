import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Persona Predictor", page_icon="🧠", layout="wide")

# --- SLEEK FRONTEND STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FAIL-SAFE LOTTIE LOADER ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_ai = load_lottieurl("https://lottie.host/825441ec-3c35-4277-9877-33a887413c60/X7U0Yw0rSj.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open("Model1.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Model1.pkl: {e}")
        return None

model = load_model()

# --- HEADER SECTION ---
with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_ai:
            st_lottie(lottie_ai, height=280, key="main_anim")
        else:
            st.markdown("# 🤖")
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("AI Usage Intelligence")
        st.markdown("#### Predicting user behavior patterns via KNN Analysis.")

st.divider()

# --- INPUT SECTION ---
if model:
    # Feature labels for your specific Model1
    features = ["Age", "Gender", "Education Level", "City", "AI Tool Used", "Daily Usage Hours", "Purpose"]
    
    st.subheader("📊 User Profile Input")
    
    with st.container():
        # Input UI inside the "Glass" card effect
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        # Note: KNN usually requires numeric input. 
        # Make sure these match the encoding (0, 1, 2...) used in training.
        with c1:
            age = st.number_input("Age", 18, 100, 25)
            gender = st.selectbox("Gender", [0, 1], help="0: Male, 1: Female (Example)")
            edu = st.selectbox("Education Level", [0, 1, 2, 3], help="0: High School, 1: Bachelors...")
            
        with c2:
            city = st.selectbox("City", [0, 1, 2], help="Numeric representation of city")
            tool = st.selectbox("AI Tool Used", [0, 1, 2, 3], help="0: ChatGPT, 1: Gemini...")
            
        with c3:
            usage = st.slider("Daily Usage Hours", 0.0, 24.0, 3.5)
            purpose = st.selectbox("Purpose", [0, 1, 2], help="0: Work, 1: Education...")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- PREDICTION TRIGGER ---
    if st.button("Generate AI Prediction"):
        # Prepare data for model
        input_array = np.array([[age, gender, edu, city, tool, usage, purpose]])
        
        with st.spinner("Analyzing neural clusters..."):
            prediction = model.predict(input_array)
            
            st.balloons()
            st.markdown(f"""
                <div style="background: rgba(0, 210, 255, 0.1); border: 2px solid #00d2ff; padding: 30px; border-radius: 20px; text-align: center;">
                    <h2 style="color: #00d2ff; margin: 0;">PREDICTED CLASS: {prediction[0]}</h2>
                    <p style="color: #ccc;">Classification based on Model1 K-Nearest Neighbors</p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.error("Model file 'Model1.pkl' missing in directory!")
