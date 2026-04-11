import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import time
import os

# Optional Lottie import (safe)
try:
    from streamlit_lottie import st_lottie
except:
    st_lottie = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Impact Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white; border-radius: 10px; border: none; padding: 10px 24px;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover { transform: translateY(-2px) scale(1.02); }
    h1, h2, h3 { color: #4BA3E3; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "Model1.pkl")
        with open("Model1.pkl", 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        return e

# --- LOAD ASSETS ---
lottie_ai = load_lottieurl("https://lottie.host/8b7d27e7-3b95-46f9-90d0-4bd24687d69b/gX9T2Wj39T.json")

model = load_model()

if isinstance(model, Exception):
    st.error(f"❌ Failed to load model1.pkl → {model}")
    model_loaded = False
else:
    model_loaded = True

# --- HEADER SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🚀 AI Impact Predictor")
    st.write("Enter user details to predict **Impact on Grades**.")

with col2:
    if lottie_ai and st_lottie:
        st_lottie(lottie_ai, height=150)
    else:
        st.info("Animation not available")

st.markdown("---")

# --- INPUT SECTION ---
st.subheader("📊 User Profile & Usage Metrics")

left_col, mid_col, right_col = st.columns(3)

with left_col:
    age = st.number_input("Age", 10, 100, 20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

with mid_col:
    education_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate"])
    ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Claude", "Gemini", "Copilot", "Other"])

with right_col:
    daily_hours = st.number_input("Daily Usage Hours", 0.0, 24.0, 2.0)
    purpose = st.selectbox("Primary Purpose", ["Research", "Coding", "Writing", "General Query", "Entertainment"])

# --- ENCODING ---
mappings = {
    "gender": {"Male": 0, "Female": 1, "Other": 2},
    "education": {"High School": 0, "Undergraduate": 1, "Postgraduate": 2},
    "city": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2},
    "ai_tool": {"ChatGPT": 0, "Claude": 1, "Gemini": 2, "Copilot": 3, "Other": 4},
    "purpose": {"Research": 0, "Coding": 1, "Writing": 2, "General Query": 3, "Entertainment": 4}
}

# --- PREDICTION ---
st.markdown("---")

if model_loaded:
    if st.button("🔮 Predict Impact"):

        # FIXED INDENTATION HERE ✅
        features = pd.DataFrame([{
            "Age": age,
            "Gender": mappings["gender"][gender],
            "Education_Level": mappings["education"][education_level],
            "City": mappings["city"][city],
            "AI_Tool_Used": mappings["ai_tool"][ai_tool],
            "Daily_Usage_Hours": daily_hours,
            "Purpose": mappings["purpose"][purpose]
        }])

        label_map = {0: "Low", 1: "Medium", 2: "High"}

        with st.spinner("Analyzing data..."):
            time.sleep(1)

            try:
                prediction = model.predict(features)

                raw_output = prediction[0]
                predicted_class = label_map.get(raw_output, raw_output)

                st.markdown("### 🎯 Prediction Result")

                if predicted_class == "High":
                    st.success("✅ **High Impact** – AI usage is highly beneficial!")
                    st.balloons()

                elif predicted_class == "Medium":
                    st.info("ℹ️ **Medium Impact** – Balanced usage.")

                else:
                    st.warning("⚠️ **Low Impact** – Try improving usage strategy.")

            except Exception as e:
                st.error(f"❌ Prediction error → {e}")
else:
    st.warning("⚠️ Model not loaded. Please fix model.pkl file.")
