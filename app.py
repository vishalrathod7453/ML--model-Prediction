import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="AI Impact Predictor", page_icon="🎓", layout="centered")

# --- ANIMATION ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_anim = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3v8p.json")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Make sure 'Model.pkl' is the exact name in your GitHub repo
    with open('Model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model.pkl not found. Please upload it to your repository.")
    st.stop()

# --- UI ---
st.title("🎓 Student AI Impact Analyzer")
if lottie_anim:
    st_lottie(lottie_anim, height=200)

st.write("Enter all 8 parameters to get a prediction.")

with st.form("main_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 10, 60, 20)
        # Note: You must use the same numerical encoding used during training
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        edu = st.selectbox("Education Level", [0, 1, 2], format_func=lambda x: ["School", "UG", "PG"][x])
        city = st.number_input("City ID", 0, 50, 1)
        
    with col2:
        ai_tool = st.selectbox("AI Tool Used", [0, 1, 2, 3], format_func=lambda x: ["ChatGPT", "Gemini", "Claude", "Other"][x])
        hours = st.number_input("Daily Usage Hours", 0.0, 24.0, 2.0)
        purpose = st.selectbox("Purpose", [0, 1, 2], format_func=lambda x: ["Study", "Research", "Other"][x])
        impact = st.selectbox("Current Grade Impact", [0, 1, 2], format_func=lambda x: ["Negative", "Neutral", "Positive"][x])

    submit = st.form_submit_button("Predict Outcome")

if submit:
    try:
        # Construct the array with exactly 8 features in the correct order 
        input_data = np.array([[
            age,          # 1. Age
            gender,       # 2. Gender
            edu,          # 3. Education_Level
            city,         # 4. City
            ai_tool,      # 5. AI_Tool_Used
            hours,        # 6. Daily_Usage_Hours
            purpose,      # 7. Purpose
            impact        # 8. Impact_on_Grades
        ]])
        
        prediction = model.predict(input_data)
        
        st.balloons()
        st.success(f"### Predicted Classification: {prediction[0]}")
        
    except ValueError as e:
        st.error(f"Feature Mismatch Error: {e}")
        st.info("The model requires exactly 8 features. Check the logs to see what was sent.")
