import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model and encoders
with open("C:\\Users\\SHINJITA\\OneDrive\\Desktop\\Autism Files\\best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Users\\SHINJITA\\OneDrive\\Desktop\\Autism Files\\encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Set page config
st.set_page_config(page_title="ASD Screening", layout="centered")

# App Title
st.markdown("<h1 style='text-align: center;'>🧠 Autism Spectrum Disorder (ASD) Screening App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Answer the following questions to screen for the likelihood of Autism Spectrum Disorder.</p>", unsafe_allow_html=True)

st.markdown("---")

# AQ-10 Questions
questions = {
    "A1_Score": "Do you prefer to do things the same way over and over again?",
    "A2_Score": "Do social situations confuse you?",
    "A3_Score": "Do you find it hard to make new friends?",
    "A4_Score": "Do you notice small sounds when others do not?",
    "A5_Score": "Do you tend to focus on details rather than the big picture?",
    "A6_Score": "Do you find it difficult to understand what others are feeling from their facial expressions?",
    "A7_Score": "Do you enjoy social chit-chat?",
    "A8_Score": "Do you frequently notice small changes in situations or people's appearance?",
    "A9_Score": "Do you have trouble understanding jokes or sarcasm?",
    "A10_Score": "Do you find it hard to imagine what someone else might be thinking?"
}

st.markdown("### AQ-10 Screening Questions")
aq_scores = []
for key, question in questions.items():
    response = st.radio(question, ["Yes", "No"], key=key, horizontal=True)
    score = 1 if response == "Yes" else 0
    aq_scores.append(score)

st.markdown("---")

# Other Demographic Inputs
st.markdown("### Demographic Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ["m", "f"])
    ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)

with col2:
    jaundice = st.selectbox("Did you have jaundice as a newborn?", ["yes", "no"])
    austim = st.selectbox("Family history of autism?", ["yes", "no"])
    country = st.selectbox("Country of residence", encoders["contry_of_res"].classes_)

relation = st.selectbox("Who is completing the screening?", encoders["relation"].classes_)

# Internally set values
used_app_before = "yes"
result = 10.0

# Prediction
st.markdown("---")
if st.button("🔍 Predict ASD Likelihood"):
    input_data = aq_scores + [
        int(age),
        encoders["gender"].transform([gender])[0],
        encoders["ethnicity"].transform([ethnicity])[0],
        encoders["jaundice"].transform([jaundice])[0],
        encoders["austim"].transform([austim])[0],
        encoders["contry_of_res"].transform([country])[0],
        encoders["used_app_before"].transform([used_app_before])[0],
        float(result),
        encoders["relation"].transform([relation])[0]
    ]

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    label = "✅ Likely" if prediction == 1 else "❌ Unlikely"
    confidence = proba[prediction] * 100

    st.success(f"### Prediction: {label}")
    st.markdown(f"**Confidence Level:** {confidence:.2f}%")

    if prediction == 1:
        st.info("This result suggests a likelihood of ASD. Please consult with a medical professional for a full evaluation.")
    else:
        st.info("This result suggests it is unlikely. If you still have concerns, consider speaking with a healthcare provider.")

