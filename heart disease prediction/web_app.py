import streamlit as st
import pickle
import pandas as pd

# Load model + threshold
with open("heart_disease_pipeline.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
threshold = saved["threshold"]

st.title("❤️ 10-Year Heart Disease Prediction")
st.write("Enter patient details:")

male = st.selectbox("Male (0 = Female, 1 = Male)", [0, 1])
age = st.number_input("Age", 20, 100)
education = st.selectbox("Education (1-4)", [1, 2, 3, 4])
currentSmoker = st.selectbox("Current Smoker (0/1)", [0, 1])
cigsPerDay = st.number_input("Cigarettes Per Day", 0, 70)
BPMeds = st.selectbox("On BP Medication (0/1)", [0, 1])
prevalentStroke = st.selectbox("Prevalent Stroke (0/1)", [0, 1])
prevalentHyp = st.selectbox("Hypertension (0/1)", [0, 1])
diabetes = st.selectbox("Diabetes (0/1)", [0, 1])
totChol = st.number_input("Total Cholesterol", 100, 600)
sysBP = st.number_input("Systolic BP", 80, 250)
diaBP = st.number_input("Diastolic BP", 40, 150)
BMI = st.number_input("BMI", 10.0, 60.0)
heartRate = st.number_input("Heart Rate", 40, 200)
glucose = st.number_input("Glucose", 40, 400)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'male': male,
        'age': age,
        'education': education,
        'currentSmoker': currentSmoker,
        'cigsPerDay': cigsPerDay,
        'BPMeds': BPMeds,
        'prevalentStroke': prevalentStroke,
        'prevalentHyp': prevalentHyp,
        'diabetes': diabetes,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose
    }])

    # Ensure correct column order
    input_data = input_data[model.feature_names_in_]

    proba = model.predict_proba(input_data)[0][1]

    st.write(f"Risk Probability: {proba:.2%}")

    if proba > threshold:
        st.error("⚠️ High Risk of Heart Disease (10 Year)")
    else:
        st.success("✅ Low Risk")