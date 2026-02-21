import numpy as np
import streamlit as st
import pickle

st.set_page_config(page_title="Heart_Disease_prediction", page_icon=":shark:", layout="wide")
st.title("Heart Disease Prediction")

def load_model():
    return pickle.load(open("heart.pkl", "rb"))

value1 = st.number_input("Age", min_value=1, max_value=120, value=30)
value2 = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
value3 = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
value4 = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
value5 = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
value6 = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
value7 = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
value8 = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
value9 = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
value10 = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
value11 = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
value12 = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
value13 = st.selectbox("Thalassemia (thal)", [3, 6, 7])

features = np.array([[value1, value2, value3, value4, value5, value6, value7,
                      value8, value9, value10, value11, value12, value13]])

if st.button("Predict"):
    model = load_model()
    prediction = model.predict(features)
    st.write("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
