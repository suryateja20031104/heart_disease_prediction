import streamlit as st
import numpy as np
import pickle 
from sklearn.metrics import accuracy_score

model = pickle.load(open("heart_disease_model.pkl", "rb"))

st.title("Heart Disease Prediction App")
st.write("Enter the patient's details to predict the likelihood of heart disease.")


age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
cpt = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=2)
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1: Yes, 0: No)", [0, 1])
restecg = st.number_input("Resting ECG Results", min_value=0, max_value=2, value=1)
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina (1: Yes, 0: No)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
slope = st.number_input("Slope of Peak Exercise ST Segment", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thall = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, value=2)


input_data = np.array([age, sex, cpt, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thall]).reshape(1, -1)


if st.button("Predict"): 
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.success("Good News! The patient does not have any heart disease.")
    else:
        st.error("The patient should visit the doctor for further check-up.")
