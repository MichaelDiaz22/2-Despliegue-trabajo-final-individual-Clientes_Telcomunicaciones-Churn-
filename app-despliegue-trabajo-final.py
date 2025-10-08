import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the classical model pipeline
classical_model = joblib.load('best_classical_model_pipeline.joblib')

# Load the ensemble model pipeline
ensemble_model = joblib.load('best_ensemble_model_pipeline.joblib')

# Load the preprocessor (if needed for individual input processing)
# preprocessor = joblib.load('/content/drive/MyDrive/ANALITICA PREDICTIVA/Trabajo final individual/preprocessor.joblib')


st.title("Customer Churn Prediction")

st.write("Enter the customer's details to predict churn.")

# Create input fields for each feature
# Refer to your dataset columns for the exact feature names and types
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 1)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
StreamingTV = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
StreamingMovies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

# Create a button to trigger the prediction
if st.button("Predict Churn"):
    # Create a DataFrame from the input values
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    input_df = pd.DataFrame(input_data)

    # Add feature engineered columns if they were used in training and not handled by pipeline
    # These were commented out in a previous cell, assuming pipelines handle them.
    # If your pipelines don't add these, uncomment and add them here:
    # input_df['Contract_encoded'] = input_df['Contract'].astype('category').cat.codes
    # input_df['IsNewCustomer'] = (input_df['tenure'] == 1).astype(int)
    # input_df['CustomerValue'] = input_df['MonthlyCharges'] * input_df['tenure']

    # Apply the models to the input data
    # Assuming the pipelines handle preprocessing and scaling
    try:
        prediction_classical = classical_model.predict(input_df)
        prediction_ensemble = ensemble_model.predict(input_df)

        st.subheader("Prediction Results:")
        st.write(f"Classical Model Prediction: {'Churn' if prediction_classical[0] == 1 else 'No Churn'}")
        st.write(f"Ensemble Model Prediction: {'Churn' if prediction_ensemble[0] == 1 else 'No Churn'}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")