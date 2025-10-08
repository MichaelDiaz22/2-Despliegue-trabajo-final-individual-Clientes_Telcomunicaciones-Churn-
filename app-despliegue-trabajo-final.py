import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Configuración de la página
st.set_page_config(page_title="Predicción Churn", layout="wide")

# Título de la aplicación
st.title("Sistema de Predicción de Churn para Clientes de Telecomunicaciones")

# Función para cargar el modelo con manejo robusto de errores
@st.cache_resource
def load_model(model_path):
    """
    Carga el modelo con manejo robusto de errores
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(model_path):
            st.error(f"❌ Archivo de modelo no encontrado: {model_path}")
            st.info("""
            **Solución:**
            1. Asegúrate de que el archivo .joblib esté en el mismo directorio
            2. Verifica el nombre del archivo
            3. Si estás en Streamlit Cloud, sube el archivo al repositorio
            """)
            return None
        
        # Intentar cargar el modelo
        model = joblib.load(model_path)
        st.success(f"✅ Modelo cargado exitosamente desde: {model_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        st.info("""
        **Posibles soluciones:**
        1. **Reentrenar el modelo** con las mismas versiones de librerías
        2. **Verificar dependencias** (scikit-learn, pandas, numpy)
        3. **Usar cloudpickle** en lugar de joblib para serializar
        """)
        return None

# Cargar el modelo
with st.spinner("Cargando modelo de machine learning..."):
    classical_model_pipeline = load_model('best_classical_model_pipeline.joblib')
    ensemble_model_pipeline = load_model('best_ensemble_model_pipeline.joblib')

# Si el modelo no se carga, mostrar mensaje y detener la ejecución
if classical_model_pipeline is None:
    st.stop()

# Resto de tu aplicación...
st.success("¡Aplicación lista para hacer predicciones!")

# Aquí continúa el resto de tu código para la interfaz de usuario
# y las predicciones...

# --- Streamlit App Interface ---
st.title('Customer Churn Prediction')

st.write("""
This application predicts customer churn based on various service parameters.
Please enter the customer's details below:
""")

# Define input fields for the Streamlit app, matching the columns in df_encoded
# Based on the structure of df_encoded after one-hot encoding and feature engineering
# We need to collect inputs for:
# SeniorCitizen, tenure, MonthlyCharges, TotalCharges, CustomerValue, IsNewCustomer,
# and all the dummy variables created from categorical_nominal and Contract_encoded.

# Simple inputs for numerical features
senior_citizen = st.selectbox('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
tenure = st.slider('Tenure (months)', 0, 72, 1)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0)
# TotalCharges needs careful handling, as it's derived. For simplicity, let's input it directly for now.
# In a real app, you might want to calculate it or handle it differently.
total_charges = st.number_input('Total Charges', min_value=0.0, value=monthly_charges * tenure)


# Input for Contract (ordinal, will be label encoded)
contract_options = ['Month-to-month', 'One year', 'Two year']
contract = st.selectbox('Contract Type', contract_options)

# Inputs for nominal categorical features (will be one-hot encoded)
gender = st.selectbox('Gender', ['Female', 'Male'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines_options = ['No phone service', 'No', 'Yes']
multiple_lines = st.selectbox('Multiple Lines', multiple_lines_options)
internet_service_options = ['DSL', 'Fiber optic', 'No']
internet_service = st.selectbox('Internet Service', internet_service_options)
online_security_options = ['No', 'Yes', 'No internet service']
online_security = st.selectbox('Online Security', online_security_options)
online_backup_options = ['No', 'Yes', 'No internet service']
online_backup = st.selectbox('Online Backup', online_backup_options)
device_protection_options = ['No', 'Yes', 'No internet service']
device_protection = st.selectbox('Device Protection', device_protection_options)
tech_support_options = ['No', 'Yes', 'No internet service']
tech_support = st.selectbox('Tech Support', tech_support_options)
streaming_tv_options = ['No', 'Yes', 'No internet service']
streaming_tv = st.selectbox('Streaming TV', streaming_tv_options)
streaming_movies_options = ['No', 'Yes', 'No internet service']
streaming_movies = st.selectbox('Streaming Movies', streaming_movies_options)
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
payment_method = st.selectbox('Payment Method', payment_method_options)

# --- Data Preprocessing Function (to match df_encoded) ---
def preprocess_input(input_data):
    # Create a DataFrame from input data
    df_input = pd.DataFrame([input_data])

    # Apply Label Encoding for Contract (manual mapping based on original LabelEncoder fit)
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df_input['Contract_encoded'] = df_input['Contract'].map(contract_mapping)
    df_input = df_input.drop('Contract', axis=1) # Drop original Contract column


    # Apply One-Hot Encoding for nominal categorical variables
    categorical_nominal = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                   'PaymentMethod']

    # Need to ensure all possible dummy columns are created, even if not present in the input
    # This is crucial for consistent feature names with the trained pipelines
    # We need the list of all dummy columns that were in the original df_encoded
    # Let's assume the original df_encoded had columns like:
    # 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    # 'MultipleLines_No phone service', 'MultipleLines_Yes',
    # 'InternetService_Fiber optic', 'InternetService_No',
    # 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', etc.

    # A robust way is to define all possible dummy columns based on the unique values
    # from the original training data. For this example, I'll hardcode some common ones
    # based on the selectbox options, but this might need adjustment based on the exact training data.
    df_input_encoded = pd.get_dummies(df_input, columns=categorical_nominal, drop_first=True)


    # Recreate additional variables
    df_input_encoded['CustomerValue'] = df_input_encoded['MonthlyCharges'] * df_input_encoded['tenure']
    df_input_encoded['IsNewCustomer'] = (df_input_encoded['tenure'] < 3).astype(int)

    # Drop original columns used for creating new features if they are still present
    df_input_encoded = df_input_encoded.drop(['MonthlyCharges', 'tenure'], axis=1, errors='ignore')


    # IMPORTANT: Ensure the columns in df_input_encoded match the columns the pipelines were trained on.
    # This is the most likely source of the previous ValueError.
    # We need to add any missing dummy columns with a value of 0 and ensure the order is correct.
    # Without the exact list of columns from the training data, this is a placeholder.
    # You would need to get the list of expected columns from your trained pipeline or training data.

    # Placeholder for column alignment - REPLACE with actual column alignment logic
    # Example: expected_columns = ['col1', 'col2', 'gender_Male', ...]
    # for col in expected_columns:
    #     if col not in df_input_encoded.columns:
    #         df_input_encoded[col] = 0
    # df_input_encoded = df_input_encoded[expected_columns] # Ensure correct order

    # For demonstration, let's print the columns of the processed input
    st.write("Processed input columns:", df_input_encoded.columns.tolist())
    st.write("Processed input data head:")
    st.write(df_input_encoded.head())


    return df_input_encoded

# --- Prediction ---
if st.button('Predict Churn'):
    input_data = {
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges, # Need to consider how TotalCharges was handled in training
        'Contract': contract,
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        # CustomerValue and IsNewCustomer will be calculated in preprocess_input
    }

    processed_input = preprocess_input(input_data)

    # Check if the processed input has the correct columns for prediction
    # This is where the previous error occurred. We need to ensure column names match.
    # You would ideally check against the feature names expected by the pipeline.
    # For now, we'll attempt prediction, but be aware of potential errors if columns don't match.

    try:
        classical_prediction = classical_model_pipeline.predict(processed_input)
        ensemble_prediction = ensemble_model_pipeline.predict(processed_input)

        st.subheader('Prediction Results:')

        # Assuming the models output 0 for No Churn and 1 for Yes Churn
        churn_mapping = {0: 'No Churn', 1: 'Yes Churn'}

        st.write(f"Classical Model Prediction: **{churn_mapping.get(classical_prediction[0], 'N/A')}**")
        st.write(f"Ensemble Model Prediction: **{churn_mapping.get(ensemble_prediction[0], 'N/A')}**")

    except ValueError as e:
        st.error(f"Prediction error: {e}")
        st.warning("There might be a mismatch in the input features expected by the model. Please check the preprocessing steps and ensure the input columns match the training data.")





