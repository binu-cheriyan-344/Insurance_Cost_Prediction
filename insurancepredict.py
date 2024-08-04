import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('best_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to calculate BMI
def calculate_bmi(height, weight):
    if height > 0:  # to avoid division by zero
        return (weight / (height/100)**2)
    else:
        return 0.0

# Function to make predictions
def predict_premium(age,any_transplants,height,weight, num_major_surgeries):
    bmi = calculate_bmi(height, weight)

    # Create a DataFrame for input features
    input_data = pd.DataFrame({
        'Age': [float(age)],
        'AnyTransplants': [float(any_transplants)],
        'Height': [float(height)],
        'Weight': [float(weight)],
        'NumberOfMajorSurgeries': [float(num_major_surgeries)],
        'BMI': [float(bmi)]
    })

    # Apply scaling
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    return prediction[0], bmi

    #features = np.array([['Age','AnyTransplants','Height','Weight','NumberOfMajorSurgeries','BMI']])


# Streamlit app
st.title('Insurance Premium Prediction')

# Create columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=15, max_value=120, value=15)
    height = st.number_input('Height (cm)', min_value=145, max_value=200, value=145)
    weight = st.number_input('Weight (kg)', min_value=50, max_value=150, value=50)
    bmi = calculate_bmi(height, weight)  # Calculate BMI based on height and weight

with col2:
    any_transplants = st.selectbox('Any Transplants', [0, 1])
    num_major_surgeries = st.number_input('Number Of Major Surgeries', min_value=0, max_value=3, value=0)

# Calculate BMI
bmi = calculate_bmi(height, weight)

# Predict button
if st.button('Predict Premium'):
    prediction,bmi = predict_premium(age,any_transplants,height,weight, num_major_surgeries)
    st.write(f'Predicted Insurance Premium: ₹ {prediction:.2f}')
    st.write(f'Calculated BMI: {bmi:.2f}')
