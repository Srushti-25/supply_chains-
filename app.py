import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('iris_model.pkl')

# Set the page title
st.title("Machine Learning on Iris Data")

# Create input fields for the features
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Create a predict button
if st.button("Predict"):  # Changed button text to be more descriptive
  input_data = pd.DataFrame([[sepal_length,
                              sepal_width,
                              petal_length,
                              petal_width]],
                             columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
  prediction = model.predict(input_data)
  st.write(f"The predicted Iris species is: {prediction[0]}")
