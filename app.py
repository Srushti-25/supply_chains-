
import streamlit as st
import pandas as pd
import joblib

# Load the saved model components
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = joblib.load('logistic_regression_model.pkl')

st.set_page_config(layout="wide")
st.title('Supply Chain Risk Classification')
st.write('Enter the feature values below to predict the risk classification:')

# Get the feature names from X (assuming X is a pandas DataFrame)
# We can infer these from the original X DataFrame
# For simplicity, we'll list them out, but in a real app, you might automate this.
feature_names = [
    'vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate',
    'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level',
    'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status',
    'weather_condition_severity', 'port_congestion_level', 'shipping_costs',
    'supplier_reliability_score', 'lead_time_days', 'historical_demand',
    'iot_temperature', 'cargo_condition_status', 'route_risk_level',
    'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score',
    'disruption_likelihood_score', 'delay_probability'
]

# Create input fields for each feature
input_data = {}
num_cols = 3 # Number of columns for layout
cols = st.columns(num_cols)

for i, feature in enumerate(feature_names):
    with cols[i % num_cols]:
        # You can add default values or hints based on your data distribution
        input_data[feature] = st.number_input(f'**{feature.replace("_", " ").title()}**', value=0.0, format="%.4f")

if st.button('Predict Risk'):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(input_scaled)

    # Decode the prediction back to original class labels
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    st.success(f'The predicted Supply Chain Risk Classification is: **{prediction_label[0]}**')
