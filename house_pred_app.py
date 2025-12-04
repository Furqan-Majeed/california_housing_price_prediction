import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and column names
# Ensure these files are in the same folder as app.py
model = joblib.load("forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏡 California Housing Price Estimator")
st.write("Adjust the sliders and values below to estimate the property price.")

# 2. Create Input Fields for User
# We use st.columns to make it look clean (2 columns side-by-side)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location Details")
    longitude = st.number_input("Longitude", value=-122.23, help="e.g., -122.4 for SF, -118.2 for LA")
    latitude = st.number_input("Latitude", value=37.88, help="e.g., 37.7 for SF, 34.0 for LA")
    ocean_proximity = st.selectbox("Location Type", 
                     ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
                     help="How close is the house to the ocean?")
    
    st.subheader("🏠 House Details")
    housing_median_age = st.number_input("House Age (Years)", value=30.0, step=1.0)
    # Asking for "Average" is easier for a layman than "Total rooms in block"
    avg_rooms = st.number_input("Average Rooms per House", value=5.0, step=0.5, help="Total rooms including living room, bedrooms, etc.")
    avg_bedrooms = st.number_input("Average Bedrooms per House", value=1.0, step=0.1)

with col2:
    st.subheader("👥 Neighborhood Stats")
    # Explaining that this is for the whole block, not just one house
    households = st.number_input("Total Households in Block", value=400.0, step=10.0, help="How many families live in this census block?")
    population = st.number_input("Total Population in Block", value=1200.0, step=50.0)
    
    st.subheader("💰 Economic Factors")
    # Clarifying the weird scale (3.8 = $38k)
    median_income = st.number_input("Median Income (Score)", value=3.8, step=0.1, 
                                    help="3.8 represents roughly $38,000 annual income (1990 standard). High income areas use 8-10.")

# 3. Processing the Input
if st.button("Predict Price"):
    
    # --- LOGIC: Convert Layman Inputs to Model Inputs ---
    # The model was trained on "Total Rooms in Block", so we multiply:
    calculated_total_rooms = avg_rooms * households
    calculated_total_bedrooms = avg_bedrooms * households

    # A. Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [calculated_total_rooms],      # Calculated Value
        'total_bedrooms': [calculated_total_bedrooms], # Calculated Value
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

    # B. Log Transformations (replicating your training logic)
    # The model expects log-transformed values, so we apply it here
    input_data["total_rooms"] = np.log(input_data["total_rooms"] + 1)
    input_data["total_bedrooms"] = np.log(input_data["total_bedrooms"] + 1)
    input_data["population"] = np.log(input_data["population"] + 1)
    input_data["households"] = np.log(input_data["households"] + 1)

    # C. Feature Engineering (Ratios)
    input_data["bedroom_ratio"] = input_data["total_bedrooms"] / input_data["total_rooms"]
    input_data["household_rooms"] = input_data["total_rooms"] / input_data["households"]

    # D. One-Hot Encoding & Alignment
    # First, convert the categorical text to numbers
    input_data = input_data.join(pd.get_dummies(input_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

    # CRITICAL FIX: Ensure columns match the training data exactly
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # 4. Make Prediction
    prediction = model.predict(input_data)[0]
    
    # 5. Display Results
    st.divider() # Adds a nice visual line
    st.subheader("🏷️ Prediction Results")
    
    st.success(f"Estimated 1990 Price: ${prediction:,.2f}")
    
    # Inflation Calculation
    inflation_factor = 3.33 
    current_price = prediction * inflation_factor
    
    st.info(f"💰 Adjusted for Inflation (Approx. 2025 Value): ${current_price:,.2f}")
