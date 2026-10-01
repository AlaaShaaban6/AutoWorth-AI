
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AutoWorth AI",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# Load Model and Configuration
# ============================================================

model = joblib.load("tuned_random_forest.pkl")
preprocessor = joblib.load("preprocessor.pkl")
config = joblib.load("deployment_config.pkl")

REFERENCE_YEAR = config["reference_year"]

make_options = config["make_options"]
model_options = config["model_options"]
transmission_options = config["transmission_options"]
fuel_type_options = config["fuel_type_options"]


# ============================================================
# Feature Engineering
# ============================================================

def create_features(data):

    data = data.copy()

    data["car_age"] = REFERENCE_YEAR - data["year"]

    data["mileage_per_year"] = (
        data["mileage"] / data["car_age"]
    )

    data["engine_mpg_ratio"] = (
        data["engineSize"] / data["mpg"]
    )

    return data


# ============================================================
# Deal Advisor
# ============================================================

def deal_advisor(predicted_price, seller_price):

    difference = seller_price - predicted_price

    difference_percent = (
        difference / predicted_price
    ) * 100

    if difference_percent < -10:
        rating = "Great Deal"

    elif difference_percent < -5:
        rating = "Good Deal"

    elif difference_percent <= 5:
        rating = "Fair Price"

    elif difference_percent <= 10:
        rating = "Slightly Overpriced"

    else:
        rating = "Overpriced"

    return difference, difference_percent, rating


# ============================================================
# App Header
# ============================================================

st.title("🚗 AutoWorth AI")

st.subheader("Used Car Price & Deal Advisor")

st.write(
    "Enter the vehicle details below to estimate its market price "
    "and compare it with the seller's asking price."
)


# ============================================================
# User Inputs
# ============================================================

st.markdown("### Vehicle Information")

make = st.selectbox(
    "Make",
    make_options
)

car_model = st.selectbox(
    "Model",
    model_options
)

year = st.number_input(
    "Year",
    min_value=1991,
    max_value=2020,
    value=2018,
    step=1
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=500000,
    value=30000,
    step=100
)

transmission = st.selectbox(
    "Transmission",
    transmission_options
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_type_options
)

engine_size = st.number_input(
    "Engine Size",
    min_value=0.1,
    max_value=7.0,
    value=1.6,
    step=0.1
)

seller_price = st.number_input(
    "Seller Price (£)",
    min_value=100,
    max_value=200000,
    value=15000,
    step=100
)


# ============================================================
# Prediction
# ============================================================

if st.button("Estimate Price & Check Deal", type="primary"):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "model": car_model,
        "year": year,
        "mileage": mileage,
        "transmission": transmission,
        "fuelType": fuel_type,
        "tax": np.nan,
        "mpg": np.nan,
        "engineSize": engine_size,
        "Make": make
    }])

    # Feature engineering
    input_data_fe = create_features(input_data)

    # Preprocessing
    input_processed = preprocessor.transform(input_data_fe)

    # Prediction
    predicted_price = model.predict(input_processed)[0]

    # Deal analysis
    difference, difference_percent, rating = deal_advisor(
        predicted_price,
        seller_price
    )


    # ========================================================
    # Results
    # ========================================================

    st.markdown("---")
    st.markdown("### 💰 Price Estimate")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Market Price",
            f"£{predicted_price:,.2f}"
        )

    with col2:
        st.metric(
            "Seller Price",
            f"£{seller_price:,.2f}"
        )


    st.markdown("### 📊 Deal Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Difference",
            f"£{difference:,.2f}"
        )

    with col2:
        st.metric(
            "Difference %",
            f"{difference_percent:+.2f}%"
        )


    st.markdown("### 🏷️ Deal Rating")

    if rating == "Great Deal":
        st.success(f"**{rating}**")

    elif rating == "Good Deal":
        st.success(f"**{rating}**")

    elif rating == "Fair Price":
        st.info(f"**{rating}**")

    elif rating == "Slightly Overpriced":
        st.warning(f"**{rating}**")

    else:
        st.error(f"**{rating}**")


    st.caption(
        "The estimate is based on the trained AutoWorth AI model "
        "and should be used as a data-driven pricing aid."
    )
