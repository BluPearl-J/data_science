import streamlit as st
import requests


st.set_page_config(page_title="California House Price Predictor", layout="centered")
st.title("  Housing Price Predictor")
st.write("Adjust the sliders below to estimate the value of a house.")


col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    age = st.slider("House Median Age", 1, 52, 20)
    rooms = st.number_input("Total Rooms", value=800)

with col2:
    bedrooms = st.number_input("Total Bedrooms", value=150)
    population = st.number_input("Population", value=300)
    households = st.number_input("Households", value=120)
    income = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.0)

ocean = st.selectbox("Ocean Proximity",
                     ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

if st.button("Predict House Value", use_container_width=True):

    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": age,
        "total_rooms": rooms,
        "total_bedrooms": bedrooms,
        "population": population,
        "households": households,
        "median_income": income,
        "ocean_proximity": ocean
    }


    try:

        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            prediction = response.json()["estimated_value"]
            st.success(f"### Estimated Price: ${prediction:,.2f}")
        else:
            st.error("API returned an error. Check your main.py terminal.")

    except Exception as e:
        st.error("Could not connect to the API. Is main.py running?")