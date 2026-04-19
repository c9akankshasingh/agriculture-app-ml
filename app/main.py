import streamlit as st
from PIL import Image
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import predict_crop, predict_soil, predict_disease
from api.weather import get_weather_data

st.set_page_config(page_title="Smart Agriculture AI", layout="wide")

st.title("Smart Agriculture AI System")

# Initialize session state only once
if "weather" not in st.session_state:
    st.session_state.weather = None
    st.session_state.weather_loaded = False

# Initialize form inputs for Tab 1
if "nitrogen" not in st.session_state:
    st.session_state.nitrogen = 0
if "phosphorus" not in st.session_state:
    st.session_state.phosphorus = 0
if "potassium" not in st.session_state:
    st.session_state.potassium = 0
if "soil_ph" not in st.session_state:
    st.session_state.soil_ph = 7.0
if "temperature" not in st.session_state:
    st.session_state.temperature = 25.0
if "humidity" not in st.session_state:
    st.session_state.humidity = 50.0
if "rainfall" not in st.session_state:
    st.session_state.rainfall = 0.0

# Load weather data only on first run, not on every rerun
if not st.session_state.weather_loaded:
    try:
        with st.spinner("Loading default weather data..."):
            st.session_state.weather = get_weather_data(25.8732, 82.0992)  # Default coordinates (India)
            st.session_state.weather_loaded = True
    except Exception as e:
        st.session_state.weather = {}
        st.session_state.weather_loaded = True
        st.warning(f"Could not load default weather data: {e}")

# Check for model loading errors
from models.predict import model_errors
if model_errors:
    st.warning("⚠️ **Some models failed to load:**")
    for model_name, error in model_errors.items():
        st.caption(f"• {model_name}: {error}")

with st.sidebar:
    st.header("Weather Data")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=25.8732, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=82.0992, format="%.4f")
    if st.button("Load Weather Data", key="load_weather_button"):
        with st.spinner("Loading weather data..."):
            try:
                st.session_state.weather = get_weather_data(latitude, longitude)
                st.success("Weather data loaded successfully!")
            except Exception as e:
                st.error(f"Unable to load weather data: {e}")
                st.session_state.weather = {}
    
    # Display weather data safely
    if st.session_state.weather:
        try:
            weather = st.session_state.weather
            st.metric("Temperature", f"{weather.get('temperature', 'N/A')} °C")
            st.metric("Humidity", f"{weather.get('humidity', 'N/A')} %")
            st.metric("Precipitation", f"{weather.get('precipitation', 'N/A')} mm")
            st.write(f"Location: {weather.get('latitude')}, {weather.get('longitude')} ({weather.get('timezone')})")
        except Exception as e:
            st.error(f"Error displaying weather data: {e}")



tab1, tab2, tab3 = st.tabs(["Crop Recommendation", "Soil Analysis", "Plant Disease Detection"])

with tab1:
    st.subheader("Enter Soil and Weather Details")
    weather = st.session_state.weather or {}
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen", value=0, key="nitrogen")
        P = st.number_input("Phosphorus", value=0, key="phosphorus")
        K = st.number_input("Potassium", value=0, key="potassium")
        ph = st.number_input("Soil pH", value=7.0, key="soil_ph")
    with col2:
        temperature = st.number_input("Temperature (°C)", value=weather.get("temperature", 25.0), key="temperature")
        humidity = st.number_input("Humidity (%)", value=weather.get("humidity", 50.0), key="humidity")
        rainfall = st.number_input("Rainfall (mm)", value=weather.get("precipitation", 0.0), key="rainfall")

    if st.button("Predict Crop", key="predict_crop_button"):
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        if result.get("success", False):
            st.success(f"✅ Recommended Crop: **{result.get('message')}**")
        else:
            st.error(f"❌ Error: {result.get('message')}")

with tab2:
    st.subheader("Upload Soil Image")
    soil_file = st.file_uploader("Upload soil photo", type=["jpg", "png", "jpeg"], key="soil_up")
    if soil_file:
        try:
            soil_file.seek(0)
            img = Image.open(soil_file)
            st.image(img, width=300)
            if st.button("Predict Soil Type", key="predict_soil_button"):
                with st.spinner("Analyzing soil..."):
                    res = predict_soil(img)
                    if res.get("success", False):
                        st.success(f"✅ Soil Type: **{res.get('message')}**")
                    else:
                        st.error(f"❌ Error: {res.get('message')}")
        except Exception as e:
            st.error(f"Error loading image: {e}. Please upload a valid image file.")

with tab3:
    st.subheader("Upload Crop Leaf Image")
    disease_file = st.file_uploader("Upload leaf photo", type=["jpg", "png", "jpeg"], key="leaf_up")
    if disease_file:
        try:
            disease_file.seek(0)
            img = Image.open(disease_file)
            st.image(img, width=300)
            if st.button("Predict Disease", key="predict_disease_button"):
                with st.spinner("Analyzing leaf..."):
                    res = predict_disease(img)
                    if res.get("success", False):
                        st.success(f"✅ Plant Disease Result: **{res.get('message')}**")
                    else:
                        st.error(f"❌ Error: {res.get('message')}")
        except Exception as e:
            st.error(f"Error loading image: {e}. Please upload a valid image file.")
