import streamlit as st
from PIL import Image

from models.predict import predict_crop, predict_soil, predict_disease
from api.weather import get_weather_data

st.set_page_config(page_title="Smart Agriculture AI", layout="wide")

st.title("Smart Agriculture AI System")

if "weather" not in st.session_state:
    st.session_state.weather = None

with st.sidebar:
    st.header("Weather Data")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=25.873200918122755, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=82.09919645244908, format="%.4f")
    if st.button("Load Weather Data"):
        try:
            st.session_state.weather = get_weather_data(latitude, longitude)
        except Exception as e:
            st.error(f"Unable to load weather data: {e}")
    if st.session_state.weather:
        weather = st.session_state.weather
        st.metric("Temperature", f"{weather.get('temperature', 'N/A')} °C")
        st.metric("Humidity", f"{weather.get('humidity', 'N/A')} %")
        st.metric("Precipitation", f"{weather.get('precipitation', 'N/A')} mm")
        st.write(f"Location: {weather.get('latitude')}, {weather.get('longitude')} ({weather.get('timezone')})")


tab1, tab2, tab3 = st.tabs(["Crop Recommendation", "Soil Analysis", "Plant Disease Detection"])

with tab1:
    st.subheader("Enter Soil and Weather Details")
    weather = st.session_state.weather or {}
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen", value=0)
        P = st.number_input("Phosphorus", value=0)
        K = st.number_input("Potassium", value=0)
        ph = st.number_input("Soil pH", value=7.0)
    with col2:
        temperature = st.number_input("Temperature (°C)", value=weather.get("temperature", 25.0))
        humidity = st.number_input("Humidity (%)", value=weather.get("humidity", 50.0))
        rainfall = st.number_input("Rainfall (mm)", value=weather.get("precipitation", 0.0))

    if st.button("Predict Crop"):
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"Recommended Crop: **{result}**")

with tab2:
    st.subheader("Upload Soil Image")
    soil_file = st.file_uploader("Upload soil photo", type=["jpg", "png", "jpeg"], key="soil_up")
    if soil_file:
        img = Image.open(soil_file)
        st.image(img, width=300)
        if st.button("Predict Soil Type"):
            with st.spinner("Analyzing soil..."):
                res = predict_soil(img)
                st.success(f"Soil Type: **{res}**")

with tab3:
    st.subheader("Upload Crop Leaf Image")
    disease_file = st.file_uploader("Upload leaf photo", type=["jpg", "png", "jpeg"], key="leaf_up")
    if disease_file:
        img = Image.open(disease_file)
        st.image(img, width=300)
        if st.button("Predict Disease"):
            with st.spinner("Analyzing leaf..."):
                res = predict_disease(img)
                st.success(f"Plant Disease Result: **{res}**")
