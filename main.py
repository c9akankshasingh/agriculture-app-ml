import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Agriculture AI", layout="wide")

# --- LOAD MODELS ---
# Model load karne se pehle check kar rahe hain ki file maujood hai ya nahi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
soil_model_path = os.path.join(BASE_DIR, "soil_analysis_model.h5")

@st.cache_resource
def load_models():
    try:
        # Yahan aap apne models load karein
        # Agar disease detection ka alag model hai toh wo bhi yahan load kar sakte hain
        s_model = tf.keras.models.load_model(soil_model_path)
        return s_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

soil_model = load_models()

# --- BACKEND FUNCTIONS (Functions jo pehle back.py mein thi) ---

def predict_crop(n, p, k, temp, hum, ph, rain):
    # Yahan aapka Crop Recommendation ka logic aayega
    # Sample logic (Aap isse apne logic se replace karein):
    if ph < 6:
        return "Rice (Chawal)"
    elif temp > 30:
        return "Millets (Bajra)"
    else:
        return "Maize (Makka)"

def predict_soil(image):
    if soil_model is None:
        return "Model not loaded"
    
    # Image preprocessing
    size = (224, 224) # Jo bhi aapke model ka size ho
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Prediction
    prediction = soil_model.predict(data)
    # Maan lijiye classes ye hain:
    classes = ["Alluvial", "Black", "Red", "Clayey"]
    return classes[np.argmax(prediction)]

def model_prediction(test_image):
    # Plant Disease Detection Logic
    # Agar iska alag model hai toh yahan uska logic aayega
    return "Healthy Leaf (Sample Result)"

# --- STREAMLIT UI ---

st.title("🌾 Smart Agriculture AI System")

tab1, tab2, tab3 = st.tabs([
    "Crop Recommendation", 
    "Soil Analysis", 
    "Plant Disease Detection"
])

# TAB 1: Crop Recommendation
with tab1:
    st.subheader("Enter Soil and Weather Details")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen", value=0)
        P = st.number_input("Phosphorus", value=0)
        K = st.number_input("Potassium", value=0)
        ph = st.number_input("Soil pH", value=7.0)
    with col2:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=50.0)
        rainfall = st.number_input("Rainfall (mm)", value=100.0)

    if st.button("Predict Crop"):
        result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"Recommended Crop: **{result}**")

# TAB 2: Soil Analysis
with tab2:
    st.subheader("Upload Soil Image")
    soil_file = st.file_uploader("Upload soil photo", type=["jpg","png","jpeg"], key="soil_up")

    if soil_file:
        img = Image.open(soil_file)
        st.image(img, width=300)
        if st.button("Predict Soil Type"):
            with st.spinner("Analyzing soil..."):
                res = predict_soil(img)
                st.success(f"Soil Type: **{res}**")

from back import get_weather

city = input("Enter city name: ")

data = get_weather(city)

if data:
    print("\n🌍 City:", data["city"])
    print("🌡️ Temperature:", data["temp"], "°C")
    print("☁️ Condition:", data["desc"])
    print(data["advice"])
else:
    print("❌ City not found!")