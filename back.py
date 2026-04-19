import numpy as np
import joblib
import tensorflow as tf
from PIL import ImageOps
import numpy as np

# Load Models

crop_model = joblib.load("smart_crop_model.pkl")
crop_mapping = joblib.load("crop_mapping.pkl")

soil_model = tf.keras.models.load_model("soil_analysis_model.h5")
disease_model = tf.keras.models.load_model("plant_disease_model.h5")


# -----------------------
# Crop Prediction
# -----------------------

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    prediction = crop_model.predict(data)[0]

    return crop_mapping[prediction]


# -----------------------
# Soil Prediction
# -----------------------

def predict_soil(image):

    img = ImageOps.fit(image, (224,224))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = soil_model.predict(img)

    soil_classes = [
        "Alluvial Soil",
        "Black Soil",
        "Clay Soil",
        "Red Soil",
        "Sandy Soil"
    ]

    return soil_classes[np.argmax(prediction)]

import requests

API_KEY = "ef738c0ee81af39955550213d0d3285c"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    if data.get("main"):
        result = {
            "city": city,
            "temp": data["main"]["temp"],
            "desc": data["weather"][0]["description"]
        }

        # farming suggestion
        if "rain" in result["desc"]:
            result["advice"] = "🌧️ No irrigation needed"
        else:
            result["advice"] = "💧 Irrigation recommended"

        return result
    else:
        return None
    