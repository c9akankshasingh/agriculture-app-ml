import os
import numpy as np
import joblib
import tensorflow as tf
from PIL import ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Failed to load {os.path.basename(path)}: {exc}")
        return None


def safe_load_tf_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as exc:
        print(f"Failed to load {os.path.basename(path)}: {exc}")
        return None

crop_model = safe_load_joblib(os.path.join(BASE_DIR, "smart_crop_model.pkl"))
crop_mapping = safe_load_joblib(os.path.join(BASE_DIR, "crop_mapping.pkl"))
soil_model = safe_load_tf_model(os.path.join(BASE_DIR, "soil_analysis_model.h5"))
disease_model = safe_load_tf_model(os.path.join(BASE_DIR, "plant_disease_model.h5"))


# -----------------------
# Crop Prediction
# -----------------------

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    if crop_model is None or crop_mapping is None:
        return "Crop model not available"

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        prediction = crop_model.predict(data)[0]
        return crop_mapping[prediction]
    except Exception as e:
        return f"Prediction error: {e}"


# -----------------------
# Soil Prediction
# -----------------------

def predict_soil(image):
    if soil_model is None:
        return "Soil model not available"

    img = ImageOps.fit(image, (224, 224))
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


# -----------------------
# Disease Prediction
# -----------------------

def predict_disease(image):
    if disease_model is None:
        return "Disease model not available"

    img = ImageOps.fit(image, (224, 224))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = disease_model.predict(img)
    disease_classes = [
        "Healthy",
        "Disease A",
        "Disease B",
        "Disease C"
    ]

    return disease_classes[np.argmax(prediction)]

