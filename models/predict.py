import os
import numpy as np
import joblib
import tensorflow as tf
from PIL import ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_MODEL_PATH = os.path.join(BASE_DIR, "smart_crop_model.pkl")
CROP_MAPPING_PATH = os.path.join(BASE_DIR, "crop_mapping.pkl")
SOIL_MODEL_PATH = os.path.join(BASE_DIR, "soil_analysis_model.h5")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")

crop_model = None
crop_mapping = None
soil_model = None
disease_model = None
model_errors = {}  # Track model loading errors


def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        error_msg = f"Failed to load {os.path.basename(path)}: {exc}"
        print(error_msg)
        model_errors[os.path.basename(path)] = str(exc)
        return None


def safe_load_tf_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as exc:
        error_msg = f"Failed to load {os.path.basename(path)}: {exc}"
        print(error_msg)
        model_errors[os.path.basename(path)] = str(exc)
        return None


def load_crop_model():
    global crop_model, crop_mapping
    if crop_model is None or crop_mapping is None:
        crop_model = safe_load_joblib(CROP_MODEL_PATH)
        crop_mapping = safe_load_joblib(CROP_MAPPING_PATH)
    return crop_model, crop_mapping


def load_soil_model():
    global soil_model
    if soil_model is None:
        soil_model = safe_load_tf_model(SOIL_MODEL_PATH)
    return soil_model


def load_disease_model():
    global disease_model
    if disease_model is None:
        disease_model = safe_load_tf_model(DISEASE_MODEL_PATH)
    return disease_model


# -----------------------
# Helper function to format results
# -----------------------

def format_result(success, message):
    """Return a structured result with success flag."""
    return {"success": success, "message": message}


# -----------------------
# Crop Prediction
# -----------------------

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    crop_model, crop_mapping = load_crop_model()
    if crop_model is None or crop_mapping is None:
        return format_result(False, "Crop model not available")

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    try:
        prediction = crop_model.predict(data)[0]
        crop_name = crop_mapping[prediction]
        return format_result(True, crop_name)
    except Exception as e:
        return format_result(False, f"Prediction error: {e}")


# -----------------------
# Soil Prediction
# -----------------------

def predict_soil(image):
    soil_model = load_soil_model()
    if soil_model is None:
        return format_result(False, "Soil model not available")

    try:
        img = ImageOps.fit(image, (224, 224))
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = soil_model.predict(img)
    except Exception as e:
        return format_result(False, f"Soil prediction error: {e}")

    soil_classes = [
        "Alluvial Soil",
        "Black Soil",
        "Clay Soil",
        "Red Soil",
        "Sandy Soil"
    ]

    soil_type = soil_classes[np.argmax(prediction)]
    return format_result(True, soil_type)


# -----------------------
# Disease Prediction
# -----------------------

def predict_disease(image):
    disease_model = load_disease_model()
    if disease_model is None:
        return format_result(False, "Disease model not available")

    try:
        img = ImageOps.fit(image, (224, 224))
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = disease_model.predict(img)
    except Exception as e:
        return format_result(False, f"Disease prediction error: {e}")

    disease_classes = [
        "Healthy",
        "Disease A",
        "Disease B",
        "Disease C"
    ]

    disease_name = disease_classes[np.argmax(prediction)]
    return format_result(True, disease_name)

