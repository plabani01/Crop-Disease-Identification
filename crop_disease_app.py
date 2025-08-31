import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# File paths
MODEL_PATH = "crop_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"

# Try loading model and class names
model = None
class_names = []

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"  Error loading model or class names: {e}")
        st.stop()
else:
    st.error("  No model available. Please train (`train_model.py`) first to generate `crop_disease_model.h5` and `class_names.json`.")
    st.stop()


# =9 Prediction function
def predict_image(image):
    img = image.resize((224, 224))  # resize to training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence


# =9 Streamlit UI
st.set_page_config(page_title="Crop Disease Identification", page_icon="<1")
st.title("<1 Crop Disease Identification App")
st.write("Upload a crop image, and the model will identify the disease.")

uploaded_file = st.file_uploader("=ä Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="=¼ Uploaded Image", use_container_width=True)

    if st.button("= Predict"):
        with st.spinner("Analyzing... Please wait ó"):
            predicted_class, confidence = predict_image(image)
        st.success(f" Prediction: **{predicted_class}**")
        st.info(f"=Ê Confidence: {confidence*100:.2f}%")
