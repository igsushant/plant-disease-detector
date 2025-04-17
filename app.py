import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Live Plant Disease Detection")

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Session state to handle reset
if "reset" not in st.session_state:
    st.session_state.reset = False

# Reset button
if st.button("🔄 Reset App"):
    st.session_state.reset = True
    st.experimental_rerun()

# Image input (camera or uploader)
image = st.camera_input("📷 Capture a leaf photo") if not st.session_state.reset else None
if image is None and not st.session_state.reset:
    image = st.file_uploader("📁 Or upload a leaf image", type=["jpg", "jpeg", "png"])

# Prediction
if image:
    img = Image.open(image).convert("RGB")
    st.image(img, caption="Selected Image", use_column_width=True)
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.success(f"🧪 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: `{confidence:.2%}`")
