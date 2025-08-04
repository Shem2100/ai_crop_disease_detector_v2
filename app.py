# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model("crop_disease_model.h5")

# Class labels
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus'
]

# Streamlit UI
st.set_page_config(page_title="AI Crop Disease Detector", layout="centered")

st.title("🌿 AI Crop Disease Detection")
st.markdown("Upload a crop leaf image to detect possible diseases using a trained deep learning model.")

uploaded_file = st.file_uploader("📁 Upload a leaf image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = np.array(image.resize((128, 128)))
    if img.shape[-1] == 4:  # RGBA to RGB
        img = img[..., :3]
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🧠 Predicted Disease: **{CLASS_NAMES[class_index]}**")
    st.info(f"📊 Confidence Score: {confidence:.2%}")

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AI Crop Disease Detector", layout="centered")

# ================== CUSTOM STYLING ==================
st.markdown("""
    <style>
    .main {
        background-color: #f7f9f9;
        padding: 2rem;
    }
    .css-1v0mbdj.edgvbvh3 {
        background-color: #1a5e20 !important;
    }
    .stButton>button {
        background-color: #2b9348;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #2b9348;
    }
    </style>
""", unsafe_allow_html=True)

# ================== LOGO ==================
st.image("assets/logo.png", width=150)

# ================== TITLE ==================
st.title("🌿 AI Crop Disease Detector")

# ================== TABS ==================
tabs = st.tabs(["📤 Upload Image", "📈 Model Info", "💡 About"])

# ================== TAB 1: Upload Image ==================
with tabs[0]:
    st.header("Upload a Crop Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("🧠 Predict Disease"):
            model = tf.keras.models.load_model("crop_disease_model.h5")
            image_array = np.array(image.resize((128, 128))) / 255.0
            image_array = image_array.reshape(1, 128, 128, 3)

            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction)

            classes = [
                'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'PlantVillage',
                'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
                'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy',
                'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus'
            ]

            st.success(f"🩺 Predicted Disease: `{classes[predicted_class]}`")

# ================== TAB 2: Model Info ==================
with tabs[1]:
    st.header("Model Performance")
    st.markdown("""
    - **Model Type**: CNN (Convolutional Neural Network)
    - **Accuracy**: ~91%
    - **Input Size**: 128x128 pixels
    - **Framework**: TensorFlow + Keras
    - **Dataset**: PlantVillage
    """)

# ================== TAB 3: About ==================
with tabs[2]:
    st.header("About This Project")
    st.markdown("""
    This AI tool helps farmers and agricultural experts detect plant diseases using deep learning.

    Built by Shem Otieno using:
    - TensorFlow
    - Streamlit
    - PlantVillage Dataset
    """)


