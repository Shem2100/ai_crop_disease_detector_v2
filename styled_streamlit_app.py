import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from gtts import gTTS
from googletrans import Translator

# Set Streamlit page configuration
st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark-themed custom CSS
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        .stApp {
            background-color: #1e1e1e;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00ffcc;
        }
        .css-1d391kg {
            background-color: #006666 !important;
            color: white !important;
        }
        .css-1cpxqw2, .css-qrbaxs {
            background-color: #2c2c2c;
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("crop_disease_model.h5")
    return model


model = load_model()
translator = Translator()

# Class names and scientific names
CLASS_INFO = {
    'Pepper__bell___Bacterial_spot': 'Capsicum annuum - Bacterial spot',
    'Pepper__bell___healthy': 'Capsicum annuum - Healthy',
    'Potato___Early_blight': 'Solanum tuberosum - Early blight',
    'Potato___healthy': 'Solanum tuberosum - Healthy',
    'Potato___Late_blight': 'Solanum tuberosum - Late blight',
    'Tomato_Bacterial_spot': 'Solanum lycopersicum - Bacterial spot',
    'Tomato_Early_blight': 'Solanum lycopersicum - Early blight',
    'Tomato_healthy': 'Solanum lycopersicum - Healthy',
    'Tomato_Late_blight': 'Solanum lycopersicum - Late blight',
    'Tomato_Leaf_Mold': 'Solanum lycopersicum - Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Solanum lycopersicum - Septoria leaf spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Solanum lycopersicum - Spider mites',
    'Tomato__Target_Spot': 'Solanum lycopersicum - Target Spot',
    'Tomato__Tomato_mosaic_virus': 'Solanum lycopersicum - Tomato mosaic virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Solanum lycopersicum - Yellow Leaf Curl Virus'
}


# Grad-CAM utility
def generate_gradcam_heatmap(model, image_array, last_conv_layer_name="conv2d"):
    grad_model = tf.keras.models.Model([
        model.inputs,
        model.get_layer(last_conv_layer_name).output
    ], model.output)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Audio feedback
def speak_text(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("speech.mp3")
    audio_file = open("speech.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")


# Sidebar
st.sidebar.image("assets/logo.png", width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Predict", "Batch Prediction", "Care Tips", "About"])

# Pages
if page == "Home":
    st.title("Welcome to the AI Crop Disease Detector 🌾")
    st.markdown("""
        This application uses deep learning to identify diseases in crop leaves.

        **Features:**
        - Classifies tomato, potato, and pepper diseases
        - Simple image upload and batch prediction
        - Scientific names and care recommendations
        - Grad-CAM for visual explanation
        - Downloadable results in CSV format
    """)

elif page == "Upload & Predict":
    st.header("Upload a Leaf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    language = st.selectbox("Select Language", ["English", "Swahili", "French"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = np.array(image.resize((128, 128))) / 255.0
        img_input = img.reshape(1, 128, 128, 3)

        prediction = model.predict(img_input)
        predicted_class = list(CLASS_INFO.keys())[np.argmax(prediction)]
        confidence = np.max(prediction)
        sci_name = CLASS_INFO[predicted_class]

        result_text = f"Prediction: {predicted_class}\nScientific Name: {sci_name}\nConfidence: {confidence:.2%}"

        if language != "English":
            translated = translator.translate(result_text, dest='sw' if language == 'Swahili' else 'fr').text
            result_text = translated

        st.success(result_text)
        speak_text(result_text, lang='en' if language == 'English' else 'sw' if language == 'Swahili' else 'fr')

        st.subheader("🧠 Visual Explanation (Grad-CAM)")
        try:
            heatmap = generate_gradcam_heatmap(model, img_input)
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
            st.image(superimposed_img, caption="Grad-CAM Explanation", use_column_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM visualization failed: {str(e)}")

elif page == "Batch Prediction":
    st.header("Batch Image Prediction")
    batch_files = st.file_uploader("Upload multiple images", accept_multiple_files=True)
    results = []

    if batch_files:
        for img_file in batch_files:
            image = Image.open(img_file).convert('RGB')
            img_array = np.array(image.resize((128, 128))) / 255.0
            img_array = img_array.reshape(1, 128, 128, 3)

            pred = model.predict(img_array)
            predicted_class = list(CLASS_INFO.keys())[np.argmax(pred)]
            confidence = np.max(pred)
            sci_name = CLASS_INFO[predicted_class]

            results.append({
                "Filename": img_file.name,
                "Prediction": predicted_class,
                "Scientific Name": sci_name,
                "Confidence": f"{confidence:.2%}"
            })

        df = pd.DataFrame(results)
        st.dataframe(df)

        # Download link
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report", csv, "predictions.csv", "text/csv")

elif page == "Care Tips":
    st.header("Crop Disease Care Tips")
    st.markdown("""
    ✅ **General Tips**:
    - Ensure good air circulation to reduce fungal growth.
    - Avoid overhead watering to prevent water-borne diseases.
    - Rotate crops to reduce soil-borne pathogens.

    🛠️ **Disease-Specific Suggestions**:
    - **Early Blight (Potato/Tomato):** Use copper-based fungicides and remove infected leaves.
    - **Late Blight:** Use resistant varieties, apply fungicides before rainy seasons.
    - **Bacterial Spot:** Avoid working with wet plants, apply bactericides.
    - **Spider Mites:** Use insecticidal soap and maintain humidity.
    """)

elif page == "About":
    st.header("About This App")
    st.markdown("""
        Developed using:
        - **TensorFlow + Keras** for deep learning
        - **Streamlit** for the web app

        For support or improvements, reach out via [GitHub](https://github.com/) or email.
    """)


