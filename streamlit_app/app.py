import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource  # Caches the model to avoid reloading on every run
def load_my_model():
    return load_model("models/currency_model.keras")

model = load_my_model()

# Streamlit UI
st.title("💵 Fake Currency Detection")
st.write("Upload an image of currency to check if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_container_width=True)


    # Preprocess image
    def preprocess_image(img):
        img = img.resize((128, 128))  # Resize to model's input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    img_array = preprocess_image(image_display)

    # Make prediction
    prediction = model.predict(img_array)[0][0]  # Extracts scalar value

    # Display result
    if prediction > 0.5:
        st.markdown("### 🟢 **Real currency**", unsafe_allow_html=True)
    else:
        st.markdown("### 🔴 **fake Currency**", unsafe_allow_html=True)

