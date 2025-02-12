import streamlit as st
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import preprocess_image  # Import fixed
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'currency_model.keras')
model = load_model(MODEL_PATH)

# Streamlit UI
st.title("Fake Currency Detection")
st.write("Upload an image to check if it's real or fake currency.")

# File uploader
uploaded_file = st.file_uploader("Choose a currency image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Processing...")
    
    # Preprocess the image
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (128, 128))  # Resize to match model input shape
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Prediction
    prediction = model.predict(img_array)
    
    # Display result
    if prediction[0][0] > 0.5:
        st.write("🔴 **Fake Currency Detected!**")
    else:
        st.write("🟢 **Real Currency**")
