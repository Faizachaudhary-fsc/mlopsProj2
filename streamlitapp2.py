# Importing libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# App Title
st.title("Digit Classification with Neural Networks")
st.write("""
Upload an image of a digit (0-9) and the app will predict its label using a pre-trained neural network model.
""")

# Load the pre-trained model (Assuming model.h5 is the pre-trained model)
model = load_model('model.h5')

# Accepting image upload from user
st.write("### Upload an Image of a Digit (0-9) for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((8, 8))  # Resize to match the dataset (8x8 pixels)
    img_array = img_to_array(img) / 16.0  # Scale pixel values to match training data (0-16 range)
    img_array = img_array.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    pred = model.predict(img_array)
    predicted_digit = np.argmax(pred)
    
    st.write(f"### Predicted Digit: {predicted_digit}")
