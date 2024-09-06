# Importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# App Title
st.title("Digit Classification with Neural Networks")
st.write("""
This app trains a neural network model on the digits dataset using TensorFlow and scikit-learn. 
It classifies digits (0-9) and predicts the label of an uploaded image.
""")

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Display the dataset shape and description
st.write("### Dataset Overview")
st.write(f"Data shape: {X.shape}")
st.write("Sample Digits Dataset (first 10 rows):")
st.dataframe(pd.DataFrame(X[:10]))

# One-hot encode the labels
y = to_categorical(y)

# Neural network model architecture
def create_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 output classes for digits 0-9
    return model

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compile and train the model
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Show a progress bar during training
with st.spinner('Training the model...'):
    history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)

# Step 6: Evaluate the Model
accuracy = model.evaluate(X_test, y_test)
st.success(f"Model trained successfully! Test Accuracy: {accuracy[1] * 100:.2f}%")

# Display accuracy and summary
st.write(f"### Model Accuracy: {accuracy[1] * 100:.2f}%")
st.write(f"Training completed over 200 epochs with a batch size of 10.")

# Confusion Matrix and Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

st.write("### Confusion Matrix")
conf_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred), 
                           index=[i for i in range(10)], 
                           columns=[i for i in range(10)])
st.write(conf_matrix)

st.write("### Classification Report")
report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)    

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