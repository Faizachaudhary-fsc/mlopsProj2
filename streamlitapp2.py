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

# App Title
st.title("Digit Classification with Neural Networks")
st.write("""
This app trains a neural network model on the digits dataset using TensorFlow and scikit-learn. 
It classifies digits (0-9) and reports model accuracy.
""")
# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode the labels
y = to_categorical(y)

# Create a neural network model
model = Sequential()
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=10)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy[1] * 100:.2f}%")