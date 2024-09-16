import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Define custom metrics or losses if you used any custom functions
def custom_mae(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

# Load the pre-trained model with custom objects if necessary
model_path = 'Age_Sex_Detection.h5'  # Ensure this path is correct
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'mae': custom_mae})
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define image preprocessing functions
def preprocess_image(image):
    image = np.array(image.convert('RGB'))
    image = cv2.resize(image, (48, 48))  # Resize to match the input shape expected by the model
    image = image / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Define prediction function
def predict_age_gender(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    age = int(np.round(predictions[1][0]))  # Extract age prediction
    gender = int(np.round(predictions[0][0]))  # Extract gender prediction
    gender_label = 'Female' if gender >= 0.5 else 'Male'  # Adjust based on your model's output
    return age, gender_label

# Streamlit app layout
st.title("Age and Gender Prediction")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict age and gender
    try:
        age, gender_label = predict_age_gender(image)
        st.write(f"Predicted Age: {age}")
        st.write(f"Predicted Gender: {gender_label}")
    except Exception as e:
        st.error(f"Error occurred: {e}")
