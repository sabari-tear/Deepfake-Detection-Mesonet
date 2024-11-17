import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import absl.logging

# Suppress Abseil logging
absl.logging.set_verbosity(absl.logging.ERROR)

import sys
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Preprocess the input image
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocesses the input image for prediction.
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size to resize the image (default is (128, 128)).
    
    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        # Load the image
        img = load_img(image_path, target_size=target_size)
        # Convert the image to array
        img_array = img_to_array(img)
        # Rescale pixel values to [0, 1]
        img_array = img_array / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in processing the image: {e}")
        return None

# Load the trained model
def load_trained_model(model_path):
    """
    Loads a trained deepfake detection model.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        tensorflow.keras.Model: Loaded model.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error in loading the model: {e}")
        return None

# Predict whether the image is fake or real
def predict_image(model, image_path):
    """
    Predicts whether the input image is fake or real.
    
    Args:
        model (tensorflow.keras.Model): Trained model.
        image_path (str): Path to the input image.
    
    Returns:
        str: Prediction result ("Fake" or "Real").
    """
    # Preprocess the image
    image = preprocess_image(image_path)
    if image is None:
        return "Error: Unable to process the image."
    
    # Make prediction
    prediction = model.predict(image)[0][0]
    
    # Interpret the result
    result = "Fake" if prediction < 0.5 else "Real"
    print(f"Prediction: {result} (Confidence: {prediction:.2f})")
    return result

# Main function
def main():
    # Check for command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <path_to_image>")
        return
    
    # Path to the input image
    image_path = sys.argv[1]
    
    # Path to the trained model
    model_path = 'mesonet.keras'
    
    # Load the trained model
    model = load_trained_model(model_path)
    if model is None:
        return
    
    # Predict the image
    result = predict_image(model, image_path)
    print(f"Result: {result}")

if __name__ == '__main__':
    main()
