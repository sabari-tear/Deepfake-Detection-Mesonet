import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import absl.logging

# Suppress Abseil logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to dataset directories
TRAIN_DIR = 'dataset/train'  # Replace with your actual train dataset path
TEST_DIR = 'dataset/test'   # Replace with your actual test dataset path

# Parameters
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'mesonet.keras'

# Data Augmentation and Loading
def create_data_generators(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize images
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'  # Binary classification: Fake (1) or Real (0)
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, test_generator

# Define the MesoNet Model
def build_model():
    model = Sequential([
        Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),  # Specify input shape here
        Conv2D(8, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(8, (5, 5), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the Model
def train_model():
    # Load data
    train_generator, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR)

    # Build the model
    model = build_model()
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(patience=5, monitor='val_loss', mode='min', restore_best_weights=True)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS
    )

    # Save the final model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return history

# Entry point
if __name__ == "__main__":
    print("Starting training...")
    train_model()
    print("Training complete!")
