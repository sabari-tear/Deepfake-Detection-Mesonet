
# Deepfake Detection using MesoNet

This repository contains the implementation of a deepfake detection system using the MesoNet architecture. The system classifies images as either "Fake" or "Real" using a trained convolutional neural network.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Making Predictions](#making-predictions)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

Deepfake content has grown exponentially with advancements in AI. This project aims to identify deepfake images using a simple yet effective convolutional neural network architecture based on MesoNet. The model was trained to distinguish between real and fake images using labeled datasets.

---

## Features
- **Binary Classification**: Detects whether an image is "Fake" or "Real".
- **Customizable**: Easily adapt the architecture and dataset for different classification tasks.
- **Preprocessing Pipeline**: Includes data augmentation and normalization.
- **Deployment Ready**: Use the trained model to make predictions on unseen data.

---

## Dataset

### Structure
The dataset is organized into two directories:
- `dataset/train`: Contains training images labeled as `fake` and `real`.
- `dataset/test`: Contains testing images labeled as `fake` and `real`.

### Example Directory Structure
```
dataset/
    train/
        fake/
        real/
    test/
        fake/
        real/
unseen/
    fake/
    real/
```

You can replace the dataset content with your own labeled data. Ensure the directory structure remains consistent.

---

## Model Architecture

The implemented MesoNet architecture consists of:
1. Convolutional layers with Batch Normalization and ReLU activation.
2. MaxPooling layers for spatial dimension reduction.
3. Fully connected layers for classification.
4. Dropout for regularization.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sabari-tear/Deepfake-Detection-Mesonet.git
   cd Deepfake-Detection-Mesonet
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model

1. Place your dataset in the `dataset` directory, organized as described above.
2. Train the model by running:
   ```bash
   python train_model.py
   ```
3. After training, the model will be saved as `mesonet.keras` in the project directory.

---

### Making Predictions

1. Use the saved model to predict whether an image is fake or real:
   ```bash
   python predict_image.py <path_to_image>
   ```
2. Example:
   ```bash
   python predict_image.py unseen/fake/example.jpg
   ```

The output will indicate whether the image is "Fake" or "Real" along with the confidence score.

---

## Results

### Model Performance
- **Dataset Used**: Labeled images of deepfake and real content.
- **Accuracy**: Achieved a high accuracy of classification (details can be added after running the model on a validation/test set).

---

## Future Work

- Enhance the dataset with more diverse examples to improve generalization.
- Experiment with advanced architectures like Xception and EfficientNet for better performance.
- Develop a web interface or browser extension for real-time detection.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to contribute to the project by opening issues or submitting pull requests. Happy coding! 😊
