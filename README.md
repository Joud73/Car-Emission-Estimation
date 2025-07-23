# Car Sound & Image Dataset Pipeline

This repository contains a complete, step-by-step pipeline for building a machine learning system that estimates car models and CO₂ emissions from real-world driving videos.

## Project Overview

This project extracts car images and sounds from a video, classifies the car model, links it with CO₂ emission data, and trains models to perform:

- Car model detection from images
- CO₂ emission estimation from engine sounds

---

## Notebooks Overview (in order)

### 1. `car_audio_pairing.ipynb`
Extracts:
- Best frames of passing cars using YOLO
- Corresponding engine sounds from video audio
- Output: Paired car images and audio clips

### 2. `car_model_detection_clip.ipynb`
Uses CLIP to:
- Predict the car model for each image
- Output: `{image → predicted_model}`

### 3. `dataset_integration.ipynb`
Integrates:
- Your detected images with an external car dataset
- Unifies file structure and model labels

### 4. `train_car_classifier.ipynb`
Trains an image classification model (e.g., EfficientNet or ViT) to:
- Predict car model from images
- Output: Trained car recognition model

### 5. `link_audio_to_emissions.ipynb`
Links:
- Car sounds to their predicted model
- Models to their CO₂ emissions data
- Output: `{audio → car_model → CO₂}` dataset

### 6. `train_audio_co2_model.ipynb`
Trains a regression model to:
- Predict CO₂ emissions from engine sound features
- Output: Sound-to-emission estimation model
