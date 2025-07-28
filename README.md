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

### 2. `CLIP_Evaluate.ipynb`
Uses CLIP to:
- Predict the car model for each image
- Output: `{image → predicted_model}`

### 3. `TrainCarClassifier.ipynb`
Integrates:
- our dataset with an VMMRdb car dataset
- Unifies file structure and model labels

- Trains an image classification model (EfficientNet-B0) to:
- Predict car model from images
- Output: Trained car recognition model

You can find the dataset [here](https://drive.google.com/drive/folders/1B6m2mo-CyEaEXO0TD1EcsxyxKIK_REmo?usp=drive_link)
## TO DO:
### 4. `AudioToEmission.ipynb`
Links:
- Car sounds to their predicted model
- Models to their CO₂ emissions data
- Output: `{audio → car_model → CO₂}` dataset
Trains a regression model to:
- Predict CO₂ emissions from engine sound features
- Output: Sound-to-emission estimation model
