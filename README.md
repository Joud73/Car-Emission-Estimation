# Car Emission Estimation Pipeline  

This repository contains two main components for estimating car gas emissions using video and audio data:  

1. **Full_pipline.py**  
   - A complete pipeline that runs on Raspberry Pi.  
   - Captures short video+audio clips when triggered by sound.  
   - Detects cars using YOLO, tracks them across the frame, and estimates velocity.  
   - Extracts audio segments synchronized with car crossings.  
   - Pairs car images with audio clips.  
   - Predicts car model using a trained EfficientNetV2 model.  
   - Retrieves emission values from a reference CSV and attaches them to predictions.  
   - Supports GPIO button/LED control for real-time recording and processing.  

2. **CarModel_train.ipynb**  
   - Jupyter notebook for training the car classification model.  
   - Uses EfficientNetV2 for fine-grained car model recognition.  
   - Includes data preprocessing, training, and evaluation steps.  
   - Produces model weights used in the pipeline.  

## Dataset  
The model was trained on a **combined dataset** consisting of the [VMMRdb dataset](https://github.com/faezetta/VMMRdb) (Vehicle Make and Model Recognition Database) together with a **custom scraped dataset** of car images collected for additional coverage.

## Requirements  
- Python 3.9+  
- PyTorch, torchvision, timm  
- OpenCV, librosa, pandas, soundfile  
- YOLOv8 (ultralytics)  
- ffmpeg  
- Raspberry Pi GPIO libraries (if running on Pi)  

## Usage  
1. Train the model using `CarModel_train.ipynb` and save weights.  
2. Update paths to model weights and CSV files in `Full_pipline.py`.  
3. Run `Full_pipline.py` on Raspberry Pi:  
   ```bash
   python3 Full_pipline.py

