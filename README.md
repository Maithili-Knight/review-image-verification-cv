# Online Product Review Image Verification System

Automated Verification of Online Product Review Images Using Computer Vision and Pattern Recognition.

This project was built to train a deep learning model to verify whether an uploaded product review image is authentic or tampered. 
The system detects fraudulent or manipulated images that appear in online product reviews, using the **CASIA 2.0 Image Tampering Detection Dataset**.

## Overview
- **Transfer Learning**: Built using `EfficientNetB0` pretrained on ImageNet.
- **Data Augmentation**: Enhances the input data by applying rotations, flips, zoom, and brightness adjustments.
- **Web Interface**: Easy-to-use Streamlit web app for predicting tampering.
- **Computer Vision**: Utilizes OpenCV ORB matching to discover copy-move forgery duplication in images.

## Project Structure
```text
image_verification_project/
│
├── dataset/                    # Ensure dataset is structured into train, val, and test here
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── model/                      # Directory for the trained model output
│   └── forgery_model.h5
│
├── README.md                   # Project documentation
├── requirements.txt            # Required Python libraries
├── model.py                    # EfficientNetB0 deep learning architecture definition
├── preprocessing.py            # Image preprocessing and data augmentation
├── train.py                    # Training pipeline (loss/accuracy metrics computation)
├── predict.py                  # Single image prediction logic and ORB Feature extraction
├── streamlit_app.py            # Streamlit-based web interface
└── frontend/                   # React frontend interacting with the FastAPI backend
```

## 🚀 How to Run the Project

**Important**: Make sure you navigate completely into the project directory before executing any scripts!
```bash
cd image_verification_project
```

### 1. Web Interface (Streamlit Dashboard)
To launch the user-friendly application to upload and verify images visually:
```bash
streamlit run streamlit_app.py
```
*A browser window will open automatically at `http://localhost:8501`.*

### 2. Modern Web App (React Frontend)
In addition to Streamlit, there is a modern React/Vite web application. This frontend requires the Backend API Server to be running.
```bash
cd frontend
npm install
npm run dev
```
*The React application will launch in your browser (typically at `http://localhost:5173`). Make sure your API (`api.py`) is also running first!*

### 3. Backend API Server (FastAPI)
If you want to integrate the model verification into another application or prefer a REST API:
```bash
python api.py
```
*The API will be available at `http://localhost:8000`. You can send a `POST` request with an image file to `/verify` to get the verification status.*

### 3. Model Training
To retrain the EfficientNetB0 classification model using a new dataset:

1. Ensure your dataset is extracted into `image_verification_project/dataset/` with `train`, `val`, and `test` sub-folders appropriately separated into `real` and `fake` classes.
2. Run the main training pipeline:
```bash
python train.py
```
*Training progress (loss and accuracy) plots and evaluation metrics (classification report & confusion matrix) will be saved in the repository root directory.*
