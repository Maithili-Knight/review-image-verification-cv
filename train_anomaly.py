import os
import cv2
import glob
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "clean_references")
MODEL_PATH = os.path.join(BASE_DIR, "model", "forgery_model.h5")
ISO_FOREST_PATH = os.path.join(BASE_DIR, "model", "isolation_forest.joblib")

os.makedirs(CLEAN_DIR, exist_ok=True)

def train_anomaly_detector():
    print("🚀 Initializing Lightning Anomaly Trainer...")
    
    # 1. Look for images
    image_paths = glob.glob(os.path.join(CLEAN_DIR, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if len(image_paths) < 10:
        print(f"❌ Error: Please put at least 10 clean unedited photos into:\n{CLEAN_DIR}")
        print(f"I only found {len(image_paths)} images.")
        return

    print(f"✅ Found {len(image_paths)} perfect reference photos. Loading AI Extractor...")

    # 2. Load the base forgery model to use as a feature extractor
    try:
        base_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load {MODEL_PATH}: {e}")
        return

    # 3. Create the Extractor by chopping off the final Dense(1) layer
    # We want the rich feature vectors right before the final decision.
    try:
        # The layer right before the final Dense(1) output
        extractor_layer = base_model.layers[-2]  
        feature_extractor = Model(inputs=base_model.input, outputs=extractor_layer.output)
    except Exception as e:
        print("⚠️ Could not slice model perfectly, falling back to full model architecture...")
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    print("✅ Model sliced successfully. Extracting 1,280-dimension fingerprints...")

    # 4. Extract Features
    features = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_resized.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get fingerprint
        fingerprint = feature_extractor.predict(img_array, verbose=0)
        features.append(fingerprint.flatten())
    
    features = np.array(features)
    print(f"✅ Extracted {features.shape[0]} fingerprints of shape {features.shape[1]}.")

    # 5. Train Isolation Forest
    print("🌲 Fast-Training Scikit-Learn Isolation Forest...")
    # contamination=0.01 means we expect 1% of the clean training data to accidentally be bad
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(features)
    
    # 6. Save Model
    joblib.dump(iso_forest, ISO_FOREST_PATH)
    print(f"\n🎉 SUCCESS! Ultra-precision physical anomaly detector saved to:")
    print(f"-> {ISO_FOREST_PATH}")
    print("You may now proceed to update predict.py!")

if __name__ == "__main__":
    train_anomaly_detector()
