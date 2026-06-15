import os
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import get_data_generators

def evaluate_thresholds(predictions, y_true, thresholds, class_names, save_dir='.'):
    """
    Evaluates the predicted probabilities across multiple thresholds without needing to run inference again.
    """
    for threshold in thresholds:
        print(f"\n=========================================")
        print(f"=== EVALUATING THRESHOLD: {threshold:.2f} ===")
        print(f"=========================================")
        
        # Convert probabilities to binary predictions
        y_pred = (predictions > threshold).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # Print metrics
        print(f"Test Accuracy: {accuracy:.4%}")
        print(f"Precision:     {precision:.4%}")
        print(f"Recall:        {recall:.4%}")
        print(f"F1 Score:      {f1:.4%}")
        print(f"Confusion Matrix:\n{cm}")

        # Plot confusion matrix for this threshold
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Threshold: {threshold:.2f})")
        
        save_path = os.path.join(save_dir, f'cm_threshold_t{int(threshold*100)}.png')
        plt.savefig(save_path)
        print(f"=> Plot saved to {save_path}")
        plt.close()

def main():
    DATA_DIR = 'dataset'
    MODEL_PATH = 'model/forgery_model.h5'
    BATCH_SIZE = 32
    TARGET_SIZE = (224, 224)
    # A spectrum of thresholds to test
    THRESHOLDS = [0.20, 0.35, 0.50, 0.65, 0.80]
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        return
        
    print(f"Loading trained neural network from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Loading test data generator...")
    _, _, test_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    
    if test_gen is None:
        print("Error: Test dataset directory missing/empty.")
        return
        
    print(f"\n[1/2] Running model predictions...\n(This only happens once. The results are cached for instant threshold testing)")
    test_gen.reset()
    predictions = model.predict(test_gen, steps=len(test_gen), verbose=1)
    
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    print(f"\n[2/2] Calculating metrics across multiple thresholds...")
    evaluate_thresholds(predictions, y_true, THRESHOLDS, class_names, save_dir='.')
    
    print("\nAll thresholds evaluated successfully! Done.")

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    main()
