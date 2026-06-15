import os
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from model import build_model
from preprocessing import get_data_generators

def plot_training_history(history, save_dir='.', prefix=''):
    """
    Plots training & validation accuracy and loss values and saves to file.
    """
    # Accuracy plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, f'{prefix}training_accuracy.png'))
    plt.close()

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, f'{prefix}training_loss.png'))
    plt.close()

def evaluate_model(model, test_generator, save_dir='.'):
    """
    Evaluates the model on the test dataset and prints Scikit-learn metrics.
    """
    print("\nEvaluating on Test Data...")
    # Reset generator before prediction
    test_generator.reset()
    
    # Predict probabilities
    predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    class_names = list(test_generator.class_indices.keys())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


def train():
    # Setup directories
    DATA_DIR = 'dataset'
    MODEL_DIR = 'model'
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 32
    TARGET_SIZE = (224, 224)
    
    print("Loading Data Generators...")
    train_gen, val_gen, test_gen = get_data_generators(DATA_DIR, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
    
    if train_gen is None or val_gen is None:
        print("Error: Dataset directories not fully populated. Please ensure 'dataset/train' and 'dataset/val' exist.")
        return

    # Print class mapping for verification
    print(f"Class indices: {train_gen.class_indices}")

    print("Building Model...")
    model = build_model()
    
    # ===== PHASE 1: Train only the head layers (base frozen) =====
    print("\n===== PHASE 1: Training head layers (base frozen) =====")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callbacks for Phase 1
    early_stop_p1 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    
    history_p1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=[early_stop_p1],
    )
    
    print("Phase 1 Complete!")
    plot_training_history(history_p1, save_dir='.', prefix='phase1_')

    # ===== PHASE 2: Fine-tune the entire model (unfreeze base) =====
    print("\n===== PHASE 2: Fine-tuning entire model (base unfrozen) =====")
    
    # Unfreeze the base model
    base_model = model.layers[1]  # EfficientNetB0 is the second layer
    base_model.trainable = True
    
    # Use a much lower learning rate for fine-tuning to avoid destroying pretrained weights
    optimizer_ft = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer_ft, loss=loss, metrics=['accuracy'])
    
    # Callbacks for Phase 2
    early_stop_p2 = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
    
    history_p2 = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        callbacks=[early_stop_p2, reduce_lr],
    )
    
    print("Phase 2 (Fine-tuning) Complete!")
    plot_training_history(history_p2, save_dir='.', prefix='phase2_')

    # Save Model
    save_path = os.path.join(MODEL_DIR, 'forgery_model.h5')
    print(f"Saving model to {save_path}")
    model.save(save_path)
    
    # Test Evaluation
    if test_gen is not None:
        evaluate_model(model, test_gen, save_dir='.')
    else:
        print("No test generator found. Skipping evaluation.")

if __name__ == '__main__':
    train()
