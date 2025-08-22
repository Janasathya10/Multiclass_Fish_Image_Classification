import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Paths
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Get all .h5 model files
model_files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]
if not model_files:
    raise FileNotFoundError(f"No .h5 files found in {MODELS_DIR} folder!")

# Dataset path
test_dir = os.path.join("data", "test")

# Target size selector
def get_target_size(model_name):
    return (299, 299) if "inception" in model_name.lower() or "xception" in model_name.lower() else (224, 224)

# List to store summary for all models
summary_results = []

# Loop through models
for model_file in model_files:
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    print(f"\nEvaluating model: {model_name}")

    target_size = get_target_size(model_file)

    # Data generator
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    # Load model
    model = tf.keras.models.load_model(model_file)

    # Predictions
    preds = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_classification_report.csv"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.csv"))

    # Add to summary
    summary_results.append({
        "Model": model_name,
        "Accuracy": report["accuracy"],
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1-score": report["macro avg"]["f1-score"],
        "Weighted Precision": report["weighted avg"]["precision"],
        "Weighted Recall": report["weighted avg"]["recall"],
        "Weighted F1-score": report["weighted avg"]["f1-score"]
    })

# Save combined summary
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_summary.csv"), index=False)

print(f"\n✅ All results saved in '{RESULTS_DIR}' folder!")