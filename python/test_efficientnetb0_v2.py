
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = Path("data_v2/test")
MODEL_DIR = Path("models/efficientnetb0_v2")
RESULT_DIR = Path("results/efficientnetb0_v2")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Load test dataset (never shuffle test images)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["REAL", "FAKE"],
    color_mode="rgb",
    shuffle=False
)

print("\nClass mapping:", test_ds.class_names)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Extract actual labels
y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

models_to_test = {
    "Original": MODEL_DIR / "best_model.keras",
    "Continued": MODEL_DIR / "continued_best_model.keras"
}

all_results = {}

for model_name, model_path in models_to_test.items():

    print("\n" + "=" * 55)
    print("TESTING:", model_name)
    print("=" * 55)

    if not model_path.exists():
        print("Model not found:", model_path)
        continue

    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(
        test_ds,
        verbose=1
    ).reshape(-1)

    y_pred = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "FAKE"],
        digits=4,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(cm)

    all_results[model_name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

# Save comparison
output_path = RESULT_DIR / "test_comparison.json"

with open(output_path, "w") as file:
    json.dump(all_results, file, indent=4)

print("\n" + "=" * 55)
print("TESTING COMPLETED")
print("=" * 55)

print("Results saved to:", output_path)
