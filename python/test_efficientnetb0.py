"""
EfficientNetB0 test script for the AI Fake Photo Detector.

Loads the best validation checkpoint and evaluates only the held-out
*_test_* images in data/real and data/fake.

Expected paths:
    models/efficientnetb0_best.keras
    data/real/*_test_*
    data/fake/*_test_*
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------- CONFIGURATION ----------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "efficientnetb0_best.keras"
RESULTS_DIR = PROJECT_DIR / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
THRESHOLD = 0.5

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run python/train_efficientnetb0.py first."
    )


# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)


# ---------------- COLLECT HELD-OUT TEST IMAGES ----------------
paths = []
labels = []

for folder, label in [("real", 0), ("fake", 1)]:
    class_dir = DATA_DIR / folder

    if not class_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {class_dir}")

    for path in sorted(class_dir.glob("*_test_*")):
        if path.is_file():
            paths.append(str(path))
            labels.append(label)

if not paths:
    raise RuntimeError(
        "No test images found. Expected *_test_* files in "
        "data/real and data/fake."
    )

paths = np.asarray(paths)
labels = np.asarray(labels, dtype=np.int32)

print(f"Test images: {len(paths)}")
print("REAL = 0 | FAKE = 1")


# ---------------- IMAGE PREPROCESSING ----------------
def load_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # Match training: EfficientNetB0 performs its own internal rescaling.
    return tf.cast(image, tf.float32)


test_ds = tf.data.Dataset.from_tensor_slices(paths)
test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------- PREDICTION ----------------
# Output is the model's FAKE probability because FAKE is label 1.
fake_probabilities = model.predict(test_ds, verbose=1).reshape(-1)
predictions = (fake_probabilities >= THRESHOLD).astype(np.int32)

accuracy = accuracy_score(labels, predictions)
cm = confusion_matrix(labels, predictions, labels=[0, 1])

print(f"\nEfficientNetB0 Test Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix (rows=actual, columns=predicted)")
print("                 Pred REAL  Pred FAKE")
print(f"Actual REAL      {cm[0, 0]:9d}  {cm[0, 1]:9d}")
print(f"Actual FAKE      {cm[1, 0]:9d}  {cm[1, 1]:9d}")

print("\nClassification Report:")
print(
    classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        digits=4,
        zero_division=0,
    )
)


# ---------------- SAVE PREDICTIONS ----------------
results = pd.DataFrame(
    {
        "image": [Path(path).name for path in paths],
        "actual": ["REAL" if value == 0 else "FAKE" for value in labels],
        "predicted": [
            "REAL" if value == 0 else "FAKE" for value in predictions
        ],
        "real_probability": 1.0 - fake_probabilities,
        "fake_probability": fake_probabilities,
        "correct": labels == predictions,
    }
)

output_path = RESULTS_DIR / "efficientnetb0_test_predictions.csv"
results.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(f"Correct: {(labels == predictions).sum()}/{len(labels)}")
print(f"Incorrect: {(labels != predictions).sum()}/{len(labels)}")
