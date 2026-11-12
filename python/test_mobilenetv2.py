
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.keras"
RESULTS_DIR = PROJECT_DIR / "results"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# Load the best model selected using validation accuracy.
model = tf.keras.models.load_model(MODEL_PATH)

# Collect only held-out test images.
paths = []
labels = []

for folder, label in [("real", 0), ("fake", 1)]:
    for path in sorted((DATA_DIR / folder).glob("*_test_*")):
        if path.is_file():
            paths.append(str(path))
            labels.append(label)

if not paths:
    raise RuntimeError("No test images found in data/real and data/fake.")

paths = np.array(paths)
labels = np.array(labels, dtype=np.int32)

print(f"Test images: {len(paths)}")
print("REAL = 0 | FAKE = 1")

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return preprocess_input(image)

dataset = tf.data.Dataset.from_tensor_slices(paths)
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Predict: probability >= 0.5 means FAKE.
probabilities = model.predict(dataset, verbose=1).reshape(-1)
predictions = (probabilities >= 0.5).astype(np.int32)

accuracy = accuracy_score(labels, predictions)
cm = confusion_matrix(labels, predictions, labels=[0, 1])

print(f"\nMobileNetV2 Test Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix (rows=actual, columns=predicted)")
print("                 Pred REAL  Pred FAKE")
print(f"Actual REAL      {cm[0, 0]:9d}  {cm[0, 1]:9d}")
print(f"Actual FAKE      {cm[1, 0]:9d}  {cm[1, 1]:9d}")

print("\nClassification Report:")
print(classification_report(
    labels,
    predictions,
    labels=[0, 1],
    target_names=["REAL", "FAKE"],
    digits=4
))

results = pd.DataFrame({
    "image": [Path(p).name for p in paths],
    "actual": ["REAL" if x == 0 else "FAKE" for x in labels],
    "predicted": ["REAL" if x == 0 else "FAKE" for x in predictions],
    "fake_probability": probabilities,
    "correct": labels == predictions
})

output_path = RESULTS_DIR / "mobilenetv2_test_predictions.csv"
results.to_csv(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(f"Correct: {(labels == predictions).sum()}/{len(labels)}")
print(f"Incorrect: {(labels != predictions).sum()}/{len(labels)}")
