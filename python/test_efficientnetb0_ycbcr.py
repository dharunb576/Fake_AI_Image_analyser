"""
Evaluate the YCbCr + EfficientNetB0 model on the held-out test set.

Expected dataset layout:
    data/
      real/
        *_test_*.jpg
      fake/
        *_test_*.jpg

Loads:
    models/efficientnetb0_ycbcr_best.keras

Uses exactly the same RGB -> YCbCr conversion and 224x224 resizing
as train_efficientnetb0_ycbcr.py.
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
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "efficientnetb0_ycbcr_best.keras"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "efficientnetb0_ycbcr_test_predictions.csv"


# ---------------- COLLECT TEST IMAGES ----------------
def collect_test_images():
    paths, labels = [], []

    for folder, label in [("real", 0), ("fake", 1)]:
        class_dir = DATA_DIR / folder

        if not class_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {class_dir}")

        for path in sorted(class_dir.glob("*")):
            if not path.is_file():
                continue

            if (
                "_test_" in path.name.lower()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ):
                paths.append(str(path))
                labels.append(label)

    if not paths:
        raise RuntimeError(
            "No held-out test images found. Expected *_test_* files "
            "inside data/real and data/fake."
        )

    paths = np.asarray(paths)
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) != 2:
        raise RuntimeError("Both REAL and FAKE test classes are required.")

    print(f"Test images: {len(paths)}")
    print(f"REAL = {int(np.sum(labels == 0))} | FAKE = {int(np.sum(labels == 1))}")

    return paths, labels


# ---------------- RGB TO YCbCr ----------------
def rgb_to_ycbcr(image):
    """Convert RGB float pixels [0,255] to Y, Cb, Cr channels."""
    image = tf.cast(image, tf.float32)

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]

    y = 0.299000 * red + 0.587000 * green + 0.114000 * blue
    cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.500000 * blue
    cr = 128.0 + 0.500000 * red - 0.418688 * green - 0.081312 * blue

    return tf.clip_by_value(tf.stack([y, cb, cr], axis=-1), 0.0, 255.0)


# ---------------- TEST IMAGE PIPELINE ----------------
def load_test_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = rgb_to_ycbcr(image)

    return image, tf.cast(label, tf.float32)


def make_test_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        load_test_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------- EVALUATION ----------------
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train_efficientnetb0_ycbcr.py first."
        )

    test_paths, y_true = collect_test_images()
    test_ds = make_test_dataset(test_paths, y_true)

    print(f"\nLoading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nRunning YCbCr EfficientNetB0 evaluation...")
    probabilities = model.predict(test_ds, verbose=1).ravel()
    y_pred = (probabilities >= 0.5).astype(np.int32)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\nYCbCr EfficientNetB0 Test Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion Matrix (rows=actual, columns=predicted)")
    print("                 Pred REAL  Pred FAKE")
    print(f"Actual REAL      {cm[0, 0]:9d}  {cm[0, 1]:9d}")
    print(f"Actual FAKE      {cm[1, 0]:9d}  {cm[1, 1]:9d}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            digits=4,
            zero_division=0,
        )
    )

    predictions = pd.DataFrame(
        {
            "image_path": test_paths,
            "actual_label": np.where(y_true == 0, "REAL", "FAKE"),
            "predicted_label": np.where(y_pred == 0, "REAL", "FAKE"),
            "real_probability": 1.0 - probabilities,
            "fake_probability": probabilities,
            "correct": y_true == y_pred,
        }
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    correct = int(np.sum(y_true == y_pred))
    incorrect = int(len(y_true) - correct)

    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Correct: {correct}/{len(y_true)}")
    print(f"Incorrect: {incorrect}/{len(y_true)}")


if __name__ == "__main__":
    main()
