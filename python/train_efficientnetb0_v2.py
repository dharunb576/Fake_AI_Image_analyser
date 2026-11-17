
"""
EfficientNetB0 V2 Training
AI Fake Photo Detector

Input: 224x224 RGB
Dataset: data_v2
Classes: REAL=0, FAKE=1
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
SEED = 42

DATA_DIR = Path("data_v2")
MODEL_DIR = Path("models/efficientnetb0_v2")
RESULT_DIR = Path("results/efficientnetb0_v2")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# -------------------------------------------------
# DATASET LOADING
# -------------------------------------------------

def load_split(split, shuffle):

    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=["REAL", "FAKE"],
        color_mode="rgb",
        shuffle=shuffle,
        seed=SEED
    )


train_ds = load_split("train", True)
val_ds = load_split("val", False)
test_ds = load_split("test", False)

print("Class mapping:", train_ds.class_names)

# Verify mapping
assert train_ds.class_names == ["REAL", "FAKE"]

# -------------------------------------------------
# PERFORMANCE OPTIMIZATION
# -------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------------------------------------
# DATA AUGMENTATION
# -------------------------------------------------

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="augmentation")

# -------------------------------------------------
# MODEL
# -------------------------------------------------

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)

# EfficientNetB0 includes its own input rescaling.
# Pass pixel values in the 0-255 range.
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------

checkpoint_path = MODEL_DIR / "best_model.keras"

callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# -------------------------------------------------
# STAGE 1: TRANSFER LEARNING
# -------------------------------------------------

print("\nSTAGE 1: Training classification head")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks
)

# -------------------------------------------------
# STAGE 2: FINE-TUNING
# -------------------------------------------------

print("\nSTAGE 2: Fine-tuning EfficientNetB0")

base_model.trainable = True

# Freeze earlier layers; fine-tune upper layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# -------------------------------------------------
# LOAD BEST MODEL
# -------------------------------------------------

best_model = tf.keras.models.load_model(checkpoint_path)

final_path = MODEL_DIR / "efficientnetb0_v2.keras"

best_model.save(final_path)

print("\nBest model saved:", final_path)

# -------------------------------------------------
# TEST EVALUATION
# -------------------------------------------------

print("\nEvaluating on unseen test data")

test_loss, test_accuracy, test_precision, test_recall = (
    best_model.evaluate(test_ds)
)

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

probabilities = best_model.predict(test_ds, verbose=1).reshape(-1)

y_pred = (probabilities >= 0.5).astype(int)

print("\nTest Accuracy:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
report = classification_report(
    y_true,
    y_pred,
    target_names=["REAL", "FAKE"],
    digits=4
)

print(report)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------

results = {
    "image_size": IMG_SIZE,
    "class_mapping": {
        "REAL": 0,
        "FAKE": 1
    },
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "test_precision": float(test_precision),
    "test_recall": float(test_recall),
    "confusion_matrix": cm.tolist()
}

with open(RESULT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=4)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("EfficientNetB0 V2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["REAL", "FAKE"])
plt.yticks([0, 1], ["REAL", "FAKE"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix.png")
plt.close()

print("\nTraining completed successfully.")
