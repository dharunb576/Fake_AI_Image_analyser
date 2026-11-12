
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIGURATION ----------------
SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 10

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- DATASET ----------------
def collect_images():
    paths, labels = [], []

    for folder, label in [("real", 0), ("fake", 1)]:
        for path in sorted((DATA_DIR / folder).glob("*")):
            if not path.is_file():
                continue

            name = path.name.lower()

            # Never use test images for training or validation.
            if "_test_" in name:
                continue

            if "_train_" in name:
                paths.append(str(path))
                labels.append(label)

    if not paths:
        raise RuntimeError("No training images found. Check data/real and data/fake.")

    return np.array(paths), np.array(labels, dtype=np.float32)


image_paths, labels = collect_images()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print(f"Training images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print("Test images are excluded from training and validation.")

# ---------------- IMAGE PIPELINE ----------------
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)

# ---------------- MODEL ----------------
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "mobilenetv2_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

print("\nPhase 1: Training classifier head...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# ---------------- FINE-TUNING ----------------
print("\nPhase 2: Fine-tuning MobileNetV2...")

base_model.trainable = True

# Keep most pretrained layers frozen; fine-tune only the final layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for more stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# Save the final model.
model.save(MODEL_DIR / "mobilenetv2_final.keras")

# Plot training curves.
def combine_history(*histories):
    combined = {}
    for hist in histories:
        for key, values in hist.history.items():
            combined.setdefault(key, []).extend(values)
    return combined


history = combine_history(history_head, history_fine)

plt.figure()
plt.plot(history["accuracy"], label="Training accuracy")
plt.plot(history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mobilenetv2_accuracy.png", dpi=150)
plt.close()

plt.figure()
plt.plot(history["loss"], label="Training loss")
plt.plot(history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mobilenetv2_loss.png", dpi=150)
plt.close()

print("\nTraining complete!")
print(f"Best checkpoint: {MODEL_DIR / 'mobilenetv2_best.keras'}")
print(f"Final model: {MODEL_DIR / 'mobilenetv2_final.keras'}")
print(f"Accuracy plot: {RESULTS_DIR / 'mobilenetv2_accuracy.png'}")
print(f"Loss plot: {RESULTS_DIR / 'mobilenetv2_loss.png'}")
