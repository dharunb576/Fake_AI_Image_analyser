"""
EfficientNetB0 training script for the AI Fake Photo Detector.

Expected dataset layout:
    project/
      data/
        real/
          *_train_*.jpg
          *_test_*.jpg
        fake/
          *_train_*.jpg
          *_test_*.jpg
      models/
      results/
      python/
        train_efficientnetb0.py

The script:
- Uses only *_train_* images for training/validation.
- Creates a stratified 80:20 train/validation split (seed 42).
- Keeps *_test_* images out of training and validation.
- Uses ImageNet-pretrained EfficientNetB0.
- Trains a classification head, then fine-tunes the final layers.
- Saves the best validation checkpoint and training plots.

EfficientNet preprocessing note:
tf.keras.applications.EfficientNetB0 includes a Rescaling layer internally.
Therefore, this script supplies resized float32 pixels in the 0..255 range
and does NOT apply MobileNetV2 preprocess_input (which scales to -1..1).
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


# ---------------- CONFIGURATION ----------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 12
EPOCHS_FINE = 12

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- DATASET ----------------
def collect_training_images():
    """Collect training images and derive labels from their parent folders."""
    paths, labels = [], []

    for folder, label in [("real", 0), ("fake", 1)]:
        class_dir = DATA_DIR / folder

        if not class_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {class_dir}")

        for path in sorted(class_dir.glob("*")):
            if not path.is_file():
                continue

            name = path.name.lower()

            # Only use the designated training portion.
            # Test images must never enter training or validation.
            if "_train_" in name:
                paths.append(str(path))
                labels.append(label)

    if not paths:
        raise RuntimeError(
            "No training images found. Expected *_train_* files in "
            "data/real and data/fake."
        )

    paths = np.asarray(paths)
    labels = np.asarray(labels, dtype=np.int32)

    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2:
        raise RuntimeError("Both REAL and FAKE training classes are required.")

    print(f"Total training-pool images: {len(paths)}")
    print(f"REAL: {int(np.sum(labels == 0))}")
    print(f"FAKE: {int(np.sum(labels == 1))}")

    return paths, labels


image_paths, labels = collect_training_images()

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

print(f"\nTraining images: {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print("Held-out test images are excluded from training and validation.")


# ---------------- IMAGE PIPELINE ----------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    # EfficientNetB0 has its own Rescaling layer; keep pixels in 0..255.
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels_array, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_array))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)


# ---------------- MODEL ----------------
base_model = EfficientNetB0(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Phase 1: train only the new classification head.
base_model.trainable = False

inputs = layers.Input(shape=(*IMG_SIZE, 3), name="image")
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
x = layers.Dropout(0.30, name="dropout_1")(x)
x = layers.Dense(128, activation="relu", name="dense_features")(x)
x = layers.Dropout(0.20, name="dropout_2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

model = models.Model(inputs, outputs, name="EfficientNetB0_Fake_Image_Detector")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

BEST_MODEL_PATH = MODEL_DIR / "efficientnetb0_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "efficientnetb0_final.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ---------------- PHASE 1: CLASSIFIER HEAD ----------------
print("\nPhase 1: Training EfficientNetB0 classification head...")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)


# ---------------- PHASE 2: FINE-TUNING ----------------
print("\nPhase 2: Fine-tuning the final EfficientNetB0 layers...")
base_model.trainable = True

# Fine-tune only the final 30 layers; keep earlier pretrained features fixed.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for more stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks,
)

# Save the final model state. The best validation checkpoint is saved separately.
model.save(FINAL_MODEL_PATH)


# ---------------- SAVE TRAINING PLOTS ----------------
def combine_history(*histories):
    combined = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(values)
    return combined


history = combine_history(history_head, history_fine)

plt.figure()
plt.plot(history.get("accuracy", []), label="Training accuracy")
plt.plot(history.get("val_accuracy", []), label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNetB0 Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "efficientnetb0_accuracy.png", dpi=150)
plt.close()

plt.figure()
plt.plot(history.get("loss", []), label="Training loss")
plt.plot(history.get("val_loss", []), label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "efficientnetb0_loss.png", dpi=150)
plt.close()

print("\nEfficientNetB0 training complete!")
print(f"Best checkpoint: {BEST_MODEL_PATH}")
print(f"Final model: {FINAL_MODEL_PATH}")
print(f"Accuracy plot: {RESULTS_DIR / 'efficientnetb0_accuracy.png'}")
print(f"Loss plot: {RESULTS_DIR / 'efficientnetb0_loss.png'}")
