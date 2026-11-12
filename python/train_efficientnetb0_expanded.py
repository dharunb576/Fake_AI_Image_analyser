"""
EfficientNetB0 training on the expanded CIFAKE dataset.

Reads:
    data/expanded_v1/train/REAL
    data/expanded_v1/train/FAKE
    data/expanded_v1/val/REAL
    data/expanded_v1/val/FAKE

The test set is deliberately NOT used during training:
    data/expanded_v1/test/REAL
    data/expanded_v1/test/FAKE

Existing scripts and model files are not overwritten.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


# ---------------- CONFIGURATION ----------------
SEED = 42
IMG_SIZE = (224, 224)

# Start with batch size 8 for the user's CPU-only, 16 GB RAM machine.
BATCH_SIZE = 8

EPOCHS_HEAD = 12
EPOCHS_FINE = 12

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "expanded_v1"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

BEST_MODEL_PATH = MODEL_DIR / "efficientnetb0_expanded_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "efficientnetb0_expanded_final.keras"

ACCURACY_PLOT = RESULTS_DIR / "efficientnetb0_expanded_accuracy.png"
LOSS_PLOT = RESULTS_DIR / "efficientnetb0_expanded_loss.png"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- DATASET COLLECTION ----------------
def collect_split(split_name):
    """Read explicit REAL/FAKE folders for one prepared split."""
    paths, labels = [], []

    for folder_name, label in [("REAL", 0), ("FAKE", 1)]:
        class_dir = DATA_DIR / split_name / folder_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {class_dir}")

        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            raise RuntimeError(f"No supported images found in: {class_dir}")

        paths.extend(str(path) for path in files)
        labels.extend([label] * len(files))

    paths = np.asarray(paths)
    labels = np.asarray(labels, dtype=np.int32)

    real_count = int(np.sum(labels == 0))
    fake_count = int(np.sum(labels == 1))

    if real_count == 0 or fake_count == 0:
        raise RuntimeError(f"Both REAL and FAKE images are required in {split_name}.")

    print(f"{split_name.upper():>5}: REAL={real_count}, FAKE={fake_count}, TOTAL={len(paths)}")
    return paths, labels


train_paths, train_labels = collect_split("train")
val_paths, val_labels = collect_split("val")

# Test data is intentionally not loaded or evaluated by this training script.
print("The held-out test folder is not used during training or validation.")


# ---------------- IMAGE PIPELINE ----------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    # decode_image supports JPEG, PNG, BMP and GIF; expand_animations=False
    # ensures a single image tensor. The dataset is resized to 224x224.
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    # Keras EfficientNetB0 includes its own Rescaling layer.
    # Keep input pixels in the 0..255 range; do not apply preprocess_input.
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels_array, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_array))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 4000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2,
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(1)


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

model = models.Model(inputs, outputs, name="EfficientNetB0_Expanded_Fake_Image_Detector")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


def make_callbacks():
    return [
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
    callbacks=make_callbacks(),
)


# ---------------- PHASE 2: FINE-TUNING ----------------
print("\nPhase 2: Fine-tuning the final EfficientNetB0 layers...")
base_model.trainable = True

# Fine-tune only the final 30 layers; keep earlier pretrained features fixed.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stable fine-tuning.
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
    callbacks=make_callbacks(),
)

# Save final model state separately from the best-validation checkpoint.
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
plt.title("EfficientNetB0 Expanded Dataset Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(ACCURACY_PLOT, dpi=150)
plt.close()

plt.figure()
plt.plot(history.get("loss", []), label="Training loss")
plt.plot(history.get("val_loss", []), label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 Expanded Dataset Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT, dpi=150)
plt.close()

print("\nEfficientNetB0 expanded-dataset training complete!")
print(f"Best checkpoint: {BEST_MODEL_PATH}")
print(f"Final model: {FINAL_MODEL_PATH}")
print(f"Accuracy plot: {ACCURACY_PLOT}")
print(f"Loss plot: {LOSS_PLOT}")
