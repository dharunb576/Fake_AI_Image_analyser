
"""
Resume EfficientNetB0 V2 training
AI Fake Photo Detector

Loads the existing fine-tuned checkpoint and continues training.
Original checkpoint and backup remain untouched.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    BackupAndRestore
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
ADDITIONAL_EPOCHS = 5
SEED = 42

DATA_DIR = Path("data_v2")

ORIGINAL_MODEL = Path(
    "models/efficientnetb0_v2/best_model.keras"
)

MODEL_DIR = Path("models/efficientnetb0_v2")
BACKUP_DIR = Path(
    "backups/efficientnetb0_v2/resume_training_state"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# CHECKPOINT
# --------------------------------------------------

if not ORIGINAL_MODEL.exists():
    raise FileNotFoundError(
        f"Original model not found: {ORIGINAL_MODEL}"
    )

print("\nLoading existing EfficientNetB0 checkpoint...")

model = tf.keras.models.load_model(
    ORIGINAL_MODEL,
    compile=True
)

print("Model loaded successfully.")
print("Input:", model.input_shape)
print("Output:", model.output_shape)
print("Optimizer:", type(model.optimizer).__name__)
print(
    "Optimizer iterations:",
    int(model.optimizer.iterations.numpy())
)
print(
    "Learning rate:",
    float(
        tf.keras.backend.get_value(
            model.optimizer.learning_rate
        )
    )
)

# Verify the fine-tuned model configuration.
base_model = model.get_layer("efficientnetb0")

trainable_layers = sum(
    1 for layer in base_model.layers if layer.trainable
)

print("Trainable EfficientNet layers:", trainable_layers)

if trainable_layers != 24:
    raise RuntimeError(
        "Unexpected EfficientNet trainable-layer configuration. "
        "Stop and inspect the loaded model before training."
    )

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

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


print("\nLoading datasets...")

train_ds = load_split("train", True)
val_ds = load_split("val", False)

assert train_ds.class_names == ["REAL", "FAKE"]

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Training and validation datasets ready.")

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

continued_checkpoint = (
    MODEL_DIR / "continued_best_model.keras"
)

callbacks = [

    BackupAndRestore(
        backup_dir=str(BACKUP_DIR),
        save_freq="epoch"
    ),

    ModelCheckpoint(
        filepath=continued_checkpoint,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        min_lr=1e-7,
        verbose=1
    )
]

# --------------------------------------------------
# CONTINUE TRAINING
# --------------------------------------------------

print("\n========================================")
print("RESUMING EXISTING EFFICIENTNETB0 MODEL")
print("========================================")

print("Additional epochs:", ADDITIONAL_EPOCHS)
print("Original checkpoint:", ORIGINAL_MODEL)
print("New checkpoint:", continued_checkpoint)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=ADDITIONAL_EPOCHS,
    callbacks=callbacks
)

# --------------------------------------------------
# SAVE FINAL CONTINUED MODEL
# --------------------------------------------------

print("\nLoading best continued checkpoint...")

best_model = tf.keras.models.load_model(
    continued_checkpoint
)

final_path = MODEL_DIR / "continued_final_model.keras"

best_model.save(final_path)

print("\n========================================")
print("CONTINUED TRAINING COMPLETED")
print("========================================")

print("Best continued model:", continued_checkpoint)
print("Final model:", final_path)
