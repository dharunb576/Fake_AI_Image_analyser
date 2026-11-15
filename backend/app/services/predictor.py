
import tensorflow as tf

from app.core.config import (
    MODEL_PATH,
    MODEL_NAME,
    PREDICTION_THRESHOLD,
)


class ImagePredictor:

    def __init__(self):
        self.model = None

    def load_model(self):
        """Load the trained EfficientNetB0 model."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        print(f"{MODEL_NAME} model loaded successfully!")

    def predict(self, image_tensor):
        """Predict whether an image is REAL or FAKE."""

        if self.model is None:
            raise RuntimeError("Model has not been loaded.")

        # Get fake probability from sigmoid output
        fake_probability = float(
            self.model.predict(image_tensor, verbose=0)[0][0]
        )

        # Calculate real probability
        real_probability = 1.0 - fake_probability

        # Classification
        if fake_probability >= PREDICTION_THRESHOLD:
            prediction = "FAKE"
            confidence = fake_probability
        else:
            prediction = "REAL"
            confidence = real_probability

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "real_probability": round(real_probability * 100, 2),
            "fake_probability": round(fake_probability * 100, 2),
            "model": MODEL_NAME,
        }


# Create one shared predictor instance
predictor = ImagePredictor()
