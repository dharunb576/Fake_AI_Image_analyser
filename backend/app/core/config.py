from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Model configuration
MODEL_NAME = "EfficientNetB0v2"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_v2" /"continued_best_model.keras"

# Image configuration
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3

# Prediction configuration
PREDICTION_THRESHOLD = 0.5

# Upload validation
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# API configuration
API_TITLE = "AI Fake Photo Detector API"
API_VERSION = "1.0.0"
