
from fastapi import APIRouter

from app.core.config import (
    MODEL_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PREDICTION_THRESHOLD,
    ALLOWED_EXTENSIONS,
)

router = APIRouter(tags=["Model Information"])


@router.get("/model-info")
def get_model_info():
    return {
        "model_name": MODEL_NAME,
        "input_size": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "input_format": "RGB",
        "classes": ["REAL", "FAKE"],
        "prediction_threshold": PREDICTION_THRESHOLD,
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "maximum_upload_size_mb": 10
    }
