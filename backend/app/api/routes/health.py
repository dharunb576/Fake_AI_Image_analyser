
from fastapi import APIRouter

from app.services.predictor import predictor

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    model_loaded = predictor.model is not None

    return {
        "status": "healthy" if model_loaded else "starting",
        "api": "AI Fake Photo Detector",
        "model_loaded": model_loaded
    }
