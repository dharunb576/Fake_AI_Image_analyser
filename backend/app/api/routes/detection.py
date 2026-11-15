
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import MAX_IMAGE_SIZE
from app.services.image_validator import validate_image
from app.services.preprocessing import preprocess_image
from app.services.predictor import predictor

router = APIRouter(prefix="/api", tags=["Image Detection"])


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Upload an image and predict whether it is REAL or FAKE."""

    start_time = time.perf_counter()

    try:
        # Read at most 10 MB + 1 byte to enforce the upload limit
        image_bytes = await file.read(MAX_IMAGE_SIZE + 1)

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image size must not exceed 10 MB."
            )

        # Validate the uploaded image
        validate_image(image_bytes, file.filename or "")

        # Convert image into model input format
        image_tensor = preprocess_image(image_bytes)

        # Run prediction
        result = predictor.predict(image_tensor)

        processing_time = time.perf_counter() - start_time

        return {
            "success": True,
            "filename": file.filename,
            **result,
            "processing_time": round(processing_time, 3)
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Unable to process this image. Please try another image."
        )

    finally:
        await file.close()
