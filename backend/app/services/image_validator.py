
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE


def validate_image(image_bytes: bytes, filename: str) -> None:
    """
    Validate the uploaded image before model prediction.
    """

    # Check file extension
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. Upload JPG, JPEG, PNG, or WEBP."
        )

    # Check file size
    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image size must not exceed 10 MB.")

    # Check whether the file is a valid image
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        # Reopen after verify() to inspect dimensions
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")

            if width > 10000 or height > 10000:
                raise ValueError("Image dimensions are too large.")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("The uploaded file is not a valid image.")
