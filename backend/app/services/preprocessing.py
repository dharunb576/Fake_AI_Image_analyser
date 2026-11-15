
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from app.core.config import IMAGE_SIZE


def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """
    Convert uploaded image bytes into the input format
    expected by the trained EfficientNetB0 model.
    """

    # Open uploaded image
    with Image.open(BytesIO(image_bytes)) as image:

        # Correct image orientation using EXIF metadata
        image = ImageOps.exif_transpose(image)

        # Convert image to RGB
        image = image.convert("RGB")

        # Convert image into a NumPy array
        image_array = np.asarray(image, dtype=np.uint8)

    # Convert NumPy array into TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_array)

    # Resize to 224 x 224 using TensorFlow
    image_tensor = tf.image.resize(
        image_tensor,
        IMAGE_SIZE
    )

    # Keep pixel values in the 0-255 range
    image_tensor = tf.cast(image_tensor, tf.float32)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    return image_tensor
