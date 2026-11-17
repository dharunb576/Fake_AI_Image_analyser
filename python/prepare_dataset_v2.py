
"""
Prepare balanced REAL and FAKE image datasets for AI Fake Photo Detector V2.

Source:
Shanmuk4622/ai-image-detection-dataset

REAL: 10,000 images
FAKE: 10,000 images (SDXL)

Original train/validation/test splits are preserved.
Source images remain at 512x512.
Resizing to 224x224 happens during model training.
"""

from pathlib import Path
from datasets import load_dataset
from PIL import Image

REPO = "Shanmuk4622/ai-image-detection-dataset"

OUTPUT_DIR = Path("data_v2")

SUBSETS = {
    "real": "REAL",
    "sdxl": "FAKE",
}

SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def prepare_dataset():

    # Prevent accidental mixing with old files
    existing = list(OUTPUT_DIR.rglob("*.png"))

    if existing:
        raise RuntimeError(
            f"Found {len(existing)} existing PNG files in data_v2. "
            "Move or clean the old dataset before running again."
        )

    total_images = 0

    for subset, label in SUBSETS.items():

        print(f"\nLoading {subset.upper()} dataset...")

        dataset = load_dataset(REPO, name=subset)

        for source_split, target_split in SPLITS.items():

            if source_split not in dataset:
                raise ValueError(
                    f"Missing split: {source_split} in {subset}"
                )

            rows = dataset[source_split]

            output_folder = (
                OUTPUT_DIR / target_split / label
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            count = 0

            for index, row in enumerate(rows):

                image = row["image"]

                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"Unexpected image format: {type(image)}"
                    )

                image = image.convert("RGB")

                image_id = row.get(
                    "image_id",
                    f"{subset}_{index:06d}"
                )

                filename = f"{image_id}.png"

                image.save(
                    output_folder / filename,
                    format="PNG",
                    optimize=True
                )

                count += 1
                total_images += 1

                if count % 500 == 0:
                    print(
                        f"{subset} / {source_split}: "
                        f"{count} images saved"
                    )

            print(
                f"Completed {label} / {target_split}: "
                f"{count} images"
            )

    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

    print("\nExpected total: 20,000 images")

    if total_images != 20000:
        print(
            "WARNING: The total differs from 20,000. "
            "Check the downloaded dataset splits."
        )


if __name__ == "__main__":
    prepare_dataset()
