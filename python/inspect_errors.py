
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
errors = pd.read_csv(ROOT / "results/mobilenetv2_errors.csv")

# Show up to 40 incorrectly classified images.
errors = errors.head(40)

fig, axes = plt.subplots(5, 8, figsize=(20, 14))
axes = axes.flatten()

for ax, (_, row) in zip(axes, errors.iterrows()):
    image_path = ROOT / "data" / (
        "real" if row["actual"] == "REAL" else "fake"
    ) / row["image"]

    if image_path.exists():
        ax.imshow(Image.open(image_path).convert("RGB"))
        ax.set_title(
            f'{row["actual"]} → {row["predicted"]}\n'
            f'Fake p={row["fake_probability"]:.2f}',
            fontsize=8
        )

    ax.axis("off")

for ax in axes[len(errors):]:
    ax.axis("off")

plt.tight_layout()
output = ROOT / "results/mobilenetv2_error_contact_sheet.png"
plt.savefig(output, dpi=150)
plt.close()

print(f"Saved contact sheet: {output}")

