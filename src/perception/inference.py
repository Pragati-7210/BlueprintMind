import sys
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import build_model


CLASS_NAMES = ["floor", "wall", "door", "window"]

# Visualization colors: BGR for OpenCV
COLORS = {
    0: (240, 240, 235),  # floor
    1: (40, 40, 45),     # wall
    2: (50, 140, 230),   # door
    3: (220, 150, 60),   # window
}


def load_svg_as_image(svg_path: str) -> Image.Image:
    """Render SVG into a PIL RGB image."""
    png_bytes = cairosvg.svg2png(
        url=svg_path,
        output_width=None,
        output_height=None,
    )

    from io import BytesIO

    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    return image


def letterbox(image: Image.Image, size=512):
    """
    Resize while preserving aspect ratio and pad to size x size.

    Returns:
        canvas: letterboxed image
        metadata: transformation information needed to map
                  model coordinates back to original coordinates.
    """
    width, height = image.size

    scale = min(size / width, size / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    canvas.paste(resized, (left, top))

    metadata = {
        "original_width": width,
        "original_height": height,
        "scale": scale,
        "new_width": new_width,
        "new_height": new_height,
        "left": left,
        "top": top,
        "model_size": size,
    }

    return canvas, metadata

def preprocess(svg_path: str):
    image = load_svg_as_image(svg_path)

    original_size = image.size

    image, metadata = letterbox(image, 512)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    tensor = transform(image).unsqueeze(0)

    return image, tensor, original_size, metadata


def create_segmentation_visualization(mask):
    """
    Convert class mask into a colored visualization.
    """
    height, width = mask.shape

    visualization = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    for class_id, color in COLORS.items():
        visualization[mask == class_id] = color

    return visualization

def model_to_original(x, y, metadata):
    """
    Convert a coordinate from the 512x512 model image
    back to the original SVG coordinate system.
    """

    left = metadata["left"]
    top = metadata["top"]
    scale = metadata["scale"]

    original_x = (x - left) / scale
    original_y = (y - top) / scale

    return original_x, original_y

def restore_mask_to_original(mask, metadata):
    """
    Convert the 512x512 model mask back to the original
    blueprint resolution.
    """

    left = metadata["left"]
    top = metadata["top"]

    new_width = metadata["new_width"]
    new_height = metadata["new_height"]

    original_width = metadata["original_width"]
    original_height = metadata["original_height"]

    # Remove letterbox padding
    content_mask = mask[
        top:top + new_height,
        left:left + new_width,
    ]

    # Resize class-ID mask back to original resolution.
    # NEAREST is essential: class IDs must not be interpolated.
    restored = cv2.resize(
        content_mask.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return restored

def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python src/perception/inference.py "
            "<model.svg> <output.png>"
        )
        sys.exit(1)

    svg_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Input: {svg_path}")

    # Load model
    model = build_model("weights/best.safetensors")

    # Preprocess
    image, tensor, original_size, metadata = preprocess(svg_path)

    print(f"Original image size: {original_size}")
    print("Letterbox metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print(f"Model input shape: {tensor.shape}")

    # Inference
    with torch.no_grad():
        output = model(tensor)

    print(f"Model output shape: {output.shape}")

    mask = torch.argmax(output, dim=1)[0].cpu().numpy()

    # Save model-space mask
    mask_path = output_path.replace(".png", "_mask.npy")
    np.save(mask_path, mask)

    print(f"Saved raw mask: {mask_path}")

    # Restore prediction to original blueprint coordinates
    original_mask = restore_mask_to_original(
        mask,
        metadata,
    )

    original_mask_path = output_path.replace(
        ".png",
        "_mask_original.npy",
    )

    np.save(
        original_mask_path,
        original_mask,
    )

    print(
        f"Saved original-resolution mask: "
        f"{original_mask_path}"
    )

    print(
        f"Original mask shape: "
        f"{original_mask.shape}"
    )
    # Count pixels belonging to each class
    print("\nPredicted classes:")

    total_pixels = mask.size

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = np.sum(mask == class_id)
        percentage = 100 * count / total_pixels

        print(
            f"  {class_name:6s}: "
            f"{count:7d} pixels "
            f"({percentage:6.2f}%)"
        )

    # Visualization
    visualization = create_segmentation_visualization(mask)

    cv2.imwrite(output_path, visualization)

    print(f"\nSaved segmentation: {output_path}")


if __name__ == "__main__":
    main()
