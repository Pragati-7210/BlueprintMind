from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# Same class IDs used by our ResNet34-U-Net.
FLOOR = 0
WALL = 1
DOOR = 2
WINDOW = 3


def parse_points(points_text: str) -> np.ndarray:
    """Convert SVG polygon points into an Nx2 float array."""
    values = []

    for token in points_text.replace(",", " ").split():
        values.append(float(token))

    if len(values) % 2 != 0:
        raise ValueError(f"Invalid polygon points: {points_text}")

    return np.array(values, dtype=np.float32).reshape(-1, 2)


def element_polygons(element):
    """Return polygon coordinate arrays contained directly in an SVG group."""
    polygons = []

    for child in element:
        tag = child.tag.split("}")[-1]

        if tag == "polygon":
            points = child.get("points")
            if points:
                polygons.append(parse_points(points))

    return polygons


def rasterize_svg(svg_path: str | Path) -> np.ndarray:
    """
    Create a 4-class ground-truth mask from a CubiCasa SVG.

    Output:
        uint8 array with:
        0 = floor
        1 = wall
        2 = door
        3 = window
    """
    svg_path = Path(svg_path)

    root = ET.parse(svg_path).getroot()

    width = int(round(float(root.get("width"))))
    height = int(round(float(root.get("height"))))

    # Start with floor.
    mask = np.full(
        (height, width),
        FLOOR,
        dtype=np.uint8,
    )

    walls = []
    doors = []
    windows = []
    spaces = []

    for element in root.iter():
        cls = element.get("class", "").strip()

        if cls == "Wall" or cls == "Wall External":
            walls.append(element)

        elif cls.startswith("Door"):
            doors.append(element)

        elif cls.startswith("Window"):
            windows.append(element)

        elif cls.startswith("Space "):
            spaces.append(element)

    def draw_groups(groups, class_id):
        for group in groups:
            for polygon in element_polygons(group):
                pts = np.round(polygon).astype(np.int32)
                cv2.fillPoly(mask, [pts], class_id)

    # Draw broad structural regions first.
    draw_groups(spaces, FLOOR)
    draw_groups(walls, WALL)

    # Openings must overwrite wall pixels.
    draw_groups(windows, WINDOW)
    draw_groups(doors, DOOR)

    return mask


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python src/evaluation/svg_ground_truth.py "
            "<model.svg>"
        )
        raise SystemExit(1)

    svg_path = Path(sys.argv[1])

    mask = rasterize_svg(svg_path)

    print("Ground-truth mask generated")
    print("Shape:", mask.shape)

    for class_id, name in [
        (FLOOR, "floor"),
        (WALL, "wall"),
        (DOOR, "door"),
        (WINDOW, "window"),
    ]:
        pixels = int(np.sum(mask == class_id))
        percentage = 100 * pixels / mask.size
        print(
            f"{name:7s}: {pixels:8d} pixels "
            f"({percentage:6.2f}%)"
        )

    output_dir = Path("outputs/ground_truth")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{svg_path.stem}_mask.npy"
    np.save(output_path, mask)

    preview_path = output_dir / f"{svg_path.stem}_mask.png"
    cv2.imwrite(str(preview_path), mask * 85)

    print("\nSaved:")
    print(output_path)
    print(preview_path)


if __name__ == "__main__":
    main()
