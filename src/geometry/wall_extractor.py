import sys
from pathlib import Path

import cv2
import numpy as np


def extract_wall_mask(mask):
    """
    Extract the wall class from the semantic segmentation mask.

    Class mapping:
        0 = floor
        1 = wall
        2 = door
        3 = window
    """

    wall_mask = (mask == 1).astype(np.uint8) * 255

    return wall_mask


def clean_wall_mask(wall_mask):
    """
    Remove small gaps and connect nearby wall regions.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    cleaned = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return cleaned


def extract_wall_lines(wall_mask):
    """
    Extract long wall segments using probabilistic Hough transform.
    """

    lines = cv2.HoughLinesP(
        wall_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=12,
    )

    segments = []

    if lines is None:
        return segments

    for line in lines:
        x1, y1, x2, y2 = map(int, line)

        length = np.hypot(
            x2 - x1,
            y2 - y1,
        )

        angle = np.degrees(
            np.arctan2(
                y2 - y1,
                x2 - x1,
            )
        )

        # Normalize angle to [0, 180)
        angle = angle % 180

        segments.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length": float(length),
                "angle": float(angle),
            }
        )

    # Longest walls first
    segments.sort(
        key=lambda x: x["length"],
        reverse=True,
    )

    return segments


def draw_wall_lines(shape, segments):
    """
    Draw extracted wall segments for visualization.
    """

    visualization = np.zeros(
        (*shape, 3),
        dtype=np.uint8,
    )

    for segment in segments:
        x1 = segment["x1"]
        y1 = segment["y1"]
        x2 = segment["x2"]
        y2 = segment["y2"]

        cv2.line(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

    return visualization


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python src/geometry/wall_extractor.py "
            "<mask.npy>"
        )
        sys.exit(1)

    mask_path = Path(sys.argv[1])

    mask = np.load(mask_path)

    print("Mask shape:", mask.shape)

    # 1. Extract walls
    wall_mask = extract_wall_mask(mask)

    # 2. Clean segmentation
    cleaned = clean_wall_mask(wall_mask)

    # 3. Extract geometric wall segments
    segments = extract_wall_lines(cleaned)

    print(f"Wall segments detected: {len(segments)}")

    print("\nLongest wall segments:")

    for i, segment in enumerate(segments[:20]):
        print(
            f"{i + 1:2d}. "
            f"({segment['x1']}, {segment['y1']}) → "
            f"({segment['x2']}, {segment['y2']}) "
            f"length={segment['length']:.1f} "
            f"angle={segment['angle']:.1f}°"
        )

    # Save intermediate wall mask
    cv2.imwrite(
        "outputs/segmentation/1245_wall_mask.png",
        cleaned,
    )

    # Save line visualization
    visualization = draw_wall_lines(
        mask.shape,
        segments,
    )

    cv2.imwrite(
        "outputs/segmentation/1245_wall_lines.png",
        visualization,
    )

    print(
        "\nSaved:"
        "\n  outputs/segmentation/1245_wall_mask.png"
        "\n  outputs/segmentation/1245_wall_lines.png"
    )


if __name__ == "__main__":
    main()
