from __future__ import annotations

import cv2
import numpy as np

from .initializer_base import BaseInitializer
from ..detection.detector_base import Detection


class BlobInitializer(BaseInitializer):
    """
    Initialize the stationary pre-kick ball using contour/blob cues.

    Strategy:
    - convert frame to grayscale
    - blur to reduce noise
    - threshold bright regions
    - find contours
    - filter by area, aspect ratio, and circularity
    - return the best candidate
    """

    def __init__(
        self,
        threshold_value: int = 200,
        min_area: float = 50.0,
        max_area: float = 5000.0,
        min_circularity: float = 0.55,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
    ):
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def initialize(self, frame: np.ndarray, frame_idx: int) -> Detection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best_detection: Detection | None = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue

            aspect_ratio = w / h
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            # Simple score: prefer more circular blobs
            score = circularity

            if score > best_score:
                best_score = score
                best_detection = Detection(
                    frame_idx=frame_idx,
                    x=center_x,
                    y=center_y,
                    w=w,
                    h=h,
                    confidence=float(score),
                    source="blob_initializer",
                )

        return best_detection