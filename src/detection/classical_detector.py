from __future__ import annotations

import cv2
import numpy as np

from .detector_base import BaseDetector, Detection


class ClassicalDetector(BaseDetector):
    """
    Baseline contour-based detector for generating ball candidates
    from thresholded grayscale frames.
    """

    def __init__(
        self,
        threshold_value: int = 200,
        min_area: float = 20.0,
        max_area: float = 2000.0,
        exclude_region: tuple[int, int, int, int] | None = None,
    ):
        self.threshold_value = threshold_value
        self.min_area = min_area
        self.max_area = max_area
        self.exclude_region = exclude_region

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(
            blurred,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY,
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: list[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            if self.exclude_region is not None:
                x_min, y_min, x_max, y_max = self.exclude_region
                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                    continue

            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    x=center_x,
                    y=center_y,
                    w=w,
                    h=h,
                    confidence=1.0,
                    source="classical",
                )
            )

        return detections