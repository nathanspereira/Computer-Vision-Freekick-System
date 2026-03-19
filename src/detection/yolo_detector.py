from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from .detector_base import BaseDetector, Detection


class YOLODetector(BaseDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.10,
        target_class_id: int | None = 32,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self.model(frame, verbose=False)[0]
        detections: list[Detection] = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence < self.conf_threshold:
                continue

            if self.target_class_id is not None and class_id != self.target_class_id:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            center_x = x1 + w / 2
            center_y = y1 + h / 2

            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    x=center_x,
                    y=center_y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    source=f"yolo_cls_{class_id}",
                )
            )

        return detections