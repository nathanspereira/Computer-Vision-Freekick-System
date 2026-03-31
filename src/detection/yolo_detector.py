from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO
from .detector_base import BaseDetector


@dataclass
class BallTrackCandidate:
    frame_idx: int
    track_id: int | None
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int


class YOLODetector(BaseDetector):

    def __init__(
        self,
        model_path: str = "yolo26m.pt",
        confidence_threshold: float = 0.08,
        target_class_id: int = 32,
        tracker_path: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_class_id = target_class_id
        self.tracker_path = tracker_path

    # detets ball
    # runs yolo object detection with bytetrack algorithm
    def detect(self, frame, frame_idx: int) -> list[BallTrackCandidate]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_path,
            conf=self.confidence_threshold,
            classes=[self.target_class_id],
            imgsz=1280,
            verbose=False,
        )

        return self._results_to_candidates(
            results = results,
            frame_idx= frame_idx,
            x_offset = 0,
            y_offset = 0,
            force_track_id_none = False,
        )

    #crops region of interest around predicted ball location
    #runs plain yolo object detection within the crop
    # uses predict() instead of track() to not mess with ByteTrack's log
    # background_ref supreses static background before yolo
    def detect_in_roi(
        self,
        frame,
        frame_idx: int,
        center_x: float,
        center_y: float,
        roi_half_size: int = 128,
        roi_confidence_threshold: float = 0.03,
        background_ref = None,
    ) -> list[BallTrackCandidate]:

        h, w = frame.shape[:2]

        x1 = max(0, int(center_x - roi_half_size))
        y1 = max(0, int(center_y - roi_half_size))
        x2 = min(w, int(center_x + roi_half_size))
        y2 = min(h, int(center_y + roi_half_size))

        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        if background_ref is not None:
            bh, bw = background_ref.shape[:2]
            bx1 = max(0, x1)
            by1 = max(0, y1)
            bx2 = min(bw, x2)
            by2 = min(bh, y2)

            if bx2 > bx1 and by2 > by1:
                bg_crop = background_ref[by1:by2, bx1:bx2]
                frame_crop = frame[by1:by2, bx1:bx2]

                if bg_crop.shape == frame_crop.shape:
                    diff = cv2.absdiff(frame_crop, bg_crop)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                    _, fg_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    fg_mask = cv2.dilate(fg_mask, kernel)

                    bg_mask = cv2.bitwise_not(fg_mask)
                    suppressed = frame_crop.copy().astype(np.float32)
                    suppressed[bg_mask > 0] *= 0.15
                    crop = np.clip(suppressed, 0, 255).astype(np.uint8)

        results = self.model.predict(
            crop,
            conf=roi_confidence_threshold,
            classes=[self.target_class_id],
            imgsz=640,
            verbose=False,
        )

        return self._results_to_candidates(
            results=results,
            frame_idx=frame_idx,
            x_offset=x1,
            y_offset=y1,
            force_track_id_none=True,
        )

    # 
    def _results_to_candidates(
        self,
        results,
        frame_idx: int,
        x_offset: int,
        y_offset: int,
        force_track_id_none: bool,
    ) -> list[BallTrackCandidate]:
        candidates: list[BallTrackCandidate] = []

        if not results:
            return candidates

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return candidates

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())

            if class_id != self.target_class_id:
                continue

            track_id = None
            if not force_track_id_none and boxes.id is not None:
                track_id = int(boxes.id[i].item())

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            w = float(x2 - x1)
            h = float(y2 - y1)
            x = float((x1 + x2) / 2.0) + x_offset
            y = float((y1 + y2) / 2.0) + y_offset

            candidates.append(
                BallTrackCandidate(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    class_id=class_id,
                )
            )

        return candidates