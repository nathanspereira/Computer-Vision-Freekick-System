from __future__ import annotations

from math import sqrt

from .tracker_base import BaseTracker
from .track_state import TrackState, TrackStatus
from ..detection.detector_base import Detection


class SingleBallTracker(BaseTracker):
    def __init__(self, max_distance: float = 50.0, max_missed_frames: int = 5):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.current_state: TrackState | None = None
        self.missed_frames = 0

    def reset(self) -> None:
        self.current_state = None
        self.missed_frames = 0

    def initialize_from_detection(self, detection: Detection) -> TrackState:
        """
        Initialize the tracker directly from a chosen detection.
        """
        self.missed_frames = 0
        self.current_state = TrackState(
            frame_idx=detection.frame_idx,
            x=detection.x,
            y=detection.y,
            vx=0.0,
            vy=0.0,
            confidence=detection.confidence,
            status=TrackStatus.DETECTED,
        )
        return self.current_state

    def update(self, detections: list[Detection], frame_idx: int) -> TrackState:
        if not detections:
            return self._handle_no_detection(frame_idx)

        if self.current_state is None:
            best = max(detections, key=lambda d: d.confidence)
            self.missed_frames = 0
            self.current_state = TrackState(
                frame_idx=frame_idx,
                x=best.x,
                y=best.y,
                vx=0.0,
                vy=0.0,
                confidence=best.confidence,
                status=TrackStatus.DETECTED,
            )
            return self.current_state

        best = self._select_best_detection(detections)

        if best is None:
            return self._handle_no_detection(frame_idx)

        vx = best.x - self.current_state.x
        vy = best.y - self.current_state.y
        self.missed_frames = 0

        self.current_state = TrackState(
            frame_idx=frame_idx,
            x=best.x,
            y=best.y,
            vx=vx,
            vy=vy,
            confidence=best.confidence,
            status=TrackStatus.DETECTED,
        )
        return self.current_state

    def _select_best_detection(self, detections: list[Detection]) -> Detection | None:
        if self.current_state is None:
            return None

        best_detection = None
        best_distance = float("inf")

        for detection in detections:
            dx = detection.x - self.current_state.x
            dy = detection.y - self.current_state.y
            distance = sqrt(dx * dx + dy * dy)

            if distance <= self.max_distance and distance < best_distance:
                best_distance = distance
                best_detection = detection

        return best_detection

    def _handle_no_detection(self, frame_idx: int) -> TrackState:
        if self.current_state is None:
            return TrackState(
                frame_idx=frame_idx,
                x=0.0,
                y=0.0,
                vx=0.0,
                vy=0.0,
                confidence=0.0,
                status=TrackStatus.LOST,
            )

        self.missed_frames += 1

        if self.missed_frames > self.max_missed_frames:
            self.current_state = TrackState(
                frame_idx=frame_idx,
                x=self.current_state.x,
                y=self.current_state.y,
                vx=0.0,
                vy=0.0,
                confidence=0.0,
                status=TrackStatus.LOST,
            )
            return self.current_state

        predicted_x = self.current_state.x + self.current_state.vx
        predicted_y = self.current_state.y + self.current_state.vy

        self.current_state = TrackState(
            frame_idx=frame_idx,
            x=predicted_x,
            y=predicted_y,
            vx=self.current_state.vx,
            vy=self.current_state.vy,
            confidence=0.0,
            status=TrackStatus.PREDICTED,
        )
        return self.current_state