from dataclasses import dataclass
from enum import Enum


class TrackStatus(Enum):
    DETECTED = "detected"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"
    LOST = "lost"


@dataclass
class TrackState:
    frame_idx: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    status: TrackStatus = TrackStatus.LOST