from abc import ABC, abstractmethod
from ..detection.detector_base import Detection
from .track_state import TrackState


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame_idx: int) -> TrackState:
        """
        Update the tracker using the current frame's detections.

        Args:
            detections: Candidate detections for the current frame.
            frame_idx: Current frame index.

        Returns:
            The best estimated tracked state for this frame.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset tracker state before starting a new sequence.
        """
        pass