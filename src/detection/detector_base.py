from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class Detection:
    frame_idx: int
    x: float
    y: float
    w: float
    h: float
    confidence: float
    source: str = "unknown"


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect ball candidates in a single frame.

        Args:
            frame: The current video frame as a NumPy array.
            frame_idx: The index of the current frame.

        Returns:
            A list of Detection objects representing candidate ball locations.
        """
        pass