from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from ..detection.detector_base import Detection


class BaseInitializer(ABC):
    @abstractmethod
    def initialize(self, frame: np.ndarray, frame_idx: int) -> Detection | None:
        """
        Attempt to find the launch ball in a frame.

        Args:
            frame: Current video frame.
            frame_idx: Frame index.

        Returns:
            A Detection for the initialized ball, or None if no plausible
            initialization candidate is found.
        """
        pass