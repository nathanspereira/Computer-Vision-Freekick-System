import cv2
import numpy as np

from ..tracking.track_state import TrackState, TrackStatus


def draw_track_state(frame: np.ndarray, state: TrackState) -> np.ndarray:
    """
    Draw the current tracked state on a frame.

    Args:
        frame: Video frame to annotate.
        state: Current tracked ball state.

    Returns:
        Annotated frame.
    """
    output = frame.copy()

    x = int(state.x)
    y = int(state.y)

    if state.status == TrackStatus.DETECTED:
        cv2.circle(output, (x, y), 8, (0, 255, 0), -1)
        label = "DETECTED"

    elif state.status == TrackStatus.PREDICTED:
        cv2.circle(output, (x, y), 8, (0, 255, 255), 2)
        label = "PREDICTED"

    else:
        label = "LOST"

    cv2.putText(
        output,
        label,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output