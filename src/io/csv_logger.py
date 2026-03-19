import csv
from pathlib import Path

from ..tracking.track_state import TrackState


# separates output logic from the runner
# creates directories automatically
# writes clean schema

def save_track_states(track_states: list[TrackState], output_path: str) -> None:
    """
    Save tracked ball states to a CSV file.

    Args:
        track_states: List of TrackState objects from the tracker.
        output_path: Path to the CSV output file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "frame_idx",
            "x",
            "y",
            "vx",
            "vy",
            "confidence",
            "status",
        ])

        for state in track_states:
            writer.writerow([
                state.frame_idx,
                state.x,
                state.y,
                state.vx,
                state.vy,
                state.confidence,
                state.status.value,
            ])