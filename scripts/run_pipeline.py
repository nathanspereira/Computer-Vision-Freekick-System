import cv2
from pathlib import Path

from src.detection.classical_detector import ClassicalDetector
from src.tracking.single_ball_tracker import SingleBallTracker
from src.io.csv_logger import save_track_states
from src.visualization.overlay import draw_track_state
from src.initialization.blob_initializer import BlobInitializer


def run_video_pipeline(
    video_path: str,
    output_csv_path: str,
    output_video_path: str,
) -> None:
    detector = ClassicalDetector(
        threshold_value=200,
        min_area=20.0,
        max_area=2000.0,
        exclude_region=(1550, 900, 1825, 1080),
    )

    initializer = BlobInitializer(
        threshold_value=200,
        min_area=50.0,
        max_area=5000.0,
        min_circularity=0.55,
    )

    tracker = SingleBallTracker(max_distance=50.0)
    max_init_frames = 40

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path(output_video_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for: {output_video_path}")

    track_states = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if tracker.current_state is None and frame_idx < max_init_frames:
            init_detection = initializer.initialize(frame, frame_idx)

            if init_detection is not None:
                state = tracker.initialize_from_detection(init_detection)
            else:
                detections = detector.detect(frame, frame_idx)
                state = tracker.update(detections, frame_idx)
        else:
            detections = detector.detect(frame, frame_idx)
            state = tracker.update(detections, frame_idx)

        track_states.append(state)

        annotated_frame = draw_track_state(frame, state)
        writer.write(annotated_frame)

        frame_idx += 1

    cap.release()
    writer.release()

    save_track_states(track_states, output_csv_path)

    print(f"Saved CSV to: {output_csv_path}")
    print(f"Saved video to: {output_video_path}")


if __name__ == "__main__":
    video_path = "data/raw/input.mov"
    output_csv_path = "artifacts/logs/track_states.csv"
    output_video_path = "artifacts/overlays/tracked_output.mp4"

    run_video_pipeline(video_path, output_csv_path, output_video_path)