from __future__ import annotations
from math import hypot
from pathlib import Path
from src.tracking.goal_region_model import GoalRegionModel
from collections import deque
import numpy as np
import cv2

from src.detection.yolo_detector import YOLODetector, BallTrackCandidate
"""
Driver file.
After working pipeline, make this into separate files for abstraction and modularity
"""

def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return hypot(x1 - x2, y1 - y2)


#constant velocity predicator (FIX THIS WITH PHYSICS BASED LATER IF NEEDED)
#uses last 2 trusted points (COULD BE SOURCE OF OVERPREDICTION BUG)
def predict_next_center(history: deque[tuple[float, float]]) -> tuple[float, float] | None:
    if len(history) == 0:
        return None

    if len(history) == 1:
        return history[-1]

    x_prev, y_prev = history[-2]
    x_last, y_last = history[-1]

    vx = x_last - x_prev
    vy = y_last - y_prev

    return (x_last + vx, y_last + vy)

# accepts only roi detections which are:
#1) above min confidence, 
#2) within acceptable distance to predicted point, 
#3) and close enough to last trusted point
#after, uses nearest to prediction first, nearest to last second, and higher confidence last

def choose_best_roi_candidate(
    candidates: list[BallTrackCandidate],
    pred_x: float,
    pred_y: float,
    last_x: float | None,
    last_y: float | None,
    max_dist_from_prediction: float,
    max_step_from_last_locked: float,
    min_accept_conf: float,
) -> BallTrackCandidate | None:

    viable = []

    for c in candidates:
        if c.confidence < min_accept_conf:
            continue

        d_pred = distance_xy(c.x, c.y, pred_x, pred_y)
        if d_pred > max_dist_from_prediction:
            continue

        if last_x is not None and last_y is not None:
            d_last = distance_xy(c.x, c.y, last_x, last_y)
            if d_last > max_step_from_last_locked:
                continue
        else:
            d_last = 0.0

        viable.append((d_pred, d_last, -c.confidence, c))

    if not viable:
        return None

    viable.sort(key=lambda item: (item[0], item[1], item[2]))
    return viable[0][3]


def run_video_pipeline(
    video_path: str,
    output_csv_path: str,
    output_video_path: str,
) -> None:
    video_path = Path(video_path)
    output_csv_path = Path(output_csv_path)
    output_video_path = Path(output_video_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video: {video_path}")

    DISPLAY_MIN_CONF = 0.60 #min confidence for yellow boxes, reduces noise in debugging
    MIN_BOX_W = 18
    MIN_BOX_H = 18

    LOCK_MIN_CONF = 0.70
    HISTORY_LEN = 8

    # ROI rescue tuning:
    # keep confidence permissive, but tighten by geometry
    ROI_CONF_TRIGGER = 0.45 #was 0.55
    ROI_HALF_SIZE = 140
    ROI_MAX_DISTANCE = 65.0
    ROI_MAX_STEP_FROM_LAST_LOCKED = 50.0
    ROI_MIN_ACCEPT_CONF = 0.05
    ROI_MAX_CONSECUTIVE_FRAMES = 15 #was 8, gives ROI more room to brdige the goal sequence

    HOLD_LAST_BOX_FRAMES = 5 #changed from 2, 5 works better than 2 and 8

    # trajectory drawing/plotting
    TRAJECTORY_MAX_LEN = 400
    TRAJECTORY_DOT_RADIUS = 3
    TRAJECTORY_LINE_THICKNESS = 2

    #detection delcaration
    detector = YOLODetector(
        model_path="yolo26m.pt",
        confidence_threshold=0.01,
        target_class_id=32,
        tracker_path="src/tracking/bytetrack_fast_ball.yaml",
    )

    #goal model declaration
    # BUG: way overshoots bounds for goal. 
    goal_model = GoalRegionModel(
        manual_goal_rect=None,
        stable_threshold=2.5,
        stable_required_frames = 12,
        background_frames_needed =20,
        goal_margin=25,
        diff_threshold = 25,
        suppress_factor=0.15,
        debug=True,
    )

    #metadeta
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height),
    )

    #prompts could not open video
    if not writer.isOpened():
        raise RuntimeError(f"Couldn't open video writer: {output_video_path}")


    frame_idx = 0

    locked_track_id: int | None = None
    locked_history: deque[tuple[float, float]] = deque(maxlen=HISTORY_LEN)
    trajectory_points: deque[tuple[int, int]] = deque(maxlen=TRAJECTORY_MAX_LEN)

    last_locked_w = 24.0
    last_locked_h = 24.0
    misses_since_locked = 0

    #loop that reads frame by frame
    while True:
        valid, frame = cap.read()
        if not valid:
            break
        goal_model.update(frame)

        candidates = detector.detect(frame, frame_idx)

        print(f"Frame {frame_idx}: {len(candidates)} candidates")

        # draw all strong detections in yellow for debugging
        for c in candidates:
            if c.confidence < DISPLAY_MIN_CONF:
                continue
            if c.w < MIN_BOX_W or c.h < MIN_BOX_H:
                continue

            print(
                f" id = {c.track_id}, x={c.x:.1f}, y={c.y:.1f}, "
                f"w={c.w:.1f}, h={c.h:.1f}, conf={c.confidence:.2f}"
            )

            x1 = int(c.x - c.w / 2)
            y1 = int(c.y - c.h / 2)
            x2 = int(c.x + c.w / 2)
            y2 = int(c.y + c.h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"id = {c.track_id} conf = {c.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
            cv2.circle(frame, (int(c.x), int(c.y)), 4, (0, 0, 255), -1)
        
        goal_model.draw_debug(frame)

        locked_candidate: BallTrackCandidate | None = None
        predicted_center = predict_next_center(locked_history)

        last_locked_x = locked_history[-1][0] if len(locked_history) > 0 else None
        last_locked_y = locked_history[-1][1] if len(locked_history) > 0 else None

        # case 1: if already locked, prefer same ByteTrack ID
        if locked_track_id is not None:
            same_id = [c for c in candidates if c.track_id == locked_track_id]
            if same_id:
                locked_candidate = max(same_id, key=lambda c: c.confidence)

        # case 2: if not yet locked, choose strongest good candidate
        if locked_candidate is None and locked_track_id is None:
            strong = [
                c for c in candidates
                if c.confidence >= LOCK_MIN_CONF and c.w >= MIN_BOX_W and c.h >= MIN_BOX_H
            ]
            if strong:
                locked_candidate = max(strong, key=lambda c: c.confidence)
                locked_track_id = locked_candidate.track_id

        # case 3: ROI fallback when locked target disappears OR gets weak
        used_roi_fallback = False
        if (
            locked_track_id is not None
            and predicted_center is not None
            and misses_since_locked <= ROI_MAX_CONSECUTIVE_FRAMES
        ):
            should_try_roi = (
                locked_candidate is None
                or locked_candidate.confidence < ROI_CONF_TRIGGER
            )

            if should_try_roi:
                pred_x, pred_y = predicted_center

                in_goal_zone = goal_model.in_goal_zone(pred_x, pred_y)
                background_ref = (
                    goal_model.get_background_reference()
                    if (in_goal_zone and goal_model.is_ready())
                    else None
                )

                roi_candidates = detector.detect_in_roi(
                    frame=frame,
                    frame_idx=frame_idx,
                    center_x=pred_x,
                    center_y=pred_y,
                    roi_half_size=ROI_HALF_SIZE,
                    roi_confidence_threshold=0.03,
                    background_ref=background_ref,
                )

                best_roi = choose_best_roi_candidate(
                    candidates=roi_candidates,
                    pred_x=pred_x,
                    pred_y=pred_y,
                    last_x=last_locked_x,
                    last_y=last_locked_y,
                    max_dist_from_prediction=ROI_MAX_DISTANCE,
                    max_step_from_last_locked=ROI_MAX_STEP_FROM_LAST_LOCKED,
                    min_accept_conf=ROI_MIN_ACCEPT_CONF,
                )

                if best_roi is not None:
                    if locked_candidate is None or best_roi.confidence > locked_candidate.confidence:
                        locked_candidate = BallTrackCandidate(
                            frame_idx=best_roi.frame_idx,
                            track_id=locked_track_id,
                            x=best_roi.x,
                            y=best_roi.y,
                            w=best_roi.w,
                            h=best_roi.h,
                            confidence=best_roi.confidence,
                            class_id=best_roi.class_id,
                        )
                        used_roi_fallback = True

        held_prediction = None
        if locked_candidate is not None:
            # trusted track or trusted ROI rescue
            locked_history.append((locked_candidate.x, locked_candidate.y))
            trajectory_points.append((int(locked_candidate.x), int(locked_candidate.y)))
            last_locked_w = locked_candidate.w
            last_locked_h = locked_candidate.h
            misses_since_locked = 0
        else:
            misses_since_locked += 1
            if predicted_center is not None and misses_since_locked <= HOLD_LAST_BOX_FRAMES:
                held_prediction = predicted_center

        # draw connected dot plot of successful locked pings only

        if len(trajectory_points) >= 2:
            pts = np.array(trajectory_points, dtype=np.int32)
            cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=(0, 0, 255),
            thickness=TRAJECTORY_LINE_THICKNESS,
            )

        for px, py in trajectory_points:
            cv2.circle(frame, (px, py), TRAJECTORY_DOT_RADIUS, (0, 0, 255), -1)

        if locked_candidate is not None:
            x1 = int(locked_candidate.x - locked_candidate.w / 2)
            y1 = int(locked_candidate.y - locked_candidate.h / 2)
            x2 = int(locked_candidate.x + locked_candidate.w / 2)
            y2 = int(locked_candidate.y + locked_candidate.h / 2)

            color = (0, 255, 0) if not used_roi_fallback else (255, 0, 255)
            mode = "TRACK" if not used_roi_fallback else "ROI"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"LOCKED id={locked_candidate.track_id} {mode} conf={locked_candidate.confidence:.2f}",
                (x1, max(20, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            cv2.circle(frame, (int(locked_candidate.x), int(locked_candidate.y)), 5, color, -1)

        elif held_prediction is not None and locked_track_id is not None:
            pred_x, pred_y = held_prediction

            x1 = int(pred_x - last_locked_w / 2)
            y1 = int(pred_y - last_locked_h / 2)
            x2 = int(pred_x + last_locked_w / 2)
            y2 = int(pred_y + last_locked_h / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame,
                f"LOCKED id={locked_track_id} HOLD",
                (x1, max(20, y1 - 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.circle(frame, (int(pred_x), int(pred_y)), 5, (255, 255, 0), -1)

        writer.write(frame)

        frame_idx += 1

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_pipeline(
        video_path="data/raw/input.mov",
        output_csv_path="artifacts/logs/unused.csv",
        output_video_path="artifacts/overlays/new_pipeline.mp4",
    )