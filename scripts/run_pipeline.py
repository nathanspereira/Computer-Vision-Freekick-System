from __future__ import annotations
from src.physics.ball_state import BallStateTracker
from math import hypot
from pathlib import Path
#from src.tracking.goal_region_model import GoalRegionModel
from src.tracking.initialize_lock import initialize_lock
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
#linear extrapolator (motion prediction, straight line guesser)
# reworked: 
# history: deque of frame_idx, cx, cy
# current_frame_idx: absolute frame number the tracker is currently processing
def predict_next_center(history: deque[tuple[int, float, float]], current_frame_idx: int) -> tuple[float, float] | None:
    if len(history) == 0:
        return None
    if len(history) == 1:
        return (history[-1][1], history[-1][2])

    recent_history = list(history)[-60:]
    
    frames = np.array([p[0] for p in recent_history])
    xs = np.array([p[1] for p in recent_history])
    ys = np.array([p[2] for p in recent_history])
    
    last_frame, last_x, last_y = recent_history[-1]
    target_frame = current_frame_idx + 1
    frames_ahead = target_frame - last_frame
    
    # Fit BOTH x(frame) and y(frame) with degree 2 (Gravity + Magnus Effect)
    if len(recent_history) >= 20:
        poly_x = np.polyfit(frames, xs, 2)
        pred_x = np.polyval(poly_x, target_frame)
        
        poly_y = np.polyfit(frames, ys, 2)
        pred_y = np.polyval(poly_y, target_frame)
    else:
        # Fallback to linear if we only have 2 points
        poly_x = np.polyfit(frames, xs, 1)
        pred_x = np.polyval(poly_x, target_frame)
        
        poly_y = np.polyfit(frames, ys, 1)
        pred_y = np.polyval(poly_y, target_frame)

    # Clamping Rules
    dxs = np.abs(np.diff(xs) / np.diff(frames))
    dys = np.abs(np.diff(ys) / np.diff(frames))
    
    FIXED_SAFE_CAP_X = 15.0 
    FIXED_SAFE_CAP_Y = 15.0
    
    max_dx_per_frame = max(np.median(dxs) * 2, FIXED_SAFE_CAP_X)
    max_dy_per_frame = max(np.median(dys) * 2, FIXED_SAFE_CAP_Y)
    
    max_allowed_dx = max_dx_per_frame * frames_ahead
    max_allowed_dy = max_dy_per_frame * frames_ahead
    
    clamped_x = np.clip(pred_x, last_x - max_allowed_dx, last_x + max_allowed_dx)
    clamped_y = np.clip(pred_y, last_y - max_allowed_dy, last_y + max_allowed_dy)
    
    return (float(clamped_x), float(clamped_y))


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

    MIN_BOX_W = 18
    MIN_BOX_H = 18

    HISTORY_LEN = 150 #was 30, stores 1 full second of flight data

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

    ACQUISITION_FRAMES = 10
    INIT_MIN_CONF = 0.30
    INIT_MIN_BOX_W = 10.0
    INIT_MIN_BOX_H = 10.0

    #detection delcaration
    detector = YOLODetector(
        model_path="yolo26m.pt",
        confidence_threshold=0.01,
        target_class_id=32,
        tracker_path="src/tracking/bytetrack_fast_ball.yaml",
    )
    #metadeta
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
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
    locked_history: deque[tuple[int, float, float]] = deque(maxlen=HISTORY_LEN)
    trajectory_points: deque[tuple[int, int]] = deque(maxlen=TRAJECTORY_MAX_LEN)


    tracker = BallStateTracker()
    
    last_locked_w = 24.0
    last_locked_h = 24.0
    misses_since_locked = 0

    last_roi_box: tuple[int, int, int, int] | None = None

    acquisition_candidates = []
    initial_lock_done = False 
    #loop that reads frame by frame
    while True:
        valid, frame = cap.read()
        if not valid:
            break
        
        if not initial_lock_done:
            candidates = detector.detect(frame, frame_idx)
        else:
            candidates = []

        if not initial_lock_done:
            last_roi_box = None
            acquisition_candidates.append(candidates)

            if len(acquisition_candidates) == ACQUISITION_FRAMES:
                init_result = initialize_lock(
                    acquisition_candidates,
                    min_confidence=INIT_MIN_CONF,
                    min_box_w=INIT_MIN_BOX_W,
                    min_box_h=INIT_MIN_BOX_H,
                )

                if init_result.best_candidate is not None:
                    initial_lock_done = True
                    locked_track_id = init_result.locked_track_id

                    locked_candidate = init_result.best_candidate 
                    last_locked_w = locked_candidate.w 
                    last_locked_h = locked_candidate.h 
                    misses_since_locked = 0

                    tracker.initialize(
                        locked_candidate.x,
                        locked_candidate.y,
                        frame_idx=locked_candidate.frame_idx,
                    )

                    locked_history.append(
                            (locked_candidate.frame_idx, locked_candidate.x, locked_candidate.y)
                    )
                else:
                    acquisition_candidates.pop(0)
            frame_idx += 1
            continue



        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}: {len(candidates)} candidates")


        locked_candidate: BallTrackCandidate | None = None
        predicted_center = predict_next_center(locked_history, frame_idx)
        if predicted_center is not None:
            px, py = int(predicted_center[0]), int(predicted_center[1])
            #draw a orange dot for the raw math prediction
            #cv2.circle(frame, (px, py), 4, (0, 165, 255), -1)

        last_locked_x = locked_history[-1][1] if len(locked_history) > 0 else None
        last_locked_y = locked_history[-1][2] if len(locked_history) > 0 else None
        '''
        # case 1: if already locked, prefer same ByteTrack ID
        held_prediction = None
        locked_candidate = None
        if initial_lock_done:
            if locked_track_id is not None:
                same_id  = [c for c in candidates if c.track_id == locked_track_id]
                if same_id:
                    locked_candidate = max(same_id, key=lambda c: c.confidence)
        '''
        # case 2: ROI fallback when locked target disappears OR gets weak
        used_roi_fallback = False
        if (
            initial_lock_done
            and predicted_center is not None
            and misses_since_locked <= ROI_MAX_CONSECUTIVE_FRAMES
        ):
            should_try_roi = (
                locked_candidate is None
                or locked_candidate.confidence < ROI_CONF_TRIGGER
            )

            if should_try_roi:
                pred_x, pred_y = float(predicted_center[0]), float(predicted_center[1])

                search_x1 = int(pred_x - ROI_HALF_SIZE)
                search_y1 = int(pred_y - ROI_HALF_SIZE)
                search_x2 = int(pred_x + ROI_HALF_SIZE)
                search_y2 = int(pred_y + ROI_HALF_SIZE)

                last_roi_box = (search_x1, search_y1, search_x2, search_y2)

                
            
                roi_candidates = detector.detect_in_roi(
                    frame=frame,
                    frame_idx=frame_idx,
                    center_x=pred_x,
                    center_y=pred_y,
                    roi_half_size=ROI_HALF_SIZE,
                    roi_confidence_threshold=0.03,
                    background_ref= None,
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

            #feed raw yolo centers to Kalman filter
            smooth_x, smooth_y, vx, vy = tracker.update(locked_candidate.x, locked_candidate.y)

            #draw smoothed kalman state

            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 6, (255, 0, 0), -1)
            cv2.putText(
                frame, 
                f"Vx:{vx:.1f} Vy:{vy:.1f}", 
                (int(smooth_x) + 10, int(smooth_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 0, 0), 
                2
            )
            locked_history.append((frame_idx, locked_candidate.x, locked_candidate.y))
            trajectory_points.append((int(locked_candidate.x), int(locked_candidate.y)))
            last_locked_w = locked_candidate.w
            last_locked_h = locked_candidate.h
            misses_since_locked = 0

            last_roi_box = None
        else:
            misses_since_locked += 1
            
            if len(locked_history) > 0:
                pred_x, pred_y, vx, vy = tracker.predict_blind()
                cv2.circle(frame, (int(pred_x), int(pred_y)), 6, (255, 255, 0), -1)

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

        if last_roi_box is not None:
            rx1, ry1, rx2, ry2 = last_roi_box 
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 1)
            cv2.putText(
                frame, 
                "AREA OF INTEREST", 
                (rx1, ry1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (0, 0, 255), 
                1
            )
        writer.write(frame)

        frame_idx += 1

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_pipeline(
        #video_path="data/raw/input.mov",
        video_path="data/raw/nathan_day3.2.mov",
        output_csv_path="artifacts/logs/unused1.csv",
        output_video_path="artifacts/overlays/newest_output.mp4",
    )