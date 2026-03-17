# 1. Loads video from data/raw/
# 2. Use detector from src/detection/
# 3. save data via src/io/csv_logger.py
# 4. draw bozes using src/visualization/overlay.py

import cv2
from src.io.csv_logger import CSVLogger 
from src.visualization.overlay import OverlayProcessor

def run_baseline(videopath):
    cap = cv2.VideoCapture(video_path)
    logger = CSVLogger("artifacts/logs/positions.csv")
    visualizer = OverlayProcessor()

    #define codec and output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('artifacts/overlays/baseline_run.mp4', fourcc, 30.0, (1920, 1080))

    from_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Dummy detection (You will replace this with your YOLO/Classical code)
        ball_pos = (500 + frame_idx, 300) 
        
        # 1. Log Data
        logger.log_frame(frame_idx, ball_pos)
        
        # 2. Draw Overlay
        frame = visualizer.draw_ball(frame, ball_pos)
        
        # 3. Save Video Frame
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Baseline Run Complete!")

if __name__ == "__main__":
    run_baseline("data/raw/sample_kick.mp4")