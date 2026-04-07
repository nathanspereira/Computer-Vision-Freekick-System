import cv2
from src.physics.rpm_estimator import BallRPMEstimator

def run_sift_test(video_path: str, fps: float = 240.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return

    estimator = BallRPMEstimator(fps=fps)
    
    # Read the first frame
    ret, frame1 = cap.read()
    if not ret:
        return

    print("Draw a box around the ball. Press SPACE or ENTER to confirm.")
    # This pauses the script and opens a window for you to draw the bounding box
    bbox = cv2.selectROI("Select Ball", frame1, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Ball")

    # bbox is (x_top_left, y_top_left, width, height)
    # Convert it to the center (x, y, w, h) format our estimator expects
    w, h = bbox[2], bbox[3]
    x = bbox[0] + w / 2
    y = bbox[1] + h / 2

    print(f"\nCaptured ROI: x={x}, y={y}, w={w}, h={h}")
    
    # Run frame 1 to initialize SIFT keypoints
    estimator.estimate_rpm(frame1, x, y, w, h)

    # Step forward a few frames to give it time to rotate
    frames_to_skip = 5
    for _ in range(frames_to_skip - 1):
        cap.read() # throw away intermediate frames
        
    ret, frame_future = cap.read()
    
    if ret:
        # Run the future frame to calculate the rotation match
        rpm = estimator.estimate_rpm(frame_future, x, y, w, h)
        
        if rpm is not None:
            print(f"SUCCESS! Estimated RPM: {rpm:.2f}")
        else:
            print("FAILED. SIFT could not find enough matching points to calculate rotation.")

    cap.release()

if __name__ == "__main__":
    run_sift_test("data/raw/input.mov", fps=240.0)