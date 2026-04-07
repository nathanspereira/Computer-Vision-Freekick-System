'''
takes raw (x,y) centers from yolo, smooths to ignore jitter,
spits out an approximated Vx and vY
'''
import numpy as np
import cv2

class BallStateTracker:
    def __init__(self):
        # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement Matrix (Maps the 4D state to our 2D YOLO measurement)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        # Transition Matrix (The 2D physics inertia model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        # Process Noise Q (How much we trust our transition math. Lower = trust math more)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement Noise R (How much we trust YOLO. Higher = expect more box jitter)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        
        self.is_initialized = False

    def update(self, x: float, y: float) -> tuple[float, float, float, float]:
        """
        Call this every frame YOLO successfully locks onto the ball.
        Returns the smoothed (x, y, vx, vy).
        """
        if not self.is_initialized:
            # First frame: force the state to exactly where YOLO saw it, 0 velocity
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) # Initialize error covariance
            self.is_initialized = True

            return x, y, 0.0, 0.0
            
        # 1. Predict where it should be
        self.kf.predict()
        
        # 2. Correct the prediction with the actual YOLO measurement
        measurement = np.array([[x], [y]], dtype=np.float32)
        estimated = self.kf.correct(measurement)
        
        return float(estimated[0][0]), float(estimated[1][0]), float(estimated[2][0]), float(estimated[3][0])
        
    def predict_blind(self) -> tuple[float, float, float, float]:
        """
        Call this when YOLO loses the ball (e.g., in the white net).
        It advances the physics engine one frame into the future based purely on momentum.
        Returns the predicted (x, y, vx, vy).
        """
        if not self.is_initialized:
            return 0.0, 0.0, 0.0, 0.0
            
        predicted = self.kf.predict()
        return float(predicted[0][0]), float(predicted[1][0]), float(predicted[2][0]), float(predicted[3][0])

    def initialize(self, x: float, y: float, frame_idx: int | None = None) -> None:
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.is_initialized = True

        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0

        if frame_idx is not None:
            self.last_frame_idx = frame_idx