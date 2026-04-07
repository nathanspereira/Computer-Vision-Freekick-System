#1. Extract Features: Use SIFT to detect keypoints and compute descriptors for two consecutive frames It and It+1)
#2. match keypoints: Match descriptors between the two frames using a matcher (like FlannBasedMatcher) to identify corresponding points.
# 3. Use RANSAC or similar methods to filter out bad matches and find consistent clusters that agree on the motion, thus removing outliers.
#4. calculate rotation: Using the matched pairs, compute the rotational movement (delta theta) between frames. 
# This is often done by calculating the transformation matrix (Homography).
# 5. Compute angular velocity Divide the rotation angle by the time difference (delta t) between frames:

import cv2
import numpy as np
import math

class BallRPMEstimator:
    def __init__(self, fps: float):
        self.fps = fps
        # 1. Initialize SIFT (Limit features since the ball is tiny)
        self.sift = cv2.SIFT_create(nfeatures=50, contrastThreshold=0.03)
        
        # 2. Initialize FLANN Matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # State variables for frame-to-frame comparison
        self.prev_kp = None
        self.prev_desc = None

    def _extract_ball_roi(self, frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray | None:
        """Helper to crop the ball from the frame and enhance contrast."""
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Boundary checks
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return None
            
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to aggressively boost the contrast of the ball panels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced_roi = clahe.apply(gray_roi)
        return enhanced_roi

    def estimate_rpm(self, frame: np.ndarray, x: float, y: float, w: float, h: float) -> float | None:
        roi = self._extract_ball_roi(frame, x, y, w, h)
        if roi is None:
            return None

        # Step 1: Detect and Compute
        kp, desc = self.sift.detectAndCompute(roi, None)
        
        if desc is None or len(kp) < 3:
            self.prev_kp, self.prev_desc = kp, desc
            return None # Not enough features to track rotation

        if self.prev_desc is not None and len(self.prev_kp) >= 3:
            # Step 2: Match Keypoints using KNN (k=2 for Lowe's ratio test)
            matches = self.flann.knnMatch(self.prev_desc, desc, k=2)
            
            # Filter matches using Lowe's ratio test (removes ambiguous matches)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Step 3 & 4: RANSAC and Affine Transform
            if len(good_matches) >= 3:
                src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # estimateAffinePartial2D restricts the transform to rotation, translation, and uniform scale
                matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                
                if matrix is not None:
                    # The matrix is 2x3: [[a, b, tx], [-b, a, ty]]
                    # Rotation angle can be extracted using atan2
                    a = matrix[0, 0]
                    b = matrix[1, 0]
                    
                    # Angle in radians (delta theta)
                    theta_rad = math.atan2(b, a)
                    
                    # Step 5: Compute Angular Velocity (RPM)
                    # RPM = (radians / 2*pi) * fps * 60 seconds
                    rpm = (abs(theta_rad) / (2 * math.pi)) * self.fps * 60.0
                    
                    self.prev_kp, self.prev_desc = kp, desc
                    return rpm

        # Save current state for the next frame
        self.prev_kp, self.prev_desc = kp, desc
        return None