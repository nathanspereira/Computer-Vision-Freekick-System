import cv2
class OverlayProcessor:
    def draw_ball(self, frame, pos):
        # Draw a circle at the detected position
        return cv2.circle(frame, pos, 10, (0, 255, 0), -1)