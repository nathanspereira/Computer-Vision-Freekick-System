"""
1. define/store goal ROI
2. decide when the goal ROI is stable enough
3. accumulate pre-kick background frames
4. build the reference background
5. produce a foreground/difference mask inside the goal region
6. return candidate blobs/centroids from that mask
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from math import hypot

import cv2
import numpy as np


@dataclass
class GoalRegionDebug:
    goal_rect: tuple[int, int, int, int] | None
    stable_frames: int
    has_background: bool
    mean_frame_delta: float | None

# goal class for goal zone to appear after video stops moving (slomo)
# Intended to:
# 1) detect/define a goal rectange
# 2) watch the ROI until it becomes stable for N frames
# 3) collect frames from the stable period
#4) builds a median background reference
#5) when predicted ball enters goal zone, use background reference to reduce noise (net, cars, fence)

class GoalRegionModel:


    def __init__(
        self,
        manual_goal_rect: tuple[int, int, int, int] | None = None,
        stable_threshold: float = 2.5,
        stable_required_frames: int = 12,
        background_frames_needed: int = 20,
        goal_margin: int = 25,
        diff_threshold: int = 25,
        suppress_factor: float = 0.15,
        debug: bool = False,
    ):
        self.manual_goal_rect = manual_goal_rect
        self.stable_threshold = stable_threshold
        self.stable_required_frames = stable_required_frames
        self.background_frames_needed = background_frames_needed
        self.goal_margin = goal_margin
        self.diff_threshold = diff_threshold
        self.suppress_factor = suppress_factor
        self.debug = debug

        self.goal_rect: tuple[int, int, int, int] | None = manual_goal_rect
        self.prev_goal_roi_gray: np.ndarray | None = None
        self.stable_frames = 0
        self.mean_frame_delta: float | None = None

        self._bg_frames: deque[np.ndarray] = deque(maxlen=background_frames_needed)
        self.background_ref: np.ndarray | None = None

    # is called once per frame
    # detects goal post
    #tracks stability
    # accumulates background
    def update(self, frame: np.ndarray) -> None:
 
        if self.goal_rect is None:
            self.goal_rect = self._detect_goal_rect(frame)

        if self.goal_rect is None:
            return

        roi = self._crop_goal(frame)
        if roi is None or roi.size == 0:
            return

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.prev_goal_roi_gray is None or self.prev_goal_roi_gray.shape != roi_gray.shape:
            self.prev_goal_roi_gray = roi_gray
            self.stable_frames = 0
            self.mean_frame_delta = None
            return

        diff = cv2.absdiff(roi_gray, self.prev_goal_roi_gray)
        self.mean_frame_delta = float(np.mean(diff))

        if self.mean_frame_delta <= self.stable_threshold:
            self.stable_frames += 1
        else:
            self.stable_frames = 0
            self._bg_frames.clear()

        self.prev_goal_roi_gray = roi_gray

        if self.background_ref is None and self.stable_frames >= self.stable_required_frames:
            self._bg_frames.append(frame.copy())

            if len(self._bg_frames) >= self.background_frames_needed:
                self.background_ref = self._build_background_reference(list(self._bg_frames))
                if self.debug:
                    print(
                        f"[GoalRegionModel] background built with "
                        f"{len(self._bg_frames)} stable frames"
                    )

    def is_ready(self) -> bool:
        return self.goal_rect is not None and self.background_ref is not None

    def in_goal_zone(self, x: float, y: float) -> bool:
        if self.goal_rect is None:
            return False
        x1, y1, x2, y2 = self.goal_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_background_reference(self) -> np.ndarray | None:
        return self.background_ref

    def get_goal_rect(self) -> tuple[int, int, int, int] | None:
        return self.goal_rect

    def get_debug(self) -> GoalRegionDebug:
        return GoalRegionDebug(
            goal_rect=self.goal_rect,
            stable_frames=self.stable_frames,
            has_background=self.background_ref is not None,
            mean_frame_delta=self.mean_frame_delta,
        )

    def draw_debug(self, frame: np.ndarray) -> None:
        if self.goal_rect is not None:
            x1, y1, x2, y2 = self.goal_rect
            color = (255, 200, 0) if self.background_ref is None else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            status = (
                f"GOAL ROI stable={self.stable_frames}/{self.stable_required_frames}"
                f" bg={'YES' if self.background_ref is not None else 'NO'}"
            )
            if self.mean_frame_delta is not None:
                status += f" delta={self.mean_frame_delta:.2f}"

            cv2.putText(
                frame,
                status,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    # given a crop rectangle in full frame coordinaetes,
    # suppresses static pixels relative to learned background,
    # returns a preprocessed crop
    # if background is unavailble or shape mismatch, falls back to raw crop
    def suppress_static_background(
        self,
        frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
    ) -> np.ndarray:

        x1, y1, x2, y2 = roi_rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return crop

        if self.background_ref is None:
            return crop

        bh, bw = self.background_ref.shape[:2]
        bx1 = max(0, x1)
        by1 = max(0, y1)
        bx2 = min(bw, x2)
        by2 = min(bh, y2)

        if bx2 <= bx1 or by2 <= by1:
            return crop

        bg_crop = self.background_ref[by1:by2, bx1:bx2]
        frame_crop = frame[by1:by2, bx1:bx2]

        if bg_crop.shape != frame_crop.shape:
            return crop

        diff = cv2.absdiff(frame_crop, bg_crop)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, fg_mask = cv2.threshold(diff_gray, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.dilate(fg_mask, kernel)

        bg_mask = cv2.bitwise_not(fg_mask)
        suppressed = frame_crop.copy().astype(np.float32)
        suppressed[bg_mask > 0] *= self.suppress_factor

        return np.clip(suppressed, 0, 255).astype(np.uint8)

    def _crop_goal(self, frame: np.ndarray) -> np.ndarray | None:
        if self.goal_rect is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.goal_rect

        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _build_background_reference(self, frames: list[np.ndarray]) -> np.ndarray:
        stack = np.stack(frames, axis=0).astype(np.float32)
        return np.median(stack, axis=0).astype(np.uint8)

    #one shot estimate of goal opening from visible white post/crossbar structure
    #BUG, not detecting goal yet
    def _detect_goal_rect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

        lines = cv2.HoughLinesP(
            white,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=20,
        )

        if lines is None:
            return None

        v_xs: list[int] = []
        h_ys: list[int] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = hypot(x2 - x1, y2 - y1)
            if length < 30:
                continue

            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 20 or angle > 160:
                h_ys.extend([y1, y2])
            elif 70 < angle < 110:
                v_xs.extend([x1, x2])

        if not v_xs or not h_ys:
            return None

        left_x = int(np.percentile(v_xs, 5))
        right_x = int(np.percentile(v_xs, 95))
        top_y = int(np.percentile(h_ys, 5))

        if right_x - left_x < 60:
            return None

        return (
            max(0, left_x - self.goal_margin),
            max(0, top_y - self.goal_margin),
            min(frame.shape[1], right_x + self.goal_margin),
            frame.shape[0],
        )