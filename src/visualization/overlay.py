import cv2
import numpy as np

from ..tracking.track_state import TrackState, TrackStatus
from ..tracking.lock_on_tracker import LockOnState, LockState


def _heatmap_color(frac: float):
    t = max(0.0, min(1.0, frac))
    if t < 0.5:
        s = t / 0.5
        return (int(255*(1-s)), int(255*s), 0)
    s = (t-0.5)/0.5
    return (0, int(255*(1-s)), int(255*s))


def _draw_heatmap_trail(frame, history, trail_length=40): #was 90
    visible = list(history)[-trail_length:]
    n = len(visible)
    if n < 2:
        return
    for i, (px, py) in enumerate(visible):
        frac = i / max(n-1, 1)
        cv2.circle(frame, (int(px), int(py)), max(2, int(2+3*frac)),
                   _heatmap_color(frac), -1)


def _draw_lockon_reticle(frame, cx, cy, r, color=(0,255,80), label="LOCKED"):
    BLACK = (0, 0, 0)
    outer = int(r)
    tick  = 12
    gap   = 6
    blen  = 22

    for col, thick in [(BLACK, 4), (color, 2)]:
        cv2.circle(frame, (cx, cy), outer, col, thick)
        for start in [225, 315, 45, 135]:
            cv2.ellipse(frame, (cx, cy), (outer, outer),
                        0, start, start+blen, col, thick)
        for dx, dy in [(0,-1),(0,1),(1,0),(-1,0)]:
            cv2.line(frame,
                     (cx+dx*(outer+gap),   cy+dy*(outer+gap)),
                     (cx+dx*(outer+gap+tick), cy+dy*(outer+gap+tick)),
                     col, thick)

    # Centroid crosshair — marks the original YOLO detection center
    # Tracked centroid marker — black X for debugging
    xsize = 10
    cv2.line(frame, (cx - xsize, cy - xsize), (cx + xsize, cy + xsize), BLACK, 3)
    cv2.line(frame, (cx - xsize, cy + xsize), (cx + xsize, cy - xsize), BLACK, 3)

    # Label
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx, ty = cx-tw//2, cy-outer-16
    cv2.putText(frame, label, (tx+1,ty+1), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)


def _draw_charge_bar(frame, cx, cy, r, charge, max_charge):
    BAR_W  = 12
    BAR_H  = 60
    bar_x  = int(cx + r + 14)
    bar_y  = int(cy - BAR_H//2)
    fill_h = int(BAR_H * charge / max(max_charge,1))
    color  = (0, 215, 255)

    cv2.rectangle(frame, (bar_x,bar_y), (bar_x+BAR_W, bar_y+BAR_H), (40,40,40), -1)
    if fill_h > 0:
        cv2.rectangle(frame,
                      (bar_x, bar_y+BAR_H-fill_h),
                      (bar_x+BAR_W, bar_y+BAR_H), color, -1)
    cv2.rectangle(frame, (bar_x,bar_y), (bar_x+BAR_W,bar_y+BAR_H), (160,160,160), 1)

    lbl = "LOCKING"
    (tw,_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, lbl, (bar_x+BAR_W//2-tw//2, bar_y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    frac = f"{charge}/{max_charge}"
    (fw,_),_ = cv2.getTextSize(frac, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(frame, frac, (bar_x+BAR_W//2-fw//2, bar_y+BAR_H+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1, cv2.LINE_AA)


def draw_track_state(
    frame: np.ndarray,
    state: TrackState,
    lock_state: LockOnState,
    position_history=None,
    midline_y: int | None = None,
) -> np.ndarray:
    output = frame.copy()

    # ── Red midline ───────────────────────────────────────────────────────────
    if midline_y is not None:
        cv2.line(output, (0,midline_y), (frame.shape[1],midline_y), (0,0,255), 2)
        cv2.putText(output, "no-track zone", (10,midline_y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    # ── LOST ──────────────────────────────────────────────────────────────────
    if lock_state.lock_state == LockState.LOST:
        if position_history:
            _draw_heatmap_trail(output, position_history)
        cv2.rectangle(output, (14,10), (100,44), (0,0,0), -1)
        cv2.putText(output, "LOST", (19,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 2, cv2.LINE_AA)
        return output

    # ── All YOLO detections shown in blue ─────────────────────────────────────
    for yd in lock_state.yolo_detections:
        yhw = max(int(yd.w / 2), 8)
        yhh = max(int(yd.h / 2), 8)
        yx, yy = int(yd.x), int(yd.y)

        # blue bbox
        cv2.rectangle(output, (yx - yhw, yy - yhh), (yx + yhw, yy + yhh), (255, 0, 0), 2)

        # small blue center point for raw YOLO centroid
        cv2.circle(output, (yx, yy), 3, (255, 0, 0), -1)

        cv2.putText(
            output,
            f"YOLO {yd.confidence:.3f}",
            (yx - yhw, yy - yhh - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # ── Heatmap trail ─────────────────────────────────────────────────────────
    if position_history:
        _draw_heatmap_trail(output, position_history)

    cx = int(lock_state.centroid_x)
    cy = int(lock_state.centroid_y)
    r  = lock_state.circle_radius

    # ── SEARCHING ─────────────────────────────────────────────────────────────
    if lock_state.lock_state == LockState.SEARCHING:
        for ping in lock_state.blob_pings:
            px, py = int(ping.x), int(ping.y)
            cv2.circle(output, (px, py), 4, (0, 255, 255), -1)
            cv2.circle(output, (px, py), 4, (0, 140, 255), 1)
        if state.status == TrackStatus.DETECTED:
            hw = max(int(state.w/2), 8) if state.w else 20
            hh = max(int(state.h/2), 8) if state.h else 20
            scx, scy = int(state.x), int(state.y)
            cv2.rectangle(output, (scx-hw,scy-hh), (scx+hw,scy+hh), (0,255,0), 2)
            cv2.circle(output, (scx, scy), 4, (0, 255, 0), -1)
            _draw_charge_bar(output, scx, scy, hw,
                             lock_state.charge, lock_state.max_charge)
        label_txt   = f"SEARCHING  {lock_state.charge}/{lock_state.max_charge}"
        label_color = (0, 220, 0)

    # ── LOCKED ────────────────────────────────────────────────────────────────
    elif lock_state.lock_state == LockState.LOCKED:
        # Blob pings inside circle — cyan dots
        for ping in lock_state.blob_pings:
            px, py = int(ping.x), int(ping.y)
            cv2.circle(output, (px,py), 4, (255,255,0), -1)
            cv2.circle(output, (px,py), 4, (0,0,0),     1)

        _draw_lockon_reticle(output, cx, cy, r,
                             color=(0,255,80), label="LOCKED")
        label_txt   = "LOCKED"
        label_color = (0, 255, 80)

    # ── POST_KICK ─────────────────────────────────────────────────────────────
    else:
        # Expanded circle in orange — ball has been kicked
        for ping in lock_state.blob_pings:
            px, py = int(ping.x), int(ping.y)
            cv2.circle(output, (px,py), 4, (0,165,255), -1)
            cv2.circle(output, (px,py), 4, (0,0,0),     1)

        _draw_lockon_reticle(output, cx, cy, r,
                             color=(0,165,255), label="KICKED — TRACKING")
        label_txt   = "POST-KICK"
        label_color = (0, 165, 255)

    # ── Status label ──────────────────────────────────────────────────────────
    (tw,th),_ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(output, (14,10), (14+tw+10, 10+th+10), (0,0,0), -1)
    cv2.putText(output, label_txt, (19, 10+th+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

    return output