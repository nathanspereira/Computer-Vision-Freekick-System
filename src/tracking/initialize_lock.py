from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class LockInitResult:
    locked_track_id: Optional[int]
    best_score: float
    appearances: int
    best_candidate: Optional[object]

def initialize_lock(
    frame_candidates: List[List[object]],
    min_confidence: float = 0.08,
    min_box_w: float = 10.0,
    min_box_h: float = 10.0,
) -> LockInitResult:

    scores: Dict[int, float] = defaultdict(float)
    appearances: Dict[int, int] = defaultdict(int)
    best_conf_for_id: Dict[int, float] = defaultdict(float)
    best_candidate_for_id: Dict[int, object] = {}

    total_seen = 0
    no_track_id = 0
    low_conf = 0
    small_box = 0
    passed = 0

    for frame_idx, candidates in enumerate(frame_candidates):
        for c in candidates:
            total_seen += 1

            track_id = getattr(c, "track_id", None)
            confidence = float(getattr(c, "confidence", 0.0))
            w = float(getattr(c, "w", 0.0))
            h = float(getattr(c, "h", 0.0))

            if track_id is None:
                no_track_id += 1
                continue
            if confidence < min_confidence:
                low_conf += 1
                continue
            if w < min_box_w or h < min_box_h:
                small_box += 1
                continue

            passed += 1
            tid = int(track_id)
            scores[tid] += confidence
            appearances[tid] += 1

            if confidence > best_conf_for_id[tid]:
                best_conf_for_id[tid] = confidence
                best_candidate_for_id[tid] = c

    print("\n[initialize_lock debug]")
    print(f"total_seen   = {total_seen}")
    print(f"no_track_id  = {no_track_id}")
    print(f"low_conf     = {low_conf}")
    print(f"small_box    = {small_box}")
    print(f"passed       = {passed}")

    if not appearances:
        return LockInitResult(
            locked_track_id=None,
            best_score=0.0,
            appearances=0,
            best_candidate=None,
        )

    winning_track_id = max(
        appearances.keys(),
        key=lambda tid: (
            appearances[tid],
            scores[tid],
            best_conf_for_id[tid],
        ),
    )

    return LockInitResult(
        locked_track_id=winning_track_id,
        best_score=scores[winning_track_id],
        appearances=appearances[winning_track_id],
        best_candidate=best_candidate_for_id[winning_track_id],
    )