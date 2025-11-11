from typing import List, Tuple
import numpy as np


def preds_to_segments(
    preds: np.ndarray,
    window_length: float = 0.025,
    window_step: float = 0.01,
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    if preds.size == 0:
        return segments
    in_seg = False
    start_idx = 0
    for i, v in enumerate(preds):
        if v == 1 and not in_seg:
            in_seg = True
            start_idx = i
        elif v == 0 and in_seg:
            in_seg = False
            end_idx = i - 1
            start_t = start_idx * window_step
            end_t = end_idx * window_step + window_length
            segments.append((round(start_t, 3), round(end_t, 3)))
    if in_seg:
        end_idx = len(preds) - 1
        start_t = start_idx * window_step
        end_t = end_idx * window_step + window_length
        segments.append((round(start_t, 3), round(end_t, 3)))
    return segments
