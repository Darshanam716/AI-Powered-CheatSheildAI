import cv2
import numpy as np
from collections import defaultdict, deque

# Memory of previous mouth frames
mouth_memory = defaultdict(lambda: deque(maxlen=8))

# ---- Tunable Parameters ----
MOTION_THRESHOLD = 22        # sensitivity
ACTIVE_FRAME_COUNT = 5       # frames needed to confirm talking
MIN_MOTION_PERCENT = 0.08    # percent of mouth moving


def detect_talking(frame, face_box, face_id):
    """
    Smart talking detection using mouth motion consistency.
    """

    top, right, bottom, left = face_box

    # ---- Extract mouth region ----
    height = bottom - top
    mouth_top = top + int(height * 0.6)

    mouth = frame[mouth_top:bottom, left:right]

    if mouth.size == 0:
        return False

    gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

    # Normalize size to avoid shape mismatch
    gray = cv2.resize(gray, (90, 45))

    prev_frames = mouth_memory[face_id]

    if len(prev_frames) == 0:
        prev_frames.append(gray)
        return False

    # ---- Motion difference ----
    diff = cv2.absdiff(prev_frames[-1], gray)

    motion_value = np.mean(diff)

    # Percentage of pixels changing
    moving_pixels = np.sum(diff > 25)
    motion_percent = moving_pixels / diff.size

    prev_frames.append(gray)

    # ---- Count active motion frames ----
    active_frames = 0
    for i in range(1, len(prev_frames)):
        d = cv2.absdiff(prev_frames[i - 1], prev_frames[i])
        if np.mean(d) > MOTION_THRESHOLD:
            active_frames += 1

    # ---- Talking decision ----
    if (
        active_frames >= ACTIVE_FRAME_COUNT and
        motion_percent > MIN_MOTION_PERCENT
    ):
        return True

    return False
