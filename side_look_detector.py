import cv2
import numpy as np
from collections import defaultdict, deque

# Memory of face movement
face_history = defaultdict(lambda: deque(maxlen=8))

# ---- Parameters ----
SIDE_MOVE_THRESHOLD = 12   # pixel movement threshold
SIDE_FRAME_COUNT = 4       # frames needed


def detect_side_look(face_box, face_id):
    """
    Detects repeated left/right head movement.
    """

    top, right, bottom, left = face_box

    # Face center
    center_x = (left + right) // 2

    history = face_history[face_id]
    history.append(center_x)

    if len(history) < 5:
        return False

    moves = 0

    for i in range(1, len(history)):
        if abs(history[i] - history[i - 1]) > SIDE_MOVE_THRESHOLD:
            moves += 1

    if moves >= SIDE_FRAME_COUNT:
        return True

    return False
