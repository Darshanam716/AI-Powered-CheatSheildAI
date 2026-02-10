import cv2
import time
from ultralytics import YOLO

# ---------------- YOLO MODEL ---------------- #
model = YOLO("yolov8n.pt")

# ---------------- CONFIG ---------------- #
PHONE_CLASS = "cell phone"
CONF_THRESHOLD = 0.35
COOLDOWN = 15  # seconds per camera/student

last_detect_time = {}


# ---------------- PHONE DETECTION ---------------- #
def detect_phone(frame, student=None, cam_id=0, exam_mode=False):
    """
    Returns True if phone detected.

    Normal Mode:
        - student REQUIRED
        - USN cooldown

    Exam/Hybrid Mode:
        - student optional
        - camera cooldown
        - Evidence handled outside
    """

    now = time.time()

    # -------- SAFE COOLDOWN KEY -------- #
    if exam_mode or student is None:
        cooldown_key = f"cam_{cam_id}"
    else:
        cooldown_key = student["usn"]

    # -------- COOLDOWN -------- #
    if now - last_detect_time.get(cooldown_key, 0) < COOLDOWN:
        return False

    # -------- YOLO INFERENCE -------- #
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name != PHONE_CLASS or conf < CONF_THRESHOLD:
                continue

            # -------- DRAW BOX -------- #
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 0, 255), 2)

            cv2.putText(frame,
                        f"PHONE {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

            last_detect_time[cooldown_key] = now
            return True

    return False
