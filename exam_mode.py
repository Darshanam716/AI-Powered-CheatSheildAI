import cv2
import face_recognition
import numpy as np
import time
from mongo_db import students_col
from camera_config import CAMERA_CONFIG
from exam_cheating import detect_exam_cheating
from exam_evidence import save_and_alert
from phone_detector import detect_phone
from talking_detector import detect_talking
from side_look_detector import detect_side_look


# ---------------- CONFIG ---------------- #
FACE_THRESHOLD = 0.45
FRAME_SCALE = 0.5
ALERT_COOLDOWN = 20

last_alert_time = {}


# ---------------- LOAD STUDENTS ---------------- #
def load_students():
    encodings, info = [], []

    for s in students_col.find():
        for enc in s.get("face_encodings", []):
            encodings.append(np.array(enc))
            info.append({
                "usn": s["usn"],
                "name": s["name"]
            })

    return encodings, info

# ---------------- FRAME PROCESSOR (HYBRID MODE) ---------------- #
def process_exam_frame(frame, cam_id, cfg):
    hall = cfg.get("hall", f"Hall-{cam_id}")

    known_encodings, student_info = load_students()

    small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    unknown_face = False
    detected_students = []

    talking_detected_any = False
    side_look_detected_any = False

    # -------- FACE LOOP --------
    for (top, right, bottom, left), enc in zip(locations, encodings):

        distances = face_recognition.face_distance(
            known_encodings, enc
        )

        student = None
        if len(distances):
            idx = np.argmin(distances)
            if distances[idx] < FACE_THRESHOLD:
                student = student_info[idx]

        # scale back
        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)

        face_box = (top, right, bottom, left)
        face_id = (top // 40, left // 40)

        # -------- TALKING --------
        talking = detect_talking(frame, face_box, face_id)
        if talking:
            talking_detected_any = True
            cv2.putText(frame,
                        "TALKING",
                        (left, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        # -------- SIDE LOOK --------
        side_look = detect_side_look(face_box, face_id)
        if side_look:
            side_look_detected_any = True
            cv2.putText(frame,
                        "SIDE LOOK",
                        (left, bottom + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        # -------- LABEL --------
        if student:
            detected_students.append(student)
            label = f"{student['name']} ({student['usn']})"
            color = (0, 255, 0)
        else:
            unknown_face = True
            label = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

    # -------- PHONE DETECTION --------
    phone_detected = False
    if locations:
        phone_detected = detect_phone(frame, None, cam_id)

    # -------- CHEATING LOGIC --------
    is_cheating, violation = detect_exam_cheating(
        phone_detected=phone_detected,
        unknown_face=unknown_face,
        talking=talking_detected_any,
        side_look=side_look_detected_any
    )

    # -------- ALERT + EVIDENCE --------
    now = time.time()
    last = last_alert_time.get(cam_id, 0)

    if is_cheating and now - last > ALERT_COOLDOWN:

        s = detected_students[0] if detected_students else None

        save_and_alert(
            frame,
            cam_id,
            violation,
            {
                "usn": s["usn"] if s else "UNKNOWN",
                "name": s["name"] if s else "UNKNOWN",
                "hall": hall
            }
        )

        last_alert_time[cam_id] = now

    # -------- STATUS BAR --------
    cv2.putText(frame,
                f"EXAM MODE | {hall} | Faces: {len(locations)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    return frame


# ---------------- EXAM MODE ---------------- #
def start_exam_mode():
    print("\n🔴 EXAM SURVEILLANCE MODE STARTED\n")

    known_encodings, student_info = load_students()
    if not known_encodings:
        print("❌ No students registered")
        return

    # Only exam cameras
    exam_cameras = {
        cid: cfg for cid, cfg in CAMERA_CONFIG.items()
        if cfg.get("mode") == "exam"
    }

    if not exam_cameras:
        print("❌ No exam cameras configured")
        return

    caps = {cid: cv2.VideoCapture(cid) for cid in exam_cameras}

    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue

            hall = exam_cameras[cam_id].get("hall", f"Hall-{cam_id}")

            # ---- Resize for performance ----
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, locations)

            unknown_face = False
            detected_students = []

            talking_detected_any = False
            side_look_detected_any = False

            # -------- FACE LOOP -------- #
            for (top, right, bottom, left), enc in zip(locations, encodings):

                distances = face_recognition.face_distance(
                    known_encodings, enc
                )

                student = None
                if len(distances):
                    idx = np.argmin(distances)
                    if distances[idx] < FACE_THRESHOLD:
                        student = student_info[idx]

                # Scale back
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)

                face_box = (top, right, bottom, left)
                face_id = (top // 40, left // 40)

                # ---- Talking Detection ----
                talking = detect_talking(frame, face_box, face_id)
                if talking:
                    talking_detected_any = True
                    cv2.putText(
                        frame,
                        "TALKING",
                        (left, bottom + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # ---- Side Look Detection ----
                side_look = detect_side_look(face_box, face_id)
                if side_look:
                    side_look_detected_any = True
                    cv2.putText(
                        frame,
                        "SIDE LOOK",
                        (left, bottom + 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # ---- Identity Label ----
                if student:
                    detected_students.append(student)
                    label = f"{student['name']} ({student['usn']})"
                    color = (0, 255, 0)
                else:
                    unknown_face = True
                    label = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # -------- PHONE DETECTION -------- #
            phone_detected = False
            if locations:
                phone_detected = detect_phone(
                    frame,
                    cam_id=cam_id,
                    exam_mode=True
                )

            # -------- CHEATING DECISION -------- #
            is_cheating, violation = detect_exam_cheating(
                phone_detected=phone_detected,
                unknown_face=unknown_face,
                talking=talking_detected_any,
                side_look=side_look_detected_any
            )

            # -------- ALERT CONTROL -------- #
            now = time.time()
            last = last_alert_time.get(cam_id, 0)

            if is_cheating and now - last > ALERT_COOLDOWN:

                s = detected_students[0] if detected_students else None

                save_and_alert(
                    frame,
                    cam_id,
                    violation,
                    {
                        "usn": s["usn"] if s else "UNKNOWN",
                        "name": s["name"] if s else "UNKNOWN",
                        "hall": hall
                    }
                )

                last_alert_time[cam_id] = now

            # -------- STATUS BAR -------- #
            cv2.putText(
                frame,
                f"EXAM MODE | {hall} | Faces: {len(locations)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.imshow(f"CheatShieldAI | Exam | Cam {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()

    cv2.destroyAllWindows()


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    start_exam_mode()