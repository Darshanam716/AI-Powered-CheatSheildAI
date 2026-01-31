import cv2
import face_recognition
import numpy as np
import time
import csv
import os
from datetime import datetime
from mongo_db import students_col, attendance_col, violations_col
from camera_config import CAMERA_CONFIG
from phone_detector import detect_phone
from email_alert import send_email_alert   # ⬅️ helper file

ATTENDANCE_DURATION = 60
attendance_active = True


# ---------------- LOAD STUDENTS ---------------- #
def load_students():
    encodings, info = [], []

    for s in students_col.find():
        student_class = f"{s['branch']}-{s['semester']}{s['section']}"
        student_class = student_class.replace(" ", "").upper()

        for enc in s["face_encodings"]:
            encodings.append(np.array(enc))
            info.append({
                "usn": s["usn"],
                "name": s["name"],
                "branch": s["branch"],
                "classroom": student_class
            })

    return encodings, info


# ---------------- NORMAL MODE ---------------- #
def start_normal_mode_multi():
    global attendance_active

    print("\n SUBJECT ASSIGNMENT (PER CLASSROOM)\n")

    camera_subjects = {}
    normalized_classrooms = {}

    for cam_id, cfg in CAMERA_CONFIG.items():
        classroom = cfg["classroom"].replace(" ", "").upper()
        subject = input(f"Enter subject for {cfg['classroom']} (Cam {cam_id}): ").strip()

        camera_subjects[cam_id] = subject
        normalized_classrooms[cam_id] = classroom

    print("\n Normal Classroom Mode Started")
    print(f" Attendance Window: {ATTENDANCE_DURATION} seconds")

    # -------- CSV SETUP -------- #
    os.makedirs("attendance", exist_ok=True)
    session_time = datetime.now().strftime("%I%p")

    csv_files = {}
    csv_writers = {}

    for cam_id, classroom in normalized_classrooms.items():
        subject = camera_subjects[cam_id]
        path = f"attendance/{classroom}_{subject}_{session_time}.csv"

        f = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(["USN", "Name", "Subject", "Class", "Time", "Camera ID"])

        csv_files[cam_id] = f
        csv_writers[cam_id] = writer

    known_encodings, student_info = load_students()
    if not known_encodings:
        print(" No registered students")
        return

    caps = {cid: cv2.VideoCapture(cid) for cid in CAMERA_CONFIG}
    start_time = time.time()
    attendance_marked = set()

    while True:
        # -------- ATTENDANCE END -------- #
        if attendance_active and time.time() - start_time > ATTENDANCE_DURATION:
            attendance_active = False
            for f in csv_files.values():
                f.flush()
                f.close()

            print("\n Attendance CLOSED")
            print(" CSV files saved")
            print(" Discipline Mode ACTIVE\n")

        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue

            classroom = normalized_classrooms[cam_id]
            subject = camera_subjects[cam_id]

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for (top, right, bottom, left), enc in zip(locations, encodings):
                distances = face_recognition.face_distance(known_encodings, enc)
                student = None

                if len(distances):
                    idx = np.argmin(distances)
                    if distances[idx] < 0.45:
                        student = student_info[idx]

                top *= 2; right *= 2; bottom *= 2; left *= 2

                # -------- ATTENDANCE -------- #
                if attendance_active and student:
                    usn = student["usn"]

                    if usn not in attendance_marked and student["classroom"] == classroom:
                        now = datetime.now()

                        attendance_col.insert_one({
                            "usn": usn,
                            "name": student["name"],
                            "class": classroom,
                            "subject": subject,
                            "camera_id": cam_id,
                            "time": now,
                            "mode": "Normal"
                        })

                        csv_writers[cam_id].writerow([
                            usn,
                            student["name"],
                            subject,
                            classroom,
                            now.strftime("%d-%m-%Y %H:%M:%S"),
                            cam_id
                        ])

                        attendance_marked.add(usn)
                        print(f" [{classroom}] Attendance → {student['name']}")

                # -------- PHONE DISCIPLINE -------- #
                if not attendance_active and student:
                    if detect_phone(frame, student, cam_id):
                        os.makedirs("evidence/phone", exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = f"evidence/phone/{student['usn']}_{ts}.jpg"
                        cv2.imwrite(img_path, frame)

                        violations_col.insert_one({
                            "usn": student["usn"],
                            "name": student["name"],
                            "class": classroom,
                            "branch": student["branch"],
                            "violation": "Phone Usage",
                            "camera_id": cam_id,
                            "time": datetime.now(),
                            "mode": "Normal",
                            "evidence": img_path
                        })

                        send_email_alert(
                            student=student,
                            violation="Phone Usage",
                            image_path=img_path,
                            cam_id=cam_id
                        )

                        cv2.putText(frame, "🚨 PHONE DETECTED",
                                    (left, bottom + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)

                # -------- DRAW FACE -------- #
                label = f"{student['name']} ({student['usn']})" if student else "Unknown"
                color = (0, 255, 0) if student else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            status = "ATTENDANCE" if attendance_active else "DISCIPLINE MODE"
            cv2.putText(frame, f"{classroom} | {status}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

            cv2.imshow(f"CheatShieldAI | Cam {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_normal_mode_multi()
