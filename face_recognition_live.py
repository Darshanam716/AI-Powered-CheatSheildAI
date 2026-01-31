import cv2
import face_recognition
import numpy as np
import time
from collections import defaultdict, deque
from mongo_db import students_col


# ---------------- LOAD STUDENTS (GROUPED BY USN) ---------------- #
def load_registered_students():
    student_db = {}

    for student in students_col.find():
        usn = student["usn"]
        student_db[usn] = {
            "name": student["name"],
            "encodings": [np.array(e) for e in student["face_encodings"]]
        }

    return student_db


# ---------------- LIVE RECOGNITION ---------------- #
def start_live_recognition(camera_index=0):
    print(" Loading registered students from database...")
    student_db = load_registered_students()

    if not student_db:
        print(" No students registered")
        return

    cap = cv2.VideoCapture(camera_index)
    time.sleep(1)

    print(" Live recognition started — Press Q to quit")

    frame_count = 0

    #  Anti-blinking memory (last 5 labels per face position)
    face_memory = defaultdict(lambda: deque(maxlen=5))

    SCALE = 0.4
    THRESHOLD = 0.45

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        #  Process every 3rd frame for performance
        if frame_count % 3 != 0:
            cv2.imshow("CheatShieldAI | Live Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Resize + RGB
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), face_enc in zip(locations, encodings):
            best_usn = None
            best_distance = float("inf")

            #  Compare against each student (grouped encodings)
            for usn, data in student_db.items():
                distances = face_recognition.face_distance(
                    data["encodings"], face_enc
                )
                min_dist = np.min(distances)

                if min_dist < best_distance:
                    best_distance = min_dist
                    best_usn = usn

            # Decide recognition
            if best_distance < THRESHOLD:
                label = f'{student_db[best_usn]["name"]} ({best_usn})'
            else:
                label = "Unknown"

            #  Stable face key (rounded position)
            face_key = (top // 15, left // 15)
            face_memory[face_key].append(label)

            #  Majority vote (anti-blinking)
            final_label = max(
                set(face_memory[face_key]),
                key=face_memory[face_key].count
            )

            # Scale back to original frame
            top = int(top / SCALE)
            right = int(right / SCALE)
            bottom = int(bottom / SCALE)
            left = int(left / SCALE)

            # ---------- DRAW FACE BOX ----------
            color = (0, 255, 0) if final_label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # ---------- DRAW LABEL WITH BACKGROUND ----------
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (text_w, text_h), _ = cv2.getTextSize(
                final_label, font, font_scale, thickness
            )

            # Keep label inside frame
            label_y = top - 10
            if label_y - text_h < 0:
                label_y = bottom + text_h + 10

            # Background rectangle
            cv2.rectangle(
                frame,
                (left, label_y - text_h - 6),
                (left + text_w + 6, label_y),
                color,
                -1
            )

            # Text
            cv2.putText(
                frame,
                final_label,
                (left + 3, label_y - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

        cv2.imshow("CheatShieldAI | Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- RUN ---------------- #
if __name__ == "__main__":
    start_live_recognition(0)
