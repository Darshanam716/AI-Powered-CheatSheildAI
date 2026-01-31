import face_recognition
import cv2
import os
import time

def capture_multiple_faces(usn, camera_index=0, samples=10):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(" Camera not accessible")
        return []

    base_dir = f"data/faces/{usn}"
    os.makedirs(base_dir, exist_ok=True)

    encodings = []
    count = 0

    print("\n CAPTURING FACE DATA")
    print(" Look straight, left, right, up slightly")
    print(f" Capturing {samples} samples\n")

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")

        for (top, right, bottom, left) in locations:
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Samples captured: {count}/{samples}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        cv2.imshow("CheatShieldAI | Multi-Face Registration", frame)

        if len(locations) == 1:
            encoding = face_recognition.face_encodings(rgb, locations)[0]
            encodings.append(encoding.tolist())

            img_path = f"{base_dir}/img_{count+1}.jpg"
            cv2.imwrite(img_path, frame)

            print(f" Captured sample {count+1}")
            count += 1
            time.sleep(0.5)  # spacing between samples

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return encodings
