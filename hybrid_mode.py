import cv2
from camera_config import CAMERA_CONFIG

# reuse your existing pipelines
from normal_mode_multi import process_normal_frame
from exam_mode import process_exam_frame


def start_hybrid_mode():
    print("\n🟡 HYBRID MODE STARTED")
    print("Cam0 → Normal Classroom")
    print("Cam1 → Exam Surveillance\n")

    # Open cameras
    caps = {cid: cv2.VideoCapture(cid) for cid in CAMERA_CONFIG}

    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue

            cfg = CAMERA_CONFIG[cam_id]
            mode = cfg.get("mode", "normal")

            # ---------- NORMAL CLASSROOM ----------
            if mode == "normal":
                frame = process_normal_frame(frame, cam_id, cfg)

            # ---------- EXAM MODE ----------
            elif mode == "exam":
                frame = process_exam_frame(frame, cam_id, cfg)

            cv2.imshow(f"Hybrid Mode | Cam {cam_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in caps.values():
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_hybrid_mode()
