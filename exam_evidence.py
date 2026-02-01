import os
import cv2
import smtplib
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv
from mongo_db import exam_logs_col

# Load env
load_dotenv()

SENDER = os.getenv("EMAIL_SENDER")
PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER = os.getenv("EMAIL_RECEIVER")


def save_and_alert(frame, cam_id, violation_type, student=None):
    """
    student = {
        usn, name, hall
    } or None
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    usn = student["usn"] if student else "UNKNOWN"
    name = student["name"] if student else "UNKNOWN"
    hall = student.get("hall", "Unknown") if student else "Unknown"

    # ---------- SAVE EVIDENCE ----------
    folder = f"exam_evidence/{usn}"
    os.makedirs(folder, exist_ok=True)

    img_path = f"{folder}/{violation_type}_{timestamp}.jpg"
    cv2.imwrite(img_path, frame)

    # ---------- DATABASE LOG ----------
    exam_logs_col.insert_one({
        "usn": usn,
        "name": name,
        "violation": violation_type,
        "hall": hall,
        "camera_id": cam_id,
        "time": datetime.now(),
        "evidence": img_path
    })

    # ---------- EMAIL ----------
    msg = EmailMessage()
    msg["Subject"] = f"🚨 CheatShieldAI EXAM ALERT: {violation_type}"
    msg["From"] = SENDER
    msg["To"] = RECEIVER

    msg.set_content(f"""
🚨 CHEATING DETECTED 🚨

Violation Type : {violation_type}
USN            : {usn}
Name           : {name}
Exam Hall      : {hall}
Camera ID      : {cam_id}
Time           : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

Evidence attached.

— CheatShieldAI
""")

    with open(img_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(img_path)
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER, PASSWORD)
        server.send_message(msg)

    print(f"🚨 EXAM ALERT SENT → {violation_type} | USN: {usn}")
