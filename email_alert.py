import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def send_email_alert(student, violation, image_path, cam_id):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    msg = EmailMessage()
    msg["Subject"] = f"🚨 CheatShieldAI Alert: {violation}"
    msg["From"] = sender
    msg["To"] = receiver

    msg.set_content(f"""
🚨 VIOLATION DETECTED 🚨

Violation Type : {violation}

Student Name   : {student['name']}
USN            : {student['usn']}
Class          : {student['classroom']}
Department     : {student['branch']}
Camera ID      : {cam_id}
Time           : {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

— CheatShieldAI
""")

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename=os.path.basename(image_path)
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)

    print("📧 Email alert sent successfully")
