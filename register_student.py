from mongo_db import students_col
from face_engine import capture_multiple_faces

def register_student():
    print("\n CheatShieldAI | Student Registration\n")

    # --- Collect student details ---
    usn = input("Enter USN: ").strip().upper()
    name = input("Enter Student Name: ").strip()
    branch = input("Enter Branch (e.g., CS): ").strip().upper()
    semester = input("Enter Semester (e.g., 5): ").strip()
    section = input("Enter Section (e.g., A): ").strip().upper()

    # --- Check if USN already exists ---
    if students_col.find_one({"usn": usn}):
        print(" USN already exists!")
        return

    # --- Capture multiple face samples ---
    print("\n Capturing multiple face samples...")
    encodings = capture_multiple_faces(usn, camera_index=0, samples=10)

    if len(encodings) < 5:
        print(" Not enough face samples captured")
        return

    # --- Prepare student data ---
    student_data = {
        "usn": usn,
        "name": name,
        "branch": branch,
        "semester": semester,
        "section": section,
        "face_encodings": encodings   #  store multiple encodings
    }

    # --- Insert into MongoDB ---
    students_col.insert_one(student_data)
    print("\n Student registered successfully in CheatShieldAI!")

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    register_student()
