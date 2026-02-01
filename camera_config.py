"""
CAMERA_CONFIG = {
    0: {
        "mode": "normal",          # normal | exam
        "classroom": "CS-5A",
        "section": "A",
        "location": "Classroom CS-5A"
    },
    1: {
        "mode": "normal",          # change to "exam" when needed
        "classroom": "CS-5B",
        "section": "B",
        "location": "Classroom CS-5B"
    }
}
"""

    #  Example exam camera (future-ready)
CAMERA_CONFIG = {
     0: {
         "mode": "exam",
         "hall": "Exam Hall 1",
         "expected_branch": "CS"
     },
      1: {
         "mode": "exam",
         "hall": "Exam Hall 2",
         "expected_branch": "IS"
     },
}
