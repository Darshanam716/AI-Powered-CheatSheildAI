def detect_exam_cheating(phone_detected, unknown_face, talking=False):

    if phone_detected:
        return True, "Phone Usage"

    if unknown_face:
        return True, "Unknown Face"

    if talking:
        return True, "Talking"

    return False, None
