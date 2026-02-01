def detect_exam_cheating(phone_detected=False,
                          unknown_face=False,
                          talking=False,
                          side_look=False):

    if phone_detected:
        return True, "Phone Usage"

    if unknown_face:
        return True, "Unknown Person"

    if talking and side_look:
        return True, "Talking + Side Look"

    if talking:
        return True, "Talking"

    if side_look:
        return True, "Side Looking"

    return False, None
