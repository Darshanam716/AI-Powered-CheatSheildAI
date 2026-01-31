import cv2

cam0 = cv2.VideoCapture(0)  # Inbuilt camera
cam1 = cv2.VideoCapture(1)  # External webcam

while True:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0:
        cv2.imshow("Cam 0 - Inbuilt", frame0)
    if ret1:
        cv2.imshow("Cam 1 - USB Webcam", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.release()
cam1.release()
cv2.destroyAllWindows()
