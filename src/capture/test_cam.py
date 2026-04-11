import cv2

cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
print("isOpened:", cap.isOpened())

ret, frame = cap.read()
print("read:", ret, "frame is None:", frame is None, "shape:", None if frame is None else frame.shape)

if ret and frame is not None:
    cv2.imshow("USB Cam", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
