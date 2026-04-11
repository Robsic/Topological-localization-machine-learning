import cv2
import os

class VideoCapture:
    def __init__(self, fps=10, width=640, height=480, device="/dev/video1"):
        """Initializes the camera with the provided parameters."""
        device = os.path.realpath(device)  # in case you still pass by-id
        self.device = device
        self.fps = fps
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        # Prefer MJPG for stability on many UVC cameras
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"USB camera not available ({self.device})")

    def read_frame(self):
        """Reads a frame from the camera and returns it in grayscale."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from USB camera.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def release(self):
        """Unlock camera features"""
        self.cap.release()
        cv2.destroyAllWindows()
