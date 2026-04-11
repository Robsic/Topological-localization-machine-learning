import cv2


class OpenCVPreview:
    def __init__(self, window_name="Capture", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height

    def show(self, frame, fps=10):
        resized = cv2.resize(frame, (self.width, self.height))
        cv2.imshow(self.window_name, resized)
        key = cv2.waitKey(int(max(1, 1000 / max(1, fps)))) & 0xFF
        return key != ord('q')

    def close(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

