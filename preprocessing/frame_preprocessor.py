import cv2

class FramePreprocessor:
    def __init__(self, width = 84, height = 84):
        self.target_width = width
        self.target_height = height

    def preprocess(self, frame):
        resized_frame = cv2.resize(frame, (self.target_width, self.target_height),
                                   interpolation = cv2.INTER_AREA)

        return resized_frame