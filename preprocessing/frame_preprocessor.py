import cv2

class FramePreprocessor:
    def __init__(self, width = 84, height = 84):
        self.target_width = width
        self.target_height = height

    def preprocess(self, frame):
        resized_frame = cv2.resize(frame, (self.target_width, self.target_height),
                                   interpolation = cv2.INTER_AREA)

        return resized_frame

class FrameMaxer:
    def __init__(self):
        self.previous_frame = None

    def reset(self, frame):
        self.previous_frame = frame
        return frame

    def apply(self, frame):
        maxed_frame = np.maximum(self.previous_frame, frame)
        self.previous_frame = frame
        return maxed_frame