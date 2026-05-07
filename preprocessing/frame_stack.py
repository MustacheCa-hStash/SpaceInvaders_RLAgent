from collections import deque
import numpy as np

class FrameStack:
    def __init__(self, stack_size = 4):
        self.stack_size = stack_size
        self.frames = deque(maxlen = stack_size)

    def reset(self, frame):
        self.frames.clear()
        for i in range(self.stack_size):
            self.append(frame)

    def append(self, frame):
        self.frames.append(frame)

    def get_state(self):
        return np.stack(self.frames, axis = 0)