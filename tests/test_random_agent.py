from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack
import random

env = ALESpaceInvadersEnv()
preprocessor = FramePreprocessor()
frame_stack = FrameStack()

raw_frame, info = env.reset()
processed_frame = preprocessor.preprocess(raw_frame)
frame_stack.reset(processed_frame)

state = frame_stack.get_state()
print(state.shape)

try:
    while True:

        action = random.randrange(env.num_actions)
        raw_frame, reward, done, info = env.step(action)

        processed_frame = preprocessor.preprocess(raw_frame)
        frame_stack.append(processed_frame)

        state = frame_stack.get_state()

        print(info)

        env.render()

        if done:
            raw_frame, info = env.reset()
            processed_frame = preprocessor.preprocess(raw_frame)
            frame_stack.reset(processed_frame)
finally:
    env.close()