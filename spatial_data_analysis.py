import os
import matplotlib.pyplot as plt

from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack

OUTPUT_DIR = "debug_frames"
NUM_STEPS = 200
SAVE_EVERY = 10


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = ALESpaceInvadersEnv(render_delay=1)
    preprocessor = FramePreprocessor()
    frame_stack = FrameStack(stack_size=4)

    frame, info = env.reset()
    processed_frame = preprocessor.preprocess(frame)
    frame_stack.reset(processed_frame)

    for step in range(NUM_STEPS):
        # Simple action choice for inspection only.
        # You can replace this with agent.select_greedy_action(state).
        action = 0

        next_frame, reward, done, info = env.step(action)
        processed_next_frame = preprocessor.preprocess(next_frame)
        frame_stack.append(processed_next_frame)

        if step % SAVE_EVERY == 0:
            state = frame_stack.get_state()

            fig, axes = plt.subplots(1, 4, figsize=(10, 3))

            for i in range(4):
                axes[i].imshow(state[i], cmap="gray")
                axes[i].set_title(f"Stack frame {i}")
                axes[i].axis("off")

            fig.suptitle(f"Step {step} | Action: {info['action_name']} | Lives: {info['lives']}")
            plt.tight_layout()

            save_path = os.path.join(OUTPUT_DIR, f"stack_step_{step:04d}.png")
            plt.savefig(save_path)
            plt.close(fig)

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()