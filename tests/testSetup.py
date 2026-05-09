import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack
from preprocessing.frame_preprocessor import FrameMaxer
from agent.dqn_agent import DQNAgent


MODEL_PATH = "checkpoints/dqn_space_invaders_final.pt"

OUTPUT_DIR = "debug_frame_max_policy"
STEP_DOWNTIME = 130
NUM_STEPS = 300
SAVE_EVERY = 2
RENDER_DELAY = 20


def save_comparison(previous_frame, current_frame, maxed_frame,
                    stack_state, step, action_name):

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(previous_frame, cmap="gray")
    axes[0].set_title("Previous")

    axes[1].imshow(current_frame, cmap="gray")
    axes[1].set_title("Current")

    axes[2].imshow(maxed_frame, cmap="gray")
    axes[2].set_title("Pixelwise Max")

    # Show latest stacked frame channel
    axes[3].imshow(stack_state[-1], cmap="gray")
    axes[3].set_title("Stack Latest")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Step {step} | Action: {action_name}")

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"frame_max_step_{step:04d}.png"
    )

    plt.savefig(save_path)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ALESpaceInvadersEnv(render_delay=RENDER_DELAY)

    preprocessor = FramePreprocessor()
    frame_maxer = FrameMaxer()
    frame_stack = FrameStack()

    agent = DQNAgent(env, device)
    agent.load(MODEL_PATH)

    frame, info = env.reset()

    # Startup NOOP period
    for _ in range(STEP_DOWNTIME):
        frame, reward, done, info = env.step(0)

        if done:
            print("Game ended during startup.")
            env.close()
            return

    processed_frame = preprocessor.preprocess(frame)

    maxed_frame = frame_maxer.reset(processed_frame)

    frame_stack.reset(maxed_frame)

    state = frame_stack.get_state()

    previous_processed = processed_frame

    done = False

    for step in range(1, NUM_STEPS + 1):

        action = agent.select_greedy_action(state)

        next_frame, reward, done, info = env.step(action)

        env.render()

        current_processed = preprocessor.preprocess(next_frame)

        maxed_next_frame = frame_maxer.apply(current_processed)

        frame_stack.append(maxed_next_frame)

        next_state = frame_stack.get_state()

        if step % SAVE_EVERY == 0:
            save_comparison(
                previous_processed,
                current_processed,
                maxed_next_frame,
                next_state,
                step,
                info["action_name"]
            )

        previous_processed = current_processed
        state = next_state

        print(
            f"Step {step} | "
            f"Action: {info['action_name']} | "
            f"Reward: {reward:.2f}"
        )

        if done:
            print("Episode ended.")
            break

    env.close()

    print(f"\nSaved debug frames to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()