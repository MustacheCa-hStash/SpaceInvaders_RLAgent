# visualize_cnn_activations.py

import os
import torch
import matplotlib.pyplot as plt

from env.ale_space_invaders import ALESpaceInvadersEnv
from agent.dqn_agent import DQNAgent
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack


MODEL_PATH = "checkpoints/dqn_space_invaders_final.pt"
OUTPUT_DIR = "debug_activations"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_STEPS = [50, 100, 150, 200, 250, 300]


def save_frame_stack(state, output_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

    for i in range(4):
        axes[i].imshow(state[i], cmap="gray")
        axes[i].set_title(f"Frame {i}")
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def save_activation_grid(activation, output_path, title, max_channels=32):
    # activation shape: (1, C, H, W)
    activation = activation.squeeze(0).detach().cpu()

    num_channels = min(max_channels, activation.shape[0])
    cols = 8
    rows = (num_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 1.8 * rows))
    axes = axes.flatten()

    for i in range(num_channels):
        feature_map = activation[i]

        axes[i].imshow(feature_map, cmap="gray")
        axes[i].set_title(f"ch {i}", fontsize=8)
        axes[i].axis("off")

    for i in range(num_channels, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def print_q_values(agent, state, step):
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE) / 255.0
        x = x.unsqueeze(0)

        q_values = agent.online_net(x).squeeze(0).detach().cpu()

    print(f"\nStep {step}")
    print("Q-values:", q_values.tolist())
    print("Best action index:", torch.argmax(q_values).item())

    top2 = torch.topk(q_values, k=2).values
    print("Max Q:", top2[0].item())
    print("Second Q:", top2[1].item())
    print("Q gap:", (top2[0] - top2[1]).item())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = ALESpaceInvadersEnv(render_delay=1)
    preprocessor = FramePreprocessor()
    frame_stack = FrameStack(stack_size=4)

    agent = DQNAgent(env=env, device=DEVICE)
    agent.load(MODEL_PATH)
    agent.online_net.eval()

    frame, info = env.reset()
    processed_frame = preprocessor.preprocess(frame)
    frame_stack.reset(processed_frame)
    state = frame_stack.get_state()

    done = False
    step = 0

    while not done and step <= max(TARGET_STEPS):
        action = agent.select_greedy_action(state)
        next_frame, reward, done, info = env.step(action)

        processed_next_frame = preprocessor.preprocess(next_frame)
        frame_stack.append(processed_next_frame)
        state = frame_stack.get_state()

        if step in TARGET_STEPS:
            x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE) / 255.0
            x = x.unsqueeze(0)

            with torch.no_grad():
                activations = agent.online_net.extract_conv_activations(x)

            prefix = os.path.join(OUTPUT_DIR, f"step_{step:04d}")

            save_frame_stack(
                state,
                f"{prefix}_stack.png",
                f"Input stack | step {step} | action {info['action_name']} | lives {info['lives']}"
            )

            save_activation_grid(
                activations["conv1"],
                f"{prefix}_conv1.png",
                f"Conv1 activations | step {step}",
                max_channels=32
            )

            save_activation_grid(
                activations["conv2"],
                f"{prefix}_conv2.png",
                f"Conv2 activations | step {step}",
                max_channels=32
            )

            save_activation_grid(
                activations["conv3"],
                f"{prefix}_conv3.png",
                f"Conv3 activations | step {step}",
                max_channels=32
            )

            print_q_values(agent, state, step)

        step += 1

    env.close()


if __name__ == "__main__":
    main()