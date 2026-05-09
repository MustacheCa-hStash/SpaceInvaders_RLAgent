import torch

from env.ale_space_invaders import ALESpaceInvadersEnv
from agent.dqn_agent import DQNAgent
from preprocessing.frame_preprocessor import FramePreprocessor, FrameMaxer
from preprocessing.frame_stack import FrameStack

MODEL_PATH = "checkpoints/dqn_space_invaders_final.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = ALESpaceInvadersEnv(
        render_delay=10,
        terminate_on_life_loss = False
    )

    preprocessor = FramePreprocessor()
    frame_maxer = FrameMaxer()
    frame_stack = FrameStack(stack_size = 4)

    agent = DQNAgent(env = env, device = DEVICE)
    agent.load(MODEL_PATH)
    agent.online_net.eval()

    frame, info = env.reset()

    STEP_DOWNTIME = 130

    for _ in range(STEP_DOWNTIME):
        next_frame, reward, done, info = env.step(0)

    frame = next_frame

    processed_frame = preprocessor.preprocess(frame)

    maxed_frame = frame_maxer.reset(processed_frame)

    frame_stack.reset(maxed_frame)

    state = frame_stack.get_state()

    done = False
    episode_reward = 0.0
    episode_steps = 0

    while not done:
        action = agent.select_greedy_action(state)

        print("Action Selected:", env.actions[action])

        next_frame, reward, done, info = env.step(action)

        env.render()

        processed_next_frame = preprocessor.preprocess(next_frame)

        maxed_next_frame = frame_maxer.apply(processed_next_frame)

        frame_stack.append(maxed_next_frame)

        state = frame_stack.get_state()

        episode_reward += reward
        episode_steps += 1

        print(
            f"Step: {episode_steps} | "
            f"Action: {info['action_name']} | "
            f"Reward: {reward:.2f} | "
            f"Lives: {info['lives']}"
        )

    print(
        f"\nEpisode complete | "
        f"Reward: {episode_reward:.2f} | "
        f"Steps: {episode_steps}"
    )

    env.close()


if __name__ == "__main__":
    main()