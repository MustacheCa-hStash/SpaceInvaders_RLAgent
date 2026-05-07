import torch

from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack
from agent.replay_buffer import ReplayBuffer
from agent.dqn_agent import DQNAgent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ALESpaceInvaders()
    preprocessor = FramePreprocessor()
    frame_stack = FrameStack()
    replay_buffer = ReplayBuffer(capacity = 100000)
    agent = DQNAgent(env, device)

    num_episodes = 1000
    batch_size = 32
    target_update_freq = 1000
    train_start_size = 10000
    save_freq = 50
    total_steps = 0

    for episode in range(num_episodes):
        frame, info = env.reset()
        processed_frame = preprocessor.preprocess(frame)
        frame_stack.reset(preprocessed_frame)
        state = frame_stack.get_state()

        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.select_action(state)
            next_frame, reward, done, info = env.step(action)
            processed_next_frame = preprocesser.preprocess(next_frame)
            frame_stack.append(processed_next_frame)
            next_state = frame_stack.get_state()
            replay_buffer.append(state, action, reward, next_state, done)

            if len(replay_buffer) >= train_start_size:
                loss = agent.train_step(replay_buffer, batch_size)
            else:
                loss = None

            if total_steps % target_update_freq == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward
            epsiode_steps += 1
            total_steps += 1

        print(f"Episode {episode} | " f"Reward: {episode_reward:.2f} | " f"Steps: {episode_steps} | "
            f"Epsilon: {agent.epsilon:.4f} | " f"Loss: {loss}")

        if episode > 0 and episode % save_freq == 0:
            agent.save("checkpoints/dqn_space_invaders_checkpoint.pt")

    agent.save("checkpoints/dqn_space_invaders_final.pt")
    env.close()

if __name__ == "__main__":
    main()