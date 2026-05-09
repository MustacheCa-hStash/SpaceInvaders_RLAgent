import torch

from analytics_logger import EpisodeAnalytics
from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.frame_stack import FrameStack
from agent.replay_buffer import ReplayBuffer
from agent.dqn_agent import DQNAgent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ALESpaceInvadersEnv()
    preprocessor = FramePreprocessor()
    frame_stack = FrameStack()
    replay_buffer = ReplayBuffer(capacity = 100000)
    agent = DQNAgent(env, device)
    analytics_logger = EpisodeAnalytics("logs/training_analytics.csv")

    # Confirm running on GPU
    print(next(agent.online_net.parameters()).device)

    num_episodes = 1000
    batch_size = 32
    target_update_freq = 1000
    train_start_size = 10000
    save_freq = 50
    total_steps = 0
    STEP_DOWNTIME = 130

    for episode in range(num_episodes):

        frame, info = env.reset()

        for _ in range(STEP_DOWNTIME):
            #NOOP: 0 first 130 actions
            next_frame, reward, done, info = env.step(0)

        frame = next_frame
        processed_frame = preprocessor.preprocess(frame)
        frame_stack.reset(processed_frame)
        state = frame_stack.get_state()

        done = False
        episode_reward = 0.0
        cumulative_loss = 0.0
        num_training_updates = 0
        episode_steps = 0

        q_max_sum = 0.0
        q_mean_sum = 0.0
        q_gap_sum = 0.0
        q_stat_steps = 0

        episode_env_reward = 0.0
        lives_lost = 0
        previous_lives = info["lives"]

        while not done:
            action = agent.select_action(state)

            q_max, q_mean, q_gap = agent.get_q_stats(state)
            q_max_sum += q_max
            q_mean_sum += q_mean
            q_gap_sum += q_gap
            q_stat_steps += 1

            next_frame, reward, done, info = env.step(action)

            episode_env_reward += info["env_reward"]

            if info["lives"] < previous_lives:
                lives_lost += previous_lives - info["lives"]

            previous_lives = info["lives"]

            processed_next_frame = preprocessor.preprocess(next_frame)
            frame_stack.append(processed_next_frame)
            next_state = frame_stack.get_state()
            replay_buffer.append(state, action, reward, next_state, done)

            if len(replay_buffer) >= train_start_size:
                loss = agent.train_step(replay_buffer, batch_size)
                cumulative_loss += loss
                num_training_updates += 1
            else:
                loss = None

            if total_steps % target_update_freq == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        if num_training_updates > 0:
            average_loss = cumulative_loss / num_training_updates
        else:
            average_loss = None

        mean_max_q = q_max_sum / q_stat_steps
        mean_q = q_mean_sum / q_stat_steps
        mean_q_gap = q_gap_sum / q_stat_steps

        analytics_logger.write_episode({
            "episode": episode,
            "total_steps": total_steps,
            "episode_steps": episode_steps,
            "episode_reward": episode_reward,
            "episode_env_reward": episode_env_reward,
            "lives_lost": lives_lost,
            "epsilon": agent.epsilon,
            "average_loss": average_loss,
            "mean_max_q": mean_max_q,
            "mean_q": mean_q,
            "mean_q_gap": mean_q_gap,
        })

        print(f"Episode {episode} | " f"Reward: {episode_reward:.2f} | " f"Steps: {episode_steps} | "
            f"Epsilon: {agent.epsilon:.4f} | " f"Average Loss: {average_loss}")

        if episode > 0 and episode % save_freq == 0:
            agent.save("checkpoints/dqn_space_invaders_checkpoint.pt")

    agent.save("checkpoints/dqn_space_invaders_final.pt")
    env.close()

if __name__ == "__main__":
    main()