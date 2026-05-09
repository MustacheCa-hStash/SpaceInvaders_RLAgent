import torch

from analytics_logger import EpisodeAnalytics
from env.ale_space_invaders import ALESpaceInvadersEnv
from preprocessing.frame_preprocessor import FramePreprocessor, FrameMaxer
from preprocessing.frame_stack import FrameStack
from agent.replay_buffer import ReplayBuffer
from agent.dqn_agent import DQNAgent

RESUME_TRAINING = True
CHECKPOINT_PATH = "checkpoints/dqn_space_invaders_checkpoint.pt"
FINAL_MODEL_PATH = "checkpoints/dqn_space_invaders_final.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = ALESpaceInvadersEnv()
    preprocessor = FramePreprocessor()
    frame_maxer = FrameMaxer()
    frame_stack = FrameStack()
    replay_buffer = ReplayBuffer(capacity = 100000)
    agent = DQNAgent(env, device)

    total_steps = 0
    start_episode = 0

    if RESUME_TRAINING:
        metadata = agent.load(CHECKPOINT_PATH)
        total_steps = metadata["total_steps"]
        start_episode = metadata["episode"] + 1
        print(f"Resumed from episode {start_episode}, total_steps {total_steps}")

    analytics_logger = EpisodeAnalytics("logs/training_analytics.csv")

    # Confirm running on GPU
    print(next(agent.online_net.parameters()).device)

    num_episodes = 1000
    batch_size = 32
    target_update_freq = 1000
    train_start_size = 10000
    save_freq = 50
    STEP_DOWNTIME = 130

    for episode in range(start_episode, start_episode + num_episodes):

        frame, info = env.reset()

        for _ in range(STEP_DOWNTIME):
            #NOOP: 0 first 130 actions
            next_frame, reward, done, info = env.step(0)

        frame = next_frame
        processed_frame = preprocessor.preprocess(frame)
        maxed_frame = frame_maxer.reset(processed_frame)
        frame_stack.reset(maxed_frame)
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
            maxed_next_frame = frame_maxer.apply(processed_next_frame)
            frame_stack.append(maxed_next_frame)
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
            agent.save(CHECKPOINT_PATH, total_steps = total_steps, episode = episode)

    agent.save(FINAL_MODEL_PATH, total_steps = total_steps, episode = episode)
    env.close()

if __name__ == "__main__":
    main()