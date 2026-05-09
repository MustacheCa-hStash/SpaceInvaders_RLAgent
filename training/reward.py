from ale_py import Action

class SpaceInvadersRewardFunction:
    def compute(self, episode_step, net_x_pos, action_enum, env_reward, life_lost):
        reward = 0.0

        reward -= 0.01

        if episode_step <= 130:
            reward += 0.01

        if life_lost:
            reward -= 150.0

        reward += env_reward

        if action_enum in [Action.LEFT, Action.LEFTFIRE] and net_x_pos == 0:
            reward -= 0.05
        elif action_enum in [Action.RIGHT, Action.RIGHTFIRE] and net_x_pos == 164:
            reward -= 0.05

        return reward