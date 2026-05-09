class SpaceInvadersRewardFunction:
    def compute(self, env_reward, previous_lives, current_lives, done):
        reward = 0.0

        reward -= 0.01

        if previous_lives > current_lives:
            reward -= 100.0

        reward += env_reward

        return reward