class SpaceInvadersRewardFunction:
    def compute(self, env_reward, previous_lives, current_lives, done):
        reward = 0.0

        if previous_lives > current_lives:
            reward -= 15.0

        reward += env_reward

        return reward