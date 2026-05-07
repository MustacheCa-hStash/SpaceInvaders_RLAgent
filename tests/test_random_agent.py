from env.ale_space_invaders import ALESpaceInvadersEnv
import random

env = ALESpaceInvadersEnv()

obs, info = env.reset()

try:
    while True:

        action = random.randrange(env.num_actions)

        obs, reward, done, info = env.step(action)
        print(info)

        env.render()

        if done:
            obs, info = env.reset()
finally:
    env.close()