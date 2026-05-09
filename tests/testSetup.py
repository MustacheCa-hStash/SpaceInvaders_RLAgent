from env.ale_space_invaders import ALESpaceInvadersEnv


STEP_DOWNTIME = 130
RIGHT_STEPS = 165
LEFT_STEPS = 164
RENDER_DELAY = 50


def find_action_index(env, action_name):
    for i, action in enumerate(env.actions):
        if action.name == action_name:
            return i

    raise ValueError(f"{action_name} action not found.")


def run_fixed_action(env, action_index, num_steps, label):
    for step in range(1, num_steps + 1):
        frame, reward, done, info = env.step(action_index)

        env.render()

        print(
            f"{label} | "
            f"Step {step}/{num_steps} | "
            f"Action: {info['action_name']} | "
            f"Episode Frame: {info['episode_frame']}"
        )

        if done:
            print("Game ended early.")
            return True

    return False


def main():
    env = ALESpaceInvadersEnv(render_delay=RENDER_DELAY)

    print("Action Set:")
    for i, action in enumerate(env.actions):
        print(i, action.name)

    noop_idx = find_action_index(env, "NOOP")
    right_idx = find_action_index(env, "RIGHT")
    left_idx = find_action_index(env, "LEFT")

    frame, info = env.reset()

    print(f"\nRendering startup NOOP period ({STEP_DOWNTIME} steps)...\n")

    # Render startup delay steps too
    for step in range(1, STEP_DOWNTIME + 1):
        frame, reward, done, info = env.step(noop_idx)

        env.render()

        print(
            f"STARTUP | "
            f"Step {step}/{STEP_DOWNTIME} | "
            f"Action: {info['action_name']} | "
            f"Episode Frame: {info['episode_frame']}"
        )

        if done:
            print("Game ended during startup.")
            env.close()
            return

    print("\nRunning RIGHT 164 then LEFT 163...\n")

    if run_fixed_action(env, right_idx, RIGHT_STEPS, "RIGHT"):
        env.close()
        return

    if run_fixed_action(env, left_idx, LEFT_STEPS, "LEFT"):
        env.close()
        return

    print("\nSequence complete.")

    # Hold final frame briefly
    for _ in range(20):
        env.render()

    env.close()


if __name__ == "__main__":
    main()