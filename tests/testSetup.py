from env.ale_space_invaders import ALESpaceInvadersEnv


STEP_DOWNTIME = 130
RENDER_DELAY = 200
LEFT_BOUND = 0
RIGHT_BOUND = 164


def find_action_index(env, action_name):
    for i, action in enumerate(env.actions):
        if action.name == action_name:
            return i

    raise ValueError(f"{action_name} not found.")


def get_x(info):
    return info["pre_action_net_x_pos"]


def check_bounds(info, label, step):
    x = get_x(info)

    if x < LEFT_BOUND or x > RIGHT_BOUND:
        raise AssertionError(
            f"{label} step {step}: net_x_pos out of bounds: {x}"
        )


def run_action(env, action_index, steps, label, render=True):
    last_x = None
    last_info = None

    for step in range(1, steps + 1):
        frame, reward, done, info = env.step(action_index)
        last_info = info

        check_bounds(info, label, step)

        x = get_x(info)

        print(
            f"{label} | "
            f"Step {step}/{steps} | "
            f"Action: {info['action_name']} | "
            f"pre_action_net_x_pos: {x} | "
            f"reward: {reward:.2f}"
        )

        if last_x is not None:
            if label.startswith("RIGHT") and last_x == RIGHT_BOUND and x != RIGHT_BOUND:
                raise AssertionError("Drift detected at right boundary.")

            if label.startswith("LEFT") and last_x == LEFT_BOUND and x != LEFT_BOUND:
                raise AssertionError("Drift detected at left boundary.")

        last_x = x

        if render:
            env.render()

        if done:
            print("Game ended early.")
            return True, last_info

    return False, last_info


def main():
    env = ALESpaceInvadersEnv(render_delay=RENDER_DELAY)

    print("Action Set:")
    for i, action in enumerate(env.actions):
        print(i, action.name)

    noop_idx = find_action_index(env, "NOOP")
    left_idx = find_action_index(env, "LEFT")
    right_idx = find_action_index(env, "RIGHT")

    frame, info = env.reset()

    print("\nStartup...")
    for step in range(1, STEP_DOWNTIME + 1):
        frame, reward, done, info = env.step(noop_idx)
        check_bounds(info, "STARTUP", step)

        if done:
            print("Game ended during startup.")
            env.close()
            return

    print("\nStress test beginning...\n")

    tests = [
        ("LEFT slam from start", left_idx, 30),
        ("RIGHT to wall plus extra", right_idx, 220),
        ("RIGHT extra at wall", right_idx, 30),
        ("LEFT to wall plus extra", left_idx, 220),
        ("LEFT extra at wall", left_idx, 30),
        ("RIGHT to wall again", right_idx, 220),
        ("LEFT to wall again", left_idx, 220),
    ]

    for label, action_idx, steps in tests:
        ended, last_info = run_action(env, action_idx, steps, label, render=True)

        if ended:
            env.close()
            return

        print(f"Finished {label}. Last pre_action_net_x_pos = {get_x(last_info)}\n")

    print("Boundary tracking test complete. No out-of-bounds drift detected.")

    for _ in range(30):
        env.render()

    env.close()


if __name__ == "__main__":
    main()