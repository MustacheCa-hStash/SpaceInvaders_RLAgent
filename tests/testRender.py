from ale_py import ALEInterface
from ale_py.roms import get_rom_path

import cv2
import random

ale = ALEInterface()

rom_path = get_rom_path("space_invaders")

ale.loadROM(rom_path)

actions = ale.getMinimalActionSet()

while True:

    if ale.game_over():
        ale.reset_game()

    frame = ale.getScreenRGB()

    frame_bgr = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow(
        "Space Invaders",
        frame_bgr
    )

    action = random.choice(actions)

    reward = ale.act(action)
    print("Reward:", reward)

    key = cv2.waitKey(30)

    if key == ord("q"):
        break

cv2.destroyAllWindows()