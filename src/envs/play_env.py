import pygame
from meeting_room import MeetingRoom


if __name__ == "__main__":
    env = MeetingRoom(render_mode="human")
    env.metadata['video.frames_per_second'] = 60
    env.reset()
    keys2actions = env.get_keys_to_action()

    running = True
    while running:
        action = None
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_r:  # 'R': reset
                    env.reset()
                else:
                    action = keys2actions.get((event.key,))

        if action is not None:
            _, reward, _, _, _, = env.step(action)
            if float(reward) != 0:
                print(f"Reward {reward}")

        env.render()
