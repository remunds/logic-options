import numpy as np
import pygame

surface = None
clock = None


def render_oc_overlay(image, obs: np.ndarray = None, option_trace: list[int] = None) -> None:
    """Displays an RGB pixel image using pygame.
        Optional: Show the currently used option_id and/or the detected objects
        and their velocities."""

    global surface, clock

    # Prepare screen if not initialized
    if surface is None:
        if image is None:
            return
        pygame.init()
        pygame.display.set_caption("OCOC Environment")
        surface = pygame.display.set_mode((image.shape[1], image.shape[0]), flags=pygame.SCALED)
        clock = pygame.time.Clock()

    clock.tick(15)  # reduce FPS for Atari games

    # Render RGB image
    image = np.transpose(image, (1, 0, 2))
    pygame.pixelcopy.array_to_surface(surface, image)

    # If given, render option ID in top right corner
    if option_trace is not None and len(option_trace) > 0:
        option_text = f"Option {option_trace[0]}"
        for option_id in option_trace[1:]:
            option_text += "-%d" % option_id
        font = pygame.font.SysFont('Pixel12x10', 12)
        text = font.render(option_text, True, (255, 255, 255), None)
        rect = text.get_rect()
        rect.bottomright = (surface.get_size()[0], surface.get_size()[1])
        pygame.draw.rect(surface, color=(20, 20, 20), rect=rect)
        surface.blit(text, rect)

    # If given, render object coordinates and velocity vectors
    if obs is not None:
        # TODO: generalize to arbitrary focus files
        obj_positions = obs[0, :18*4].reshape(-1, 4) * 164
        for (x, y, prev_x, prev_y) in obj_positions:
            if x == np.nan:
                continue

            # Draw an 'X' at object center
            pygame.draw.line(surface, color=(255, 255, 255),
                             start_pos=(x - 2, y - 2), end_pos=(x + 2, y + 2))
            pygame.draw.line(surface, color=(255, 255, 255),
                             start_pos=(x - 2, y + 2), end_pos=(x + 2, y - 2))

            dx = x - prev_x
            dy = y - prev_y

            # Draw velocity vector
            if dx != 0 or dy != 0:
                # if abs(dx) > 100 or abs(dy) > 100:
                #     print(f"Large velocity dx={dx}, dy={dy} encountered!")
                pygame.draw.line(surface, color=(100, 200, 255),
                                 start_pos=(float(x), float(y)), end_pos=(x + 8 * dx, y + 8 * dy))

    pygame.display.flip()
    pygame.event.pump()
