import numpy as np
import pygame

surface = None
clock = None

ROT_MATRIX = np.array([[0, -1], [1, 0]])


def render_options_overlay(image, option_trace: list[int] = None, fps=None) -> None:
    """Displays an RGB pixel image using pygame.
        Optional: Show the currently used option_id and/or the detected objects
        and their velocities."""

    if fps is None:
        fps = 15

    global surface, clock

    # Prepare screen if not initialized
    if surface is None:
        if image is None:
            return
        pygame.init()
        pygame.display.set_caption("Environment")
        surface = pygame.display.set_mode((image.shape[1], image.shape[0]), flags=pygame.SCALED)
        clock = pygame.time.Clock()

    clock.tick(fps)  # reduce FPS for Atari games

    # Render RGB image
    image = np.transpose(image, (1, 0, 2))
    pygame.pixelcopy.array_to_surface(surface, image)

    # If given, render option ID in top right corner
    if option_trace is not None and len(option_trace) > 0:
        option_text = f"Option {option_trace[0]}"
        for option_id in option_trace[1:]:
            option_text += "-%d" % option_id
        draw_label(surface, option_text, (10, surface.get_size()[1] - 50),
                   font=pygame.font.SysFont('Source Code Pro', 30),
                   text_color=(0, 0, 0), bg_color=(255, 255, 255))

    pygame.display.flip()
    pygame.event.pump()


def draw_arrow(surface: pygame.Surface, start_pos: (float, float), end_pos: (float, float),
               tip_length: int = 6, tip_width: int = 6, **kwargs):
    start_pos = np.asarray(start_pos)
    end_pos = np.asarray(end_pos)

    # Arrow body
    pygame.draw.line(surface, start_pos=start_pos, end_pos=end_pos, **kwargs)

    # Arrow tip
    arrow_dir = end_pos - start_pos
    arrow_dir_norm = arrow_dir / np.linalg.norm(arrow_dir)
    tip_anchor = end_pos - tip_length * arrow_dir_norm

    left_tip_end = tip_anchor + tip_width / 2 * np.matmul(ROT_MATRIX, arrow_dir_norm)
    right_tip_end = tip_anchor - tip_width / 2 * np.matmul(ROT_MATRIX, arrow_dir_norm)

    pygame.draw.line(surface, start_pos=left_tip_end, end_pos=end_pos, **kwargs)
    pygame.draw.line(surface, start_pos=right_tip_end, end_pos=end_pos, **kwargs)


def draw_label(surface: pygame.Surface,
               text: str,
               position: (int, int),
               font: pygame.font.SysFont,
               text_color=None,
               bg_color=None):
    """Renders a framed label text to a pygame surface."""
    if text_color is None:
        text_color = (255, 255, 255)
    if bg_color is None:
        bg_color = (0, 0, 0)

    text = font.render(text, True, text_color, None)
    text_rect = text.get_rect()

    frame_rect = text_rect.copy()
    frame_rect.topleft = position
    frame_rect.w += 5
    frame_rect.h += 6

    frame_surface = pygame.Surface((frame_rect.w, frame_rect.h))
    frame_surface.set_alpha(80)  # make transparent

    # Draw label background
    frame_surface.fill(bg_color)
    surface.blit(frame_surface, position)

    # Draw text
    text_rect.topleft = position[0] + 3, position[1] + 3
    surface.blit(text, text_rect)
