"""
Rendering utilities for GridWorld environment.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from gridworld_env.layout import Layout
from gridworld_env.objects import KeyColor


# ASCII rendering characters
ASCII_CHARS = {
    "wall": "#",
    "empty": ".",
    "agent": "@",
    "agent_with_red_key": "R",
    "agent_with_blue_key": "B",
    "key_red": "r",
    "key_blue": "b",
    "door_closed": "D",
    "door_open": "_",
    "reward": "$",
    "reward_destroyed": "x",
}

# Colors for pygame rendering (RGB)
COLORS = {
    "background": (40, 40, 40),
    "wall": (80, 80, 80),
    "floor": (200, 200, 200),
    "agent": (50, 150, 50),
    "key_red": (220, 60, 60),
    "key_blue": (60, 60, 220),
    "door_closed": (139, 90, 43),
    "door_open": (180, 140, 90),
    "reward": (255, 215, 0),
    "reward_destroyed": (100, 100, 100),
    "posner_cue_red": (255, 100, 100),
    "posner_cue_blue": (100, 100, 255),
    "text": (255, 255, 255),
}


class Renderer:
    """
    Renderer for GridWorld environment.

    Supports both ASCII and pygame-based rendering.
    """

    def __init__(
        self,
        width: int,
        height: int,
        render_mode: str,
        cell_size: int = 32,
    ):
        """
        Initialize the renderer.

        Args:
            width: Grid width in cells.
            height: Grid height in cells.
            render_mode: 'human' or 'rgb_array'.
            cell_size: Size of each cell in pixels.
        """
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.cell_size = cell_size

        self._status_height = 40
        self._pygame_initialized = False
        self._screen = None
        self._clock = None
        self._font = None

    def render(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
        posner_cues: Optional[List[KeyColor]],
        debug_info: Optional[Dict] = None,
        score: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Render the current state.

        Args:
            layout: Current layout state.
            agent_pos: Agent position (row, col).
            held_key: Currently held key color.
            posner_cues: List of Posner cue colors (if any).
            debug_info: Optional debug info with 'cue_index' and 'cue_valid'.
            score: Optional score (collected rewards) to display.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human":
            return self._render_pygame(layout, agent_pos, held_key, posner_cues, debug_info, score)
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array(layout, agent_pos, held_key, posner_cues, debug_info, score)
        else:
            return self._render_ascii(layout, agent_pos, held_key, posner_cues, debug_info, score)

    def _render_ascii(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
        posner_cues: Optional[List[KeyColor]],
        debug_info: Optional[Dict] = None,
        score: Optional[float] = None,
    ) -> None:
        """Render to terminal as ASCII."""
        lines = []

        # Add score if present
        if score is not None:
            lines.append(f"Score: {score:.1f}")

        # Add Posner cue header if present
        if posner_cues is not None:
            cue_strs = []
            for i, cue in enumerate(posner_cues):
                cue_char = "R" if cue == KeyColor.RED else "B"
                # Highlight true cue index in debug mode
                if debug_info is not None and i == debug_info["cue_index"]:
                    cue_char = f"[{cue_char}]"
                cue_strs.append(cue_char)
            cue_line = f"Cues: {' '.join(cue_strs)}"
            # Add validity indicator in debug mode
            if debug_info is not None:
                validity = "✓" if debug_info["cue_valid"] else "✗"
                cue_line += f"  {validity}"
            lines.append(cue_line)
            lines.append("-" * (layout.width + 2))

        for row in range(layout.height):
            line = ""
            for col in range(layout.width):
                if (row, col) == agent_pos:
                    if held_key == KeyColor.RED:
                        line += ASCII_CHARS["agent_with_red_key"]
                    elif held_key == KeyColor.BLUE:
                        line += ASCII_CHARS["agent_with_blue_key"]
                    else:
                        line += ASCII_CHARS["agent"]
                elif layout.is_wall(row, col):
                    line += ASCII_CHARS["wall"]
                else:
                    # Check for objects
                    door = layout.get_door_at(row, col)
                    if door is not None:
                        if door.is_open:
                            line += ASCII_CHARS["door_open"]
                        else:
                            line += ASCII_CHARS["door_closed"]
                        continue

                    reward = layout.get_reward_at(row, col)
                    if reward is not None:
                        line += ASCII_CHARS["reward"]
                        continue

                    # Check for destroyed rewards
                    for r in layout.rewards:
                        if r.position == (row, col) and r.destroyed:
                            line += ASCII_CHARS["reward_destroyed"]
                            break
                    else:
                        key = layout.get_key_at(row, col)
                        if key is not None:
                            if key.color == KeyColor.RED:
                                line += ASCII_CHARS["key_red"]
                            else:
                                line += ASCII_CHARS["key_blue"]
                            continue

                        # Check key pairs
                        key_pair_result = layout.get_key_pair_at(row, col)
                        if key_pair_result is not None:
                            _, key = key_pair_result
                            if key.color == KeyColor.RED:
                                line += ASCII_CHARS["key_red"]
                            else:
                                line += ASCII_CHARS["key_blue"]
                            continue

                        line += ASCII_CHARS["empty"]

            lines.append(line)

        print("\n".join(lines))
        print()
        return None

    def _render_pygame(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
        posner_cues: Optional[List[KeyColor]],
        debug_info: Optional[Dict] = None,
        score: Optional[float] = None,
    ) -> None:
        """Render using pygame."""
        try:
            import pygame
        except ImportError:
            print("pygame not installed. Install with: pip install pygame")
            return self._render_ascii(layout, agent_pos, held_key, posner_cues, debug_info, score)

        if not self._pygame_initialized:
            pygame.init()
            pygame.display.init()

            # Extra height for status bar
            status_height = 40
            self._screen = pygame.display.set_mode((
                self.width * self.cell_size,
                self.height * self.cell_size + status_height,
            ))
            pygame.display.set_caption("GridWorld Environment")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 16)
            self._pygame_initialized = True
            self._status_height = status_height

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        self._draw_frame(layout, agent_pos, held_key, posner_cues, debug_info, score)
        pygame.display.flip()
        self._clock.tick(10)
        return None

    def _render_rgb_array(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
        posner_cues: Optional[List[KeyColor]],
        debug_info: Optional[Dict] = None,
        score: Optional[float] = None,
    ) -> np.ndarray:
        """Render to RGB numpy array."""
        try:
            import pygame
        except ImportError:
            # Fallback to simple numpy rendering
            return self._simple_rgb_render(layout, agent_pos, held_key)

        if not self._pygame_initialized:
            pygame.init()
            status_height = 40
            self._screen = pygame.Surface((
                self.width * self.cell_size,
                self.height * self.cell_size + status_height,
            ))
            self._font = pygame.font.SysFont("monospace", 16)
            self._pygame_initialized = True
            self._status_height = status_height

        self._draw_frame(layout, agent_pos, held_key, posner_cues, debug_info, score)
        return np.transpose(
            pygame.surfarray.array3d(self._screen),
            axes=(1, 0, 2)
        )

    def _draw_frame(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
        posner_cues: Optional[List[KeyColor]],
        debug_info: Optional[Dict] = None,
        score: Optional[float] = None,
    ) -> None:
        """Draw a single frame using pygame."""
        import pygame

        self._screen.fill(COLORS["background"])

        # Draw grid
        for row in range(layout.height):
            for col in range(layout.width):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

                if layout.is_wall(row, col):
                    pygame.draw.rect(self._screen, COLORS["wall"], rect)
                else:
                    pygame.draw.rect(self._screen, COLORS["floor"], rect)

                # Draw grid lines
                pygame.draw.rect(self._screen, COLORS["background"], rect, 1)

        # Draw doors
        for door in layout.doors:
            row, col = door.position
            x = col * self.cell_size
            y = row * self.cell_size
            rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

            if door.is_open:
                pygame.draw.rect(self._screen, COLORS["door_open"], rect)
            else:
                pygame.draw.rect(self._screen, COLORS["door_closed"], rect)

        # Draw rewards
        for reward in layout.rewards:
            row, col = reward.position
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 3

            if reward.destroyed:
                pygame.draw.circle(
                    self._screen, COLORS["reward_destroyed"], (x, y), radius
                )
                # Draw X
                pygame.draw.line(
                    self._screen, (200, 50, 50),
                    (x - radius // 2, y - radius // 2),
                    (x + radius // 2, y + radius // 2), 2
                )
                pygame.draw.line(
                    self._screen, (200, 50, 50),
                    (x + radius // 2, y - radius // 2),
                    (x - radius // 2, y + radius // 2), 2
                )
            elif not reward.collected:
                pygame.draw.circle(
                    self._screen, COLORS["reward"], (x, y), radius
                )

        # Draw keys
        for key in layout.keys:
            if not key.collected:
                self._draw_key(key.position, key.color)

        for key_pair in layout.key_pairs:
            for key in key_pair.get_available_keys():
                self._draw_key(key.position, key.color)

        # Draw agent
        row, col = agent_pos
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        pygame.draw.circle(self._screen, COLORS["agent"], (x, y), radius)

        # Draw held key indicator on agent
        if held_key is not None:
            color = COLORS["key_red"] if held_key == KeyColor.RED else COLORS["key_blue"]
            pygame.draw.circle(self._screen, color, (x, y), radius // 2)

        # Draw status bar
        status_y = self.height * self.cell_size
        status_rect = pygame.Rect(0, status_y, self.width * self.cell_size, self._status_height)
        pygame.draw.rect(self._screen, (30, 30, 30), status_rect)

        # Posner cues
        if posner_cues is not None:
            x_offset = 10
            label_surf = self._font.render("CUES: ", True, COLORS["text"])
            self._screen.blit(label_surf, (x_offset, status_y + 10))
            x_offset += label_surf.get_width()

            for i, cue in enumerate(posner_cues):
                cue_color = (
                    COLORS["posner_cue_red"]
                    if cue == KeyColor.RED
                    else COLORS["posner_cue_blue"]
                )
                cue_char = "R" if cue == KeyColor.RED else "B"
                # Highlight true cue index in debug mode
                if debug_info is not None and i == debug_info["cue_index"]:
                    cue_text = f"[{cue_char}]"
                else:
                    cue_text = f" {cue_char} "
                text_surf = self._font.render(cue_text, True, cue_color)
                self._screen.blit(text_surf, (x_offset, status_y + 10))
                x_offset += text_surf.get_width()

            # Add validity indicator in debug mode
            if debug_info is not None:
                validity = "✓" if debug_info["cue_valid"] else "✗"
                validity_color = (100, 255, 100) if debug_info["cue_valid"] else (255, 100, 100)
                validity_surf = self._font.render(f" {validity}", True, validity_color)
                self._screen.blit(validity_surf, (x_offset, status_y + 10))

        # Score display (right side of status bar)
        if score is not None:
            score_text = f"Score: {score:.1f}"
            score_surf = self._font.render(score_text, True, COLORS["text"])
            score_x = self.width * self.cell_size - score_surf.get_width() - 10
            self._screen.blit(score_surf, (score_x, status_y + 10))

    def _draw_key(self, position: Tuple[int, int], color: KeyColor) -> None:
        """Draw a key at the given position."""
        import pygame

        row, col = position
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2

        key_color = COLORS["key_red"] if color == KeyColor.RED else COLORS["key_blue"]

        # Draw key shape (simple rectangle + circle)
        pygame.draw.rect(
            self._screen, key_color,
            (x - 4, y - 8, 8, 16)
        )
        pygame.draw.circle(self._screen, key_color, (x, y - 8), 6)
        pygame.draw.circle(self._screen, COLORS["floor"], (x, y - 8), 3)

    def _simple_rgb_render(
        self,
        layout: Layout,
        agent_pos: Tuple[int, int],
        held_key: Optional[KeyColor],
    ) -> np.ndarray:
        """Simple RGB rendering without pygame."""
        cs = self.cell_size
        status_h = self._status_height
        grid_h = layout.height * cs
        grid_w = layout.width * cs
        img = np.zeros((grid_h + status_h, grid_w, 3), dtype=np.uint8)

        for row in range(layout.height):
            for col in range(layout.width):
                y1, y2 = row * cs, (row + 1) * cs
                x1, x2 = col * cs, (col + 1) * cs

                if layout.is_wall(row, col):
                    img[y1:y2, x1:x2] = COLORS["wall"]
                else:
                    img[y1:y2, x1:x2] = COLORS["floor"]

        # Draw agent
        row, col = agent_pos
        margin = cs // 4
        y1, y2 = row * cs + margin, (row + 1) * cs - margin
        x1, x2 = col * cs + margin, (col + 1) * cs - margin
        img[y1:y2, x1:x2] = COLORS["agent"]

        # Status bar background
        img[grid_h:, :] = (30, 30, 30)

        return img

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._pygame_initialized:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False
            self._screen = None
