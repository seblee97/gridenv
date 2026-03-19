#!/usr/bin/env python3
"""
Interactive gridworld player.

Supports both GridWorldEnv (.txt / .json layouts) and ModularMazeEnv
(modular .txt layouts and world topology files).  The environment type is
detected automatically from the file.

Usage:
    python play.py <layout_file> [options]

Examples:
    python play.py src/gridworld_env/layouts/simple_room.txt
    python play.py src/gridworld_env/layouts/two_rooms.txt --cell-size 64
    python play.py src/gridworld_env/layouts/world_example/world.txt
    python play.py src/gridworld_env/layouts/simple_room.txt --posner --posner-validity 0.8

Controls (GridWorldEnv):
    Arrow keys / WASD  Move agent
    R                  Reset episode
    ESC / Q            Quit

Controls (ModularMazeEnv):
    Arrow keys / WASD  Move agent
    E                  Use key (open adjacent door / open safe)
    F                  Collect key or material (pick up item on current cell)
    G                  Engage NPC (interact with adjacent NPC)
    Z                  Forge key (consume ore + material to create a key)
    R                  Reset episode
    ESC / Q            Quit

Procedural generation:
    python play.py --procgen --n-rooms 6 --distractor --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# Sidebar width in pixels
SIDEBAR_WIDTH = 210

GRIDWORLD_HELP = [
    "Arrows/WASD: Move",
    "R: Reset   ESC/Q: Quit",
]

MODULAR_HELP = [
    "Arrows/WASD: Move",
    "E: Use key",
    "F: Collect key/material",
    "G: Engage NPC",
    "Z: Forge key",
    "R: Reset   ESC/Q: Quit",
]


def _is_world_file(path: Path) -> bool:
    """Return True if path looks like a modular world topology file."""
    try:
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped == "rooms"
    except OSError:
        pass
    return False


def _is_modular_layout(path: Path) -> bool:
    """Return True if path looks like a ModularMaze layout (not GridWorld)."""
    # World topology files are always modular
    if _is_world_file(path):
        return True
    # Modular single-room layouts use 'a'-'z' for keys, '1'-'9' for safes.
    # GridWorld uses 'r'/'b' for keys.  Heuristic: look for digits 1-9 or
    # letters other than r/b/S/G/D as cell contents.
    modular_chars = set("abcdefghijklmnopqrstuvwxyz") - {"r", "b", "s"}
    safe_chars = set("123456789")
    try:
        with open(path) as f:
            content = f.read()
        return bool(set(content) & (modular_chars | safe_chars))
    except OSError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a gridworld environment interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "layout", nargs="?", default=None,
        help="Path to layout .txt file (or world.txt). Omit when using --procgen.",
    )
    parser.add_argument(
        "--procgen", action="store_true",
        help="Generate a random world instead of loading a file.",
    )
    parser.add_argument(
        "--n-rooms", type=int, default=4, metavar="N",
        help="Number of rooms for procedural generation (default: 4).",
    )
    parser.add_argument(
        "--distractor", action="store_true",
        help="Add irrelevant elements to rooms in procedural generation.",
    )
    parser.add_argument(
        "--procgen-method",
        choices=["bsp", "grid"],
        default="bsp",
        metavar="METHOD",
        help="Procgen algorithm: 'bsp' (default) produces variable-sized rooms;"
             " 'grid' produces equal-sized rooms (required for --partial-obs).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="RNG seed for procedural generation.",
    )
    parser.add_argument(
        "--room-h", type=int, default=9, metavar="N",
        help="Room height in cells including walls, for --procgen-method grid (default: 9).",
    )
    parser.add_argument(
        "--room-w", type=int, default=11, metavar="N",
        help="Room width in cells including walls, for --procgen-method grid (default: 11).",
    )
    parser.add_argument(
        "--posner", action="store_true", help="Enable Posner cueing mode (GridWorldEnv only)"
    )
    parser.add_argument(
        "--posner-validity",
        type=float,
        default=1.0,
        metavar="P",
        help="Posner cue validity probability (default: 1.0)",
    )
    parser.add_argument(
        "--posner-features",
        type=int,
        default=1,
        metavar="N",
        help="Number of Posner cue features (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Max steps per episode (default: unlimited)",
    )
    parser.add_argument(
        "--random-keys",
        action="store_true",
        help="Randomize key colors each episode (GridWorldEnv only)",
    )
    parser.add_argument(
        "--random-correct-key",
        action="store_true",
        help="Randomize which key is correct each episode (GridWorldEnv only)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=48,
        metavar="PX",
        help="Cell size in pixels (default: 48)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        metavar="N",
        help="Display FPS cap (default: 60)",
    )
    parser.add_argument(
        "--step-reward",
        type=float,
        default=-0.01,
        metavar="R",
        help="Reward per step (default: -0.01)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show true Posner cue index and validity (GridWorldEnv only)",
    )
    parser.add_argument(
        "--partial-obs",
        action="store_true",
        help="Partial observability: show only the current room (ModularMazeEnv only)",
    )
    parser.add_argument(
        "--global-map-mode",
        choices=["overlay", "image", "onehot", "beside"],
        default=None,
        metavar="MODE",
        help="Add a global room-level map to the observation: 'overlay' draws it in the"
             " corner of the pixel view, 'beside' places it to the right of the room obs"
             " (requires --partial-obs), 'image' adds a separate map image, 'onehot'"
             " adds a one-hot room vector (ModularMazeEnv only).",
    )
    parser.add_argument(
        "--map-cell-size",
        type=int,
        default=8,
        metavar="PX",
        help="Pixel size of each room cell in the global map (default: 8).",
    )
    return parser.parse_args()


def draw_sidebar(
    screen,
    font,
    font_small,
    info: dict,
    sidebar_x: int,
    height: int,
    help_lines: list,
):
    """Draw the info sidebar on the right side."""
    import pygame

    sidebar_rect = pygame.Rect(sidebar_x, 0, SIDEBAR_WIDTH, height)
    pygame.draw.rect(screen, (25, 25, 35), sidebar_rect)
    pygame.draw.line(screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, height), 2)

    y = 16
    line_h = 22

    def text(label, value=None, color=(200, 200, 220)):
        nonlocal y
        if value is not None:
            label_surf = font_small.render(label, True, (120, 120, 150))
            screen.blit(label_surf, (sidebar_x + 12, y))
            val_surf = font.render(str(value), True, color)
            screen.blit(val_surf, (sidebar_x + 12, y + 14))
            y += line_h + 10
        else:
            surf = font.render(label, True, color)
            screen.blit(surf, (sidebar_x + 12, y))
            y += line_h

    def divider():
        nonlocal y
        pygame.draw.line(
            screen, (50, 50, 70),
            (sidebar_x + 8, y + 4),
            (sidebar_x + SIDEBAR_WIDTH - 8, y + 4),
            1,
        )
        y += 14

    # Title
    title_surf = font.render("GRIDWORLD", True, (100, 180, 255))
    screen.blit(title_surf, (sidebar_x + 12, y))
    y += line_h + 4
    divider()

    # Episode stats
    text("EPISODE", info.get("episode", 1), color=(255, 220, 80))
    text("STEPS", info.get("steps", 0))
    text("SCORE", f"{info.get('score', 0.0):.2f}", color=(80, 220, 120))
    divider()

    # Key / inventory status
    if "inventory" in info:
        # ModularMazeEnv
        inv = info["inventory"]
        inv_text = ", ".join(str(k) for k in inv) if inv else "empty"
        text("KEYS", inv_text, color=(200, 200, 100))
        mats = info.get("materials", [])
        mat_text = ", ".join(str(m) for m in mats) if mats else "empty"
        text("MATERIALS", mat_text, color=(120, 220, 200))
        safes_opened = info.get("safes_opened", 0)
        safes_total = info.get("safes_total", 0)
        text("SAFES", f"{safes_opened}/{safes_total}", color=(80, 220, 120))
    else:
        # GridWorldEnv
        held = info.get("held_key")
        if held == "red":
            key_color = (220, 80, 80)
            key_text = "RED KEY"
        elif held == "blue":
            key_color = (80, 80, 220)
            key_text = "BLUE KEY"
        else:
            key_color = (100, 100, 120)
            key_text = "none"
        text("HOLDING", key_text, color=key_color)
    divider()

    # Status message
    msg = info.get("message", "")
    if msg:
        msg_color = info.get("message_color", (255, 255, 100))
        for line in msg.split("\n"):
            surf = font_small.render(line, True, msg_color)
            screen.blit(surf, (sidebar_x + 12, y))
            y += 18
        y += 4
        divider()

    # Controls (pinned to bottom)
    y = height - (len(help_lines) * 18 + 20)
    pygame.draw.line(
        screen, (50, 50, 70),
        (sidebar_x + 8, y - 8),
        (sidebar_x + SIDEBAR_WIDTH - 8, y - 8),
        1,
    )
    for line in help_lines:
        surf = font_small.render(line, True, (80, 80, 100))
        screen.blit(surf, (sidebar_x + 12, y))
        y += 18


def main():
    args = parse_args()

    try:
        import pygame
    except ImportError:
        print("pygame is required. Install with: pip install pygame")
        sys.exit(1)

    if not args.procgen and args.layout is None:
        print("error: provide a layout file or use --procgen to generate one.")
        sys.exit(1)

    try:
        from gridworld_env import GridWorldEnv
        from gridworld_env.environment import Action as GWAction
        from gridworld_env.modular_maze import ModularMazeEnv, _Action as MMAction
    except ImportError:
        print("gridworld_env not found. Make sure it's installed or run from the project root.")
        sys.exit(1)

    if args.procgen:
        if args.procgen_method == "grid":
            from gridworld_env.procgen import generate_world_grid
            layout_obj = generate_world_grid(
                n_rooms=args.n_rooms,
                distractor=args.distractor,
                seed=args.seed,
                room_h=args.room_h,
                room_w=args.room_w,
            )
        else:
            from gridworld_env.procgen import generate_world
            layout_obj = generate_world(
                n_rooms=args.n_rooms,
                distractor=args.distractor,
                seed=args.seed,
            )
        use_partial = args.partial_obs
        obs_mode = "room_pixels" if use_partial else "pixels"
        env = ModularMazeEnv(
            layout=layout_obj,
            max_steps=args.max_steps,
            step_reward=args.step_reward,
            obs_mode=obs_mode,
            render_mode="rgb_array" if not use_partial else None,
            show_score=False,
            global_map_mode=args.global_map_mode,
            map_cell_size=args.map_cell_size,
        )
        caption = f"GridWorld — procgen {args.n_rooms} rooms"
        is_modular = True
        help_lines = MODULAR_HELP
    else:
        layout_path = Path(args.layout)
        is_modular = _is_modular_layout(layout_path)
        caption = f"GridWorld — {args.layout}"

        if is_modular:
            use_partial = args.partial_obs
            obs_mode = "room_pixels" if use_partial else "pixels"
            env = ModularMazeEnv(
                layout=args.layout,
                max_steps=args.max_steps,
                step_reward=args.step_reward,
                obs_mode=obs_mode,
                render_mode="rgb_array" if not use_partial else None,
                show_score=False,
                global_map_mode=args.global_map_mode,
                map_cell_size=args.map_cell_size,
            )
            help_lines = MODULAR_HELP
        else:
            env = GridWorldEnv(
                layout=args.layout,
                posner_mode=args.posner,
                posner_validity=args.posner_validity,
                posner_num_features=args.posner_features,
                max_steps=args.max_steps,
                random_key_colors=args.random_keys,
                random_correct_key=args.random_correct_key,
                step_reward=args.step_reward,
                obs_mode="pixels",
                render_mode="rgb_array",
                debug_cues=args.debug,
                show_score=False,
            )
            help_lines = GRIDWORLD_HELP

    obs, info = env.reset()
    # In partial-obs mode the obs itself is the room-level pixel frame;
    # otherwise render the full world view.
    use_partial = is_modular and args.partial_obs
    frame = obs if use_partial else env.render()

    # Scale frame to requested cell size (env renders at 32px cells)
    native_h, native_w = frame.shape[:2]
    scale = args.cell_size / 32
    grid_w = int(native_w * scale)
    grid_h = int(native_h * scale)
    total_w = grid_w + SIDEBAR_WIDTH
    total_h = max(grid_h, 200)  # minimum height for sidebar

    pygame.init()
    screen = pygame.display.set_mode((total_w, total_h))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("monospace", 15, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)

    # Build key → action map depending on env type
    if is_modular:
        key_action_map = {
            pygame.K_UP: MMAction.UP,
            pygame.K_w: MMAction.UP,
            pygame.K_DOWN: MMAction.DOWN,
            pygame.K_s: MMAction.DOWN,
            pygame.K_LEFT: MMAction.LEFT,
            pygame.K_a: MMAction.LEFT,
            pygame.K_RIGHT: MMAction.RIGHT,
            pygame.K_d: MMAction.RIGHT,
            pygame.K_e: MMAction.USE_KEY,
            pygame.K_f: MMAction.COLLECT_KEY,
            pygame.K_g: MMAction.ENGAGE,
            pygame.K_z: MMAction.FORGE_KEY,
        }
    else:
        key_action_map = {
            pygame.K_UP: GWAction.UP,
            pygame.K_w: GWAction.UP,
            pygame.K_DOWN: GWAction.DOWN,
            pygame.K_s: GWAction.DOWN,
            pygame.K_LEFT: GWAction.LEFT,
            pygame.K_a: GWAction.LEFT,
            pygame.K_RIGHT: GWAction.RIGHT,
            pygame.K_d: GWAction.RIGHT,
        }

    episode = 1
    score = 0.0
    steps = 0
    message = ""
    message_color = (255, 255, 100)
    message_timer = 0  # frames to show message

    def render_frame(frame: np.ndarray):
        """Blit numpy frame (scaled) to the left part of screen."""
        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        if scale != 1.0:
            surf = pygame.transform.scale(surf, (grid_w, grid_h))
        screen.blit(surf, (0, 0))

    def make_sidebar_info():
        d = {
            "episode": episode,
            "steps": steps,
            "score": score,
            "message": message if message_timer > 0 else "",
            "message_color": message_color,
        }
        if is_modular:
            d["inventory"] = info.get("inventory", [])
            d["materials"] = info.get("materials", [])
            d["safes_opened"] = info.get("safes_opened", 0)
            d["safes_total"] = info.get("safes_total", 0)
            d["score"] = info.get("score", 0.0)
        else:
            d["held_key"] = info.get("held_key")
            d["score"] = score
        return d

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    frame = obs if use_partial else env.render()
                    episode += 1
                    score = 0.0
                    steps = 0
                    message = "Episode reset"
                    message_color = (150, 150, 255)
                    message_timer = 90

                elif event.key in key_action_map:
                    action = key_action_map[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    frame = obs if use_partial else env.render()
                    score += reward
                    steps += 1

                    if terminated:
                        message = "Episode done!\nPress R to restart"
                        message_color = (80, 255, 120)
                        message_timer = 999
                    elif truncated:
                        message = "Time limit!\nPress R to restart"
                        message_color = (255, 160, 80)
                        message_timer = 999
                    elif reward > 0:
                        message = f"+{reward:.2f}"
                        message_color = (80, 255, 120)
                        message_timer = 45
                    elif reward < -0.05:
                        message = f"{reward:.2f}"
                        message_color = (255, 100, 100)
                        message_timer = 30

        # Draw
        render_frame(frame)
        draw_sidebar(screen, font, font_small, make_sidebar_info(), grid_w, total_h, help_lines)
        pygame.display.flip()

        if message_timer > 0:
            message_timer -= 1

        clock.tick(args.fps)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
