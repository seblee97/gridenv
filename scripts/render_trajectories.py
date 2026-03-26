#!/usr/bin/env python
"""Render oracle trajectories as annotated videos.

Overlays action names and phase labels onto the full-colour environment render.

Usage
-----
    python scripts/render_trajectories.py \
        --dataset datasets/oracle/directional \
        --n 5 --out-dir datasets/oracle/videos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio.v3 as iio

from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.procgen import generate_world_grid
from gridworld_env.oracle_policy import PHASE_NAMES, MATERIAL_NAMES

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "USE_KEY", "COLLECT_KEY", "ENGAGE", "FORGE_KEY"]

# Colours per phase for the label badge
PHASE_COLORS = {
    "get_key":          (60, 180, 75),
    "make_key":         (245, 130, 48),
    "get_key_location": (145, 30, 180),
    "collect_reward":   (255, 215, 0),
    "goto_next_room":   (70, 130, 180),
}

MATERIAL_COLORS = {
    "none":      (128, 128, 128),
    "diamond":   (0, 210, 210),
    "ruby":      (210, 50, 50),
    "sapphire":  (50, 80, 210),
}


def annotate_frame(
    frame: np.ndarray,
    step: int,
    action: int,
    phase: int,
    material: int,
    room: int,
    font,
) -> np.ndarray:
    """Draw action / phase / material / room info onto a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    phase_name = PHASE_NAMES[phase]
    mat_name = MATERIAL_NAMES[material]
    action_name = ACTION_NAMES[action]
    phase_col = PHASE_COLORS.get(phase_name, (200, 200, 200))
    mat_col = MATERIAL_COLORS.get(mat_name, (128, 128, 128))

    x0, y0 = 8, 8
    line_h = 22

    # Semi-transparent backdrop
    overlay = img.copy()
    backdrop_draw = ImageDraw.Draw(overlay)
    backdrop_draw.rectangle([x0 - 4, y0 - 4, x0 + 320, y0 + line_h * 5 + 4],
                            fill=(0, 0, 0))
    img = Image.blend(img, overlay, alpha=0.55)
    draw = ImageDraw.Draw(img)

    lines = [
        (f"step {step:>3d}", (255, 255, 255)),
        (f"action:   {action_name}", (255, 255, 255)),
        (f"phase:    {phase_name}", phase_col),
        (f"material: {mat_name}", mat_col),
        (f"room:     {room}", (255, 255, 255)),
    ]
    for i, (text, color) in enumerate(lines):
        draw.text((x0, y0 + i * line_h), text, fill=color, font=font)

    return np.array(img)


def render_trajectory(
    env: ModularMazeEnv,
    traj_seed: int,
    start_pos: tuple,
    actions: np.ndarray,
    phases: np.ndarray,
    materials: np.ndarray,
    rooms: np.ndarray,
    font,
) -> list[np.ndarray]:
    """Replay a trajectory and return annotated frames."""
    env.reset(seed=traj_seed)
    env._agent_pos = tuple(start_pos)
    env._last_valid_room = env._cell_to_room.get(tuple(start_pos), 0)

    frames = []
    for t in range(len(actions)):
        raw = env.render()
        frame = annotate_frame(raw, t, int(actions[t]), int(phases[t]),
                               int(materials[t]), int(rooms[t]), font)
        frames.append(frame)
        env.step(int(actions[t]))

    return frames


def main():
    p = argparse.ArgumentParser(description="Render oracle trajectories as videos")
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to dataset dir (e.g. datasets/oracle/directional)")
    p.add_argument("--n", type=int, default=5, help="Number of trajectories to render")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <dataset>/videos)")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--offset", type=int, default=0,
                   help="First trajectory index to render")
    args = p.parse_args()

    ds_path = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else ds_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata + labels
    with open(ds_path / "metadata.json") as f:
        meta = json.load(f)
    npz = np.load(ds_path / "labels.npz")
    actions = npz["actions"]
    phases = npz["phase_labels"]
    materials = npz["material_labels"]
    rooms = npz["room_labels"]
    start_positions = npz["start_positions"]
    completion = npz["completion_steps"]

    # Create render env
    layout = generate_world_grid(
        n_rooms=meta["n_rooms"], room_h=meta["room_h"], room_w=meta["room_w"],
        seed=meta["layout_seed"],
    )
    env = ModularMazeEnv(
        layout, obs_mode="symbolic", render_mode="rgb_array",
        max_steps=meta["seq_len"] + 10,
        terminate_on_all_safes_opened=False,
    )

    # Try to get a monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i in range(args.offset, args.offset + args.n):
        traj_seed = meta["layout_seed"] + 1000 + i
        comp_str = f"done@{completion[i]}" if completion[i] > 0 else "incomplete"

        print(f"Rendering traj {i} ({comp_str})...")
        frames = render_trajectory(
            env, traj_seed, start_positions[i],
            actions[i], phases[i], materials[i], rooms[i], font,
        )

        out_path = out_dir / f"traj_{i:05d}.mp4"
        iio.imwrite(str(out_path), frames, fps=args.fps)
        print(f"  → {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
