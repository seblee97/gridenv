#!/usr/bin/env python
"""Render oracle trajectories from a distribution dataset as annotated videos.

Works with datasets produced by ``generate_distribution_dataset.py``.
Reconstructs the correct layout variant for each trajectory using the saved
``variants.npz`` file, then replays the stored actions through the environment.

Usage
-----
    python scripts/render_distribution_trajectories.py \
        --dataset datasets/oracle_distribution/directional \
        --n 5 --out-dir datasets/oracle_distribution/videos

    # Render 3 trajectories starting from index 100
    python scripts/render_distribution_trajectories.py \
        --dataset datasets/oracle_distribution/directional \
        --n 3 --offset 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio.v3 as iio

from gridworld_env.env_distribution import LayoutDistribution, apply_variant, LayoutVariant
from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.oracle_policy import PHASE_NAMES, MATERIAL_NAMES
from gridworld_env.procgen import generate_world_grid


ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "USE_KEY", "COLLECT_KEY", "ENGAGE", "FORGE_KEY"]

PHASE_COLORS = {
    "get_key":          (60,  180, 75),
    "make_key":         (245, 130, 48),
    "get_key_location": (145, 30,  180),
    "collect_reward":   (255, 215, 0),
    "goto_next_room":   (70,  130, 180),
}

MATERIAL_COLORS = {
    "none":     (128, 128, 128),
    "diamond":  (0,   210, 210),
    "ruby":     (210, 50,  50),
    "sapphire": (50,  80,  210),
}


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    step: int,
    action: int,
    phase: int,
    material: int,
    room: int,
    variant_id: int,
    keys_held: list,
    mats_held: list,
    safes_opened: int,
    safes_total: int,
    score: float,
    font,
) -> np.ndarray:
    img = Image.fromarray(frame)

    phase_name  = PHASE_NAMES[phase]
    mat_name    = MATERIAL_NAMES[material]
    action_name = ACTION_NAMES[action]
    phase_col   = PHASE_COLORS.get(phase_name, (200, 200, 200))
    mat_col     = MATERIAL_COLORS.get(mat_name, (128, 128, 128))

    keys_str  = ", ".join(sorted(keys_held)) if keys_held else "empty"
    mats_str  = ", ".join(mats_held)         if mats_held else "empty"

    x0, y0 = 8, 8
    line_h = 22

    lines = [
        # Oracle annotations
        (f"step {step:>3d}  (variant {variant_id})", (255, 255, 255)),
        (f"action:   {action_name}",                  (255, 255, 255)),
        (f"phase:    {phase_name}",                   phase_col),
        (f"material: {mat_name}",                     mat_col),
        (f"room:     {room}",                         (255, 255, 255)),
        # Game state
        (f"keys:     {keys_str}",                     (200, 200, 100)),
        (f"mats:     {mats_str}",                     (120, 220, 200)),
        (f"safes:    {safes_opened}/{safes_total}",   (80,  220, 120)),
        (f"score:    {score:.1f}",                    (80,  220, 120)),
    ]

    overlay       = img.copy()
    backdrop_draw = ImageDraw.Draw(overlay)
    backdrop_draw.rectangle(
        [x0 - 4, y0 - 4, x0 + 320, y0 + line_h * len(lines) + 4],
        fill=(0, 0, 0),
    )
    img  = Image.blend(img, overlay, alpha=0.55)
    draw = ImageDraw.Draw(img)

    for i, (text, color) in enumerate(lines):
        draw.text((x0, y0 + i * line_h), text, fill=color, font=font)

    return np.array(img)


# ---------------------------------------------------------------------------
# Trajectory replay
# ---------------------------------------------------------------------------

def render_trajectory(
    env: ModularMazeEnv,
    traj_seed: int,
    start_pos: tuple,
    actions: np.ndarray,
    phases: np.ndarray,
    materials: np.ndarray,
    rooms: np.ndarray,
    variant_id: int,
    font,
) -> list:
    env.reset(seed=traj_seed)
    env._agent_pos = tuple(start_pos)
    env._last_valid_room = env._cell_to_room.get(tuple(start_pos), 0)

    n_safes = len(env._layout.safes)
    score   = 0.0
    frames  = []

    for t in range(len(actions)):
        # Read game state before stepping
        keys_held    = [k.id for k in env._layout.keys if k.collected]
        mats_held    = list(env._material_inventory)
        safes_opened = sum(1 for s in env._layout.safes if s.opened)

        raw   = env.render()
        frame = annotate_frame(
            raw, t, int(actions[t]), int(phases[t]),
            int(materials[t]), int(rooms[t]), variant_id,
            keys_held, mats_held, safes_opened, n_safes, score,
            font,
        )
        frames.append(frame)

        _, reward, _, _, _ = env.step(int(actions[t]))
        score += reward

    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Render distribution-dataset trajectories as annotated videos"
    )
    p.add_argument("--dataset", type=str, required=True,
                   help="Path to dataset dir (e.g. datasets/oracle_distribution/directional)")
    p.add_argument("--n",       type=int, default=5,
                   help="Number of trajectories to render")
    p.add_argument("--offset",  type=int, default=0,
                   help="First trajectory index to render")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <dataset>/videos)")
    p.add_argument("--fps",     type=int, default=6)
    p.add_argument("--random",  action="store_true",
                   help="Sample n trajectories at random instead of sequentially")
    p.add_argument("--random-seed", type=int, default=0,
                   help="Seed for random trajectory selection (default: 0)")
    args = p.parse_args()

    ds_path = Path(args.dataset)
    out_dir = Path(args.out_dir) if args.out_dir else ds_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load metadata + labels ------------------------------------------
    with open(ds_path / "metadata.json") as f:
        meta = json.load(f)

    npz             = np.load(ds_path / "labels.npz")
    actions         = npz["actions"]
    phases          = npz["phase_labels"]
    materials       = npz["material_labels"]
    rooms           = npz["room_labels"]
    start_positions = npz["start_positions"]
    completion      = npz["completion_steps"]
    variant_ids     = npz["variant_ids"]

    n_trajs_per_variant = meta["n_trajs_per_variant"]
    layout_seed         = meta["layout_seed"]
    distribution_seed   = meta["distribution_seed"]

    # --- Reconstruct base layout + all variants --------------------------
    base_layout = generate_world_grid(
        n_rooms=meta["n_rooms"],
        room_h=meta["room_h"],
        room_w=meta["room_w"],
        seed=layout_seed,
    )
    variant_data = np.load(ds_path / "variants.npz")
    n_variants   = variant_data["door_positions"].shape[0]

    def get_variant_layout(vid: int):
        variant = LayoutVariant(
            key_positions      = variant_data["key_positions"][vid],
            safe_positions     = variant_data["safe_positions"][vid],
            door_positions     = variant_data["door_positions"][vid],
            material_positions = variant_data["material_positions"][vid],
            npc_positions      = variant_data["npc_positions"][vid],
        )
        return apply_variant(base_layout, variant)

    # Cache envs by variant id to avoid rebuilding the same env repeatedly
    env_cache: dict[int, ModularMazeEnv] = {}

    def get_env(vid: int) -> ModularMazeEnv:
        if vid not in env_cache:
            layout = get_variant_layout(vid)
            env_cache[vid] = ModularMazeEnv(
                layout,
                obs_mode="symbolic",
                render_mode="rgb_array",
                max_steps=meta["seq_len"] + 10,
                terminate_on_all_safes_opened=False,
            )
        return env_cache[vid]

    # --- Font ------------------------------------------------------------
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()

    # --- Select trajectory indices ---------------------------------------
    n_total = actions.shape[0]
    if args.random:
        rng_sel = np.random.default_rng(args.random_seed)
        indices = rng_sel.choice(n_total, size=args.n, replace=False).tolist()
    else:
        indices = list(range(args.offset, args.offset + args.n))

    # --- Render ----------------------------------------------------------
    for global_i in indices:
        vid               = int(variant_ids[global_i])
        traj_within       = global_i - vid * n_trajs_per_variant
        traj_seed         = vid * 1_000_000 + layout_seed + 1000 + traj_within
        comp_str          = f"done@{completion[global_i]}" if completion[global_i] > 0 else "incomplete"

        print(f"Rendering traj {global_i}  variant={vid}  ({comp_str})...")

        env    = get_env(vid)
        frames = render_trajectory(
            env, traj_seed, start_positions[global_i],
            actions[global_i], phases[global_i],
            materials[global_i], rooms[global_i],
            vid, font,
        )

        out_path = out_dir / f"traj_{global_i:05d}_v{vid:03d}.mp4"
        iio.imwrite(str(out_path), frames, fps=args.fps)
        print(f"  → {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
