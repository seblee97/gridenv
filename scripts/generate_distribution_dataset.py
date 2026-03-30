#!/usr/bin/env python
"""Generate oracle trajectory datasets from a distribution of layout variants.

Each variant shares the same room structure (walls, topology, room types) as a
base layout, but all interactive elements (keys, safes, doors, materials, NPCs)
are randomly repositioned within their respective rooms / wall segments.

Produces one sub-directory per noise type (directional / uniform), each
containing:

- ``labels.npz``    — actions and labels for every trajectory
- ``variants.npz``  — compact position records for all layout variants
- ``metadata.json`` — generation parameters and completion statistics

Pixel observations are NOT stored; use ``replay.py`` + the saved variant
positions to render them on demand.

Usage
-----
    # Pilot: check completion rates (no files written)
    python scripts/generate_distribution_dataset.py --pilot --n-variants 10 \\
        --n-trajs-per-variant 50 --seed 0

    # Full run
    python scripts/generate_distribution_dataset.py --seed 0 \\
        --n-variants 50 --n-trajs-per-variant 600 \\
        --out-dir datasets/oracle_distribution
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from gridworld_env.env_distribution import LayoutDistribution
from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.modular_layout import ModularLayout
from gridworld_env.oracle_policy import (
    MATERIAL_NAMES,
    MATERIAL_TO_ID,
    PHASE_NAMES,
    OraclePolicy,
    apply_directional_noise,
    apply_uniform_noise,
)
from gridworld_env.procgen import generate_world_grid


# ---------------------------------------------------------------------------
# Trajectory helpers (identical logic to generate_oracle_dataset.py)
# ---------------------------------------------------------------------------

def random_start_position(
    env: ModularMazeEnv,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """Pick a random floor cell not occupied by any object."""
    layout = env._layout
    cell_to_room = env._cell_to_room

    occupied = set()
    for s in layout.safes:
        occupied.add(s.position)
    for k in layout.keys:
        if not k.collected:
            occupied.add(k.position)
    for m in layout.materials:
        if not m.collected:
            occupied.add(m.position)
    for n in layout.npcs:
        occupied.add(n.position)
    for d in layout.doors:
        occupied.add(d.position)

    free = [pos for pos in cell_to_room if pos not in occupied]
    return free[int(rng.integers(len(free)))]


def generate_trajectory(
    env: ModularMazeEnv,
    oracle: OraclePolicy,
    seq_len: int,
    noise_fn,
    epsilon: float,
    traj_seed: int,
) -> dict:
    """Run one trajectory and return compact arrays (no pixels).

    Returns dict with keys:
        actions           (T,) int8
        phase_labels      (T,) int8
        material_labels   (T,) int8
        room_labels       (T,) int8
        start_pos         (2,) int16
        all_safes_opened  bool
        steps_to_complete int or -1
    """
    rng = np.random.default_rng(traj_seed)

    env.reset(seed=traj_seed)
    start_pos = random_start_position(env, rng)
    env._agent_pos = start_pos
    env._last_valid_room = env._cell_to_room.get(start_pos, 0)

    oracle.reset(agent_pos=start_pos, rng=rng)

    actions        = np.zeros(seq_len, dtype=np.int8)
    phase_labels   = np.zeros(seq_len, dtype=np.int8)
    material_labels = np.zeros(seq_len, dtype=np.int8)
    room_labels    = np.zeros(seq_len, dtype=np.int8)

    steps_to_complete = -1

    for t in range(seq_len):
        room = env._cell_to_room.get(env._agent_pos, env._last_valid_room)
        room_labels[t] = room

        action, phase, mat_id = oracle.get_action_and_labels()
        noisy_action = noise_fn(action, epsilon, rng)

        actions[t]         = noisy_action
        phase_labels[t]    = phase
        material_labels[t] = mat_id

        env.step(int(noisy_action))
        oracle.notify_step()

        if steps_to_complete < 0 and all(s.opened for s in env._layout.safes):
            steps_to_complete = t + 1

    return {
        "actions":           actions,
        "phase_labels":      phase_labels,
        "material_labels":   material_labels,
        "room_labels":       room_labels,
        "start_pos":         np.array(start_pos, dtype=np.int16),
        "all_safes_opened":  steps_to_complete > 0,
        "steps_to_complete": steps_to_complete,
    }


# ---------------------------------------------------------------------------
# Per-variant trajectory generation
# ---------------------------------------------------------------------------

def generate_variant_trajectories(
    variant_layout: ModularLayout,
    variant_idx: int,
    n_trajs: int,
    seq_len: int,
    layout_seed: int,
    noise_fn,
    epsilon: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, int
]:
    """Generate *n_trajs* trajectories for one layout variant.

    Returns
    -------
    actions, phases, materials, rooms : (n_trajs, seq_len) int8
    start_positions                   : (n_trajs, 2) int16
    completion_steps                  : (n_trajs,) int32
    n_completed                       : int
    """
    env = ModularMazeEnv(
        variant_layout,
        obs_mode="symbolic",
        max_steps=seq_len + 10,
        terminate_on_all_safes_opened=False,
    )
    oracle = OraclePolicy(env)

    all_actions   = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_phases    = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_materials = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_rooms     = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_start_pos = np.zeros((n_trajs, 2), dtype=np.int16)
    comp_steps    = np.full(n_trajs, -1, dtype=np.int32)

    n_completed = 0
    # Seed space: variant_idx * 10^6 + layout_seed + 1000 + traj_idx
    # (guarantees distinct seeds across variants and trajectories)
    seed_base = variant_idx * 1_000_000 + layout_seed + 1000

    for i in range(n_trajs):
        result = generate_trajectory(
            env, oracle, seq_len, noise_fn, epsilon,
            traj_seed=seed_base + i,
        )
        all_actions[i]   = result["actions"]
        all_phases[i]    = result["phase_labels"]
        all_materials[i] = result["material_labels"]
        all_rooms[i]     = result["room_labels"]
        all_start_pos[i] = result["start_pos"]
        comp_steps[i]    = result["steps_to_complete"]
        if result["all_safes_opened"]:
            n_completed += 1

    return all_actions, all_phases, all_materials, all_rooms, all_start_pos, comp_steps, n_completed


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_variants: int,
    n_trajs_per_variant: int,
    seq_len: int,
    n_rooms: int,
    room_h: int,
    room_w: int,
    layout_seed: int,
    distribution_seed: int,
    epsilon: float,
    noise_type: str,
    out_dir: Path | None,
    verbose: bool = True,
) -> dict:
    """Generate trajectories from a distribution of layout variants.

    Parameters
    ----------
    n_variants:
        Number of distinct layout variants to generate.
    n_trajs_per_variant:
        Trajectories generated per variant.
    seq_len:
        Timesteps per trajectory.
    n_rooms, room_h, room_w:
        Base layout parameters (passed to ``generate_world_grid``).
    layout_seed:
        Seed for the base layout (room structure).
    distribution_seed:
        Seed for randomizing object positions across variants.
    epsilon:
        Action-noise probability.
    noise_type:
        ``"directional"`` or ``"uniform"``.
    out_dir:
        If given, save ``labels.npz``, ``variants.npz``, ``metadata.json``.
    verbose:
        Print progress.
    """
    noise_fn = apply_directional_noise if noise_type == "directional" else apply_uniform_noise
    n_total  = n_variants * n_trajs_per_variant

    # --- Build base layout and distribution --------------------------------
    base_layout = generate_world_grid(
        n_rooms=n_rooms, room_h=room_h, room_w=room_w,
        distractor=False, seed=layout_seed,
    )
    dist = LayoutDistribution(base_layout, n_variants=n_variants, seed=distribution_seed)

    # --- Storage -----------------------------------------------------------
    all_actions    = np.zeros((n_total, seq_len), dtype=np.int8)
    all_phases     = np.zeros((n_total, seq_len), dtype=np.int8)
    all_materials  = np.zeros((n_total, seq_len), dtype=np.int8)
    all_rooms      = np.zeros((n_total, seq_len), dtype=np.int8)
    all_start_pos  = np.zeros((n_total, 2), dtype=np.int16)
    comp_steps     = np.full(n_total, -1, dtype=np.int32)
    variant_ids    = np.zeros(n_total, dtype=np.int32)

    total_completed = 0
    t0 = time.time()

    for v in range(n_variants):
        variant_layout = dist.get_layout(v)
        (
            v_actions, v_phases, v_materials, v_rooms,
            v_start_pos, v_comp, v_done,
        ) = generate_variant_trajectories(
            variant_layout, v, n_trajs_per_variant, seq_len,
            layout_seed, noise_fn, epsilon,
        )

        start = v * n_trajs_per_variant
        end   = start + n_trajs_per_variant
        all_actions[start:end]   = v_actions
        all_phases[start:end]    = v_phases
        all_materials[start:end] = v_materials
        all_rooms[start:end]     = v_rooms
        all_start_pos[start:end] = v_start_pos
        comp_steps[start:end]    = v_comp
        variant_ids[start:end]   = v
        total_completed += v_done

        if verbose:
            elapsed = time.time() - t0
            rate = (end) / elapsed
            pct = total_completed / end * 100
            done_s = comp_steps[:end]
            done_s = done_s[done_s > 0]
            mean_s = f"{done_s.mean():.0f}" if len(done_s) > 0 else "n/a"
            print(
                f"  variant {v+1:>3}/{n_variants}  "
                f"[{end:>6}/{n_total}]  {rate:.1f} traj/s  "
                f"completion: {pct:.1f}%  mean steps: {mean_s}"
            )

    stats = {
        "n_variants":            n_variants,
        "n_trajs_per_variant":   n_trajs_per_variant,
        "n_total_trajs":         n_total,
        "seq_len":               seq_len,
        "n_rooms":               n_rooms,
        "room_h":                room_h,
        "room_w":                room_w,
        "layout_seed":           layout_seed,
        "distribution_seed":     distribution_seed,
        "epsilon":               epsilon,
        "noise_type":            noise_type,
        "completion_rate":       total_completed / n_total,
        "mean_steps_to_complete": (
            float(comp_steps[comp_steps > 0].mean())
            if total_completed > 0 else -1
        ),
        "median_steps_to_complete": (
            float(np.median(comp_steps[comp_steps > 0]))
            if total_completed > 0 else -1
        ),
        "phase_names":    PHASE_NAMES,
        "material_names": MATERIAL_NAMES,
    }

    if verbose:
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        if total_completed > 0:
            print(f"  Mean steps:   {stats['mean_steps_to_complete']:.1f}")
            print(f"  Median steps: {stats['median_steps_to_complete']:.1f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / "labels.npz",
            actions          = all_actions,
            phase_labels     = all_phases,
            material_labels  = all_materials,
            room_labels      = all_rooms,
            start_positions  = all_start_pos,
            completion_steps = comp_steps,
            variant_ids      = variant_ids,
        )
        dist.save_variants(out_dir / "variants.npz")
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(stats, f, indent=2)

        if verbose:
            print(f"  Saved to {out_dir}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate oracle dataset from a distribution of layout variants"
    )
    p.add_argument("--seed",                 type=int,   default=0,
                   help="Base layout seed (room structure)")
    p.add_argument("--distribution-seed",    type=int,   default=42,
                   help="Seed for randomizing object positions across variants")
    p.add_argument("--n-variants",           type=int,   default=50,
                   help="Number of distinct layout variants")
    p.add_argument("--n-trajs-per-variant",  type=int,   default=600,
                   help="Trajectories generated per variant")
    p.add_argument("--seq-len",              type=int,   default=150)
    p.add_argument("--n-rooms",              type=int,   default=4)
    p.add_argument("--room-h",               type=int,   default=9)
    p.add_argument("--room-w",               type=int,   default=11)
    p.add_argument("--epsilon",              type=float, default=0.1)
    p.add_argument("--out-dir",              type=str,   default="datasets/oracle_distribution")
    p.add_argument("--pilot", action="store_true",
                   help="Pilot run: skip saving, print stats only")
    args = p.parse_args()

    common = dict(
        n_variants           = args.n_variants,
        n_trajs_per_variant  = args.n_trajs_per_variant,
        seq_len              = args.seq_len,
        n_rooms              = args.n_rooms,
        room_h               = args.room_h,
        room_w               = args.room_w,
        layout_seed          = args.seed,
        distribution_seed    = args.distribution_seed,
        epsilon              = args.epsilon,
    )

    if args.pilot:
        print("=== PILOT RUN (directional noise) ===")
        generate_dataset(**common, noise_type="directional", out_dir=None)
        print("\n=== PILOT RUN (uniform noise) ===")
        generate_dataset(**common, noise_type="uniform", out_dir=None)
        return

    base = Path(args.out_dir)
    for noise_type in ("directional", "uniform"):
        print(f"\n=== {noise_type} noise ===")
        generate_dataset(
            **common,
            noise_type=noise_type,
            out_dir=base / noise_type,
        )


if __name__ == "__main__":
    main()
