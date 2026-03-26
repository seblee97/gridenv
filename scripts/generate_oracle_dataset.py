#!/usr/bin/env python
"""Generate oracle trajectory datasets for ModularMazeEnv.

Produces one dataset per noise variant (directional / uniform), each containing
actions, labels, and start positions.  Pixel observations are NOT stored here —
they are rendered on demand by the RPL ``MazeOracleDataset`` class and cached
to disk on first access.

Usage
-----
    # Pilot run (check completion statistics)
    python scripts/generate_oracle_dataset.py --pilot --n-trajs 200 --seed 0

    # Full generation (both noise types)
    python scripts/generate_oracle_dataset.py --seed 0 --out-dir datasets/oracle
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.procgen import generate_world_grid
from gridworld_env.oracle_policy import (
    MATERIAL_NAMES,
    MATERIAL_TO_ID,
    PHASE_NAMES,
    OraclePolicy,
    apply_directional_noise,
    apply_uniform_noise,
)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def random_start_position(
    env: ModularMazeEnv, rng: np.random.Generator,
) -> Tuple[int, int]:
    """Pick a random floor cell (any room) that is not occupied by an object."""
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
):
    """Run one trajectory and return compact arrays (no pixels).

    Returns dict with keys:
        actions:           (T,)   int8
        phase_labels:      (T,)   int8
        material_labels:   (T,)   int8
        room_labels:       (T,)   int8
        start_pos:         (2,)   int16
        all_safes_opened:  bool
        steps_to_complete: int or -1
    """
    rng = np.random.default_rng(traj_seed)

    # Reset env and override start position
    env.reset(seed=traj_seed)
    start_pos = random_start_position(env, rng)
    env._agent_pos = start_pos
    env._last_valid_room = env._cell_to_room.get(start_pos, 0)

    oracle.reset(agent_pos=start_pos, rng=rng)

    # Pre-allocate
    actions = np.zeros(seq_len, dtype=np.int8)
    phase_labels = np.zeros(seq_len, dtype=np.int8)
    material_labels = np.zeros(seq_len, dtype=np.int8)
    room_labels = np.zeros(seq_len, dtype=np.int8)

    steps_to_complete = -1

    for t in range(seq_len):
        room = env._cell_to_room.get(env._agent_pos, env._last_valid_room)
        room_labels[t] = room

        action, phase, mat_id = oracle.get_action_and_labels()
        noisy_action = noise_fn(action, epsilon, rng)

        actions[t] = noisy_action
        phase_labels[t] = phase
        material_labels[t] = mat_id

        env.step(int(noisy_action))
        oracle.notify_step()

        if steps_to_complete < 0 and all(s.opened for s in env._layout.safes):
            steps_to_complete = t + 1

    return {
        "actions": actions,
        "phase_labels": phase_labels,
        "material_labels": material_labels,
        "room_labels": room_labels,
        "start_pos": np.array(start_pos, dtype=np.int16),
        "all_safes_opened": steps_to_complete > 0,
        "steps_to_complete": steps_to_complete,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_trajs: int,
    seq_len: int,
    n_rooms: int,
    room_h: int,
    room_w: int,
    layout_seed: int,
    epsilon: float,
    noise_type: str,
    out_dir: Path | None,
    verbose: bool = True,
):
    """Generate a batch of trajectories and optionally save to disk."""
    layout = generate_world_grid(
        n_rooms=n_rooms, room_h=room_h, room_w=room_w,
        distractor=False, seed=layout_seed,
    )
    env = ModularMazeEnv(
        layout,
        obs_mode="symbolic",
        max_steps=seq_len + 10,
        terminate_on_all_safes_opened=False,
    )
    oracle = OraclePolicy(env)

    noise_fn = apply_directional_noise if noise_type == "directional" else apply_uniform_noise

    # Storage
    all_actions = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_phases = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_materials = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_rooms = np.zeros((n_trajs, seq_len), dtype=np.int8)
    all_start_pos = np.zeros((n_trajs, 2), dtype=np.int16)
    completion_steps = np.full(n_trajs, -1, dtype=np.int32)

    t0 = time.time()
    completed = 0

    for i in range(n_trajs):
        traj_seed = layout_seed + 1000 + i
        result = generate_trajectory(
            env, oracle, seq_len, noise_fn, epsilon, traj_seed,
        )
        all_actions[i] = result["actions"]
        all_phases[i] = result["phase_labels"]
        all_materials[i] = result["material_labels"]
        all_rooms[i] = result["room_labels"]
        all_start_pos[i] = result["start_pos"]
        completion_steps[i] = result["steps_to_complete"]

        if result["all_safes_opened"]:
            completed += 1

        if verbose and (i + 1) % max(1, n_trajs // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            pct_complete = completed / (i + 1) * 100
            done_steps = completion_steps[:i+1]
            done_steps = done_steps[done_steps > 0]
            mean_s = f"{done_steps.mean():.0f}" if len(done_steps) > 0 else "n/a"
            print(
                f"  [{i+1:>6}/{n_trajs}]  {rate:.1f} traj/s  "
                f"completion: {pct_complete:.1f}%  mean steps: {mean_s}"
            )

    stats = {
        "n_trajs": n_trajs,
        "seq_len": seq_len,
        "n_rooms": n_rooms,
        "room_h": room_h,
        "room_w": room_w,
        "layout_seed": layout_seed,
        "epsilon": epsilon,
        "noise_type": noise_type,
        "completion_rate": completed / n_trajs,
        "mean_steps_to_complete": (
            float(completion_steps[completion_steps > 0].mean())
            if completed > 0 else -1
        ),
        "median_steps_to_complete": (
            float(np.median(completion_steps[completion_steps > 0]))
            if completed > 0 else -1
        ),
        "phase_names": PHASE_NAMES,
        "material_names": MATERIAL_NAMES,
    }

    if verbose:
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")
        print(f"  Completion rate: {stats['completion_rate']:.1%}")
        if completed > 0:
            print(f"  Mean steps to complete: {stats['mean_steps_to_complete']:.1f}")
            print(f"  Median steps to complete: {stats['median_steps_to_complete']:.1f}")

    # Save
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "labels.npz",
            actions=all_actions,
            phase_labels=all_phases,
            material_labels=all_materials,
            room_labels=all_rooms,
            start_positions=all_start_pos,
            completion_steps=completion_steps,
        )
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(stats, f, indent=2)
        if verbose:
            print(f"  Saved to {out_dir}")

    return stats, all_actions, all_phases, all_materials, all_rooms, completion_steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate oracle trajectory dataset")
    p.add_argument("--seed", type=int, default=0, help="Layout seed")
    p.add_argument("--n-trajs", type=int, default=30000)
    p.add_argument("--seq-len", type=int, default=150)
    p.add_argument("--n-rooms", type=int, default=4)
    p.add_argument("--room-h", type=int, default=9)
    p.add_argument("--room-w", type=int, default=11)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, default="datasets/oracle")
    p.add_argument("--pilot", action="store_true",
                   help="Pilot run: skip saving, print stats only")
    args = p.parse_args()

    if args.pilot:
        print("=== PILOT RUN (directional noise) ===")
        generate_dataset(
            n_trajs=args.n_trajs, seq_len=args.seq_len,
            n_rooms=args.n_rooms, room_h=args.room_h, room_w=args.room_w,
            layout_seed=args.seed, epsilon=args.epsilon,
            noise_type="directional", out_dir=None,
        )
        print("\n=== PILOT RUN (uniform noise) ===")
        generate_dataset(
            n_trajs=args.n_trajs, seq_len=args.seq_len,
            n_rooms=args.n_rooms, room_h=args.room_h, room_w=args.room_w,
            layout_seed=args.seed, epsilon=args.epsilon,
            noise_type="uniform", out_dir=None,
        )
        return

    base = Path(args.out_dir)
    for noise_type in ("directional", "uniform"):
        print(f"\n=== {noise_type} noise ===")
        generate_dataset(
            n_trajs=args.n_trajs, seq_len=args.seq_len,
            n_rooms=args.n_rooms, room_h=args.room_h, room_w=args.room_w,
            layout_seed=args.seed, epsilon=args.epsilon,
            noise_type=noise_type,
            out_dir=base / noise_type,
        )


if __name__ == "__main__":
    main()
