"""Deterministic trajectory replay for pixel rendering.

Given a layout seed, trajectory seed, start position, and action sequence,
replays the trajectory through ``ModularMazeEnv`` and returns pixel
observations at every timestep.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.procgen import generate_world_grid


def create_pixel_env(
    layout_seed: int,
    n_rooms: int = 4,
    room_h: int = 9,
    room_w: int = 11,
    seq_len: int = 150,
) -> ModularMazeEnv:
    """Create a ``ModularMazeEnv`` configured for pixel rendering."""
    layout = generate_world_grid(
        n_rooms=n_rooms, room_h=room_h, room_w=room_w,
        distractor=False, seed=layout_seed,
    )
    return ModularMazeEnv(
        layout,
        obs_mode="room_pixels",
        global_map_mode="image",
        max_steps=seq_len + 10,
        terminate_on_all_safes_opened=False,
    )


def replay_trajectory(
    env: ModularMazeEnv,
    traj_seed: int,
    start_pos: Tuple[int, int],
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay an action sequence and return pixel observations.

    Parameters
    ----------
    env:
        A pixel-mode ``ModularMazeEnv`` (reused across calls).
    traj_seed:
        Seed passed to ``env.reset()`` (controls env RNG for wizard key spawns).
    start_pos:
        ``(row, col)`` agent start position override.
    actions:
        ``(T,)`` int array of actions taken at each step.

    Returns
    -------
    room_pixels:
        ``(T, H, W, 3)`` uint8 — local room observation at each step.
    map_images:
        ``(T, mH, mW, 3)`` uint8 — global map image at each step.
    """
    T = len(actions)
    env.reset(seed=traj_seed)
    env._agent_pos = tuple(start_pos)
    env._last_valid_room = env._cell_to_room.get(tuple(start_pos), 0)

    # Get shapes from first obs
    obs = env._get_observation()
    local_shape = obs["obs"].shape
    map_shape = obs["map_image"].shape

    room_pixels = np.empty((T, *local_shape), dtype=np.uint8)
    map_images = np.empty((T, *map_shape), dtype=np.uint8)

    for t in range(T):
        obs = env._get_observation()
        room_pixels[t] = obs["obs"]
        map_images[t] = obs["map_image"]
        env.step(int(actions[t]))

    return room_pixels, map_images
