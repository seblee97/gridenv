"""BFS-based oracle policy for ModularMazeEnv.

Generates near-optimal trajectories with per-timestep phase labels for
training sequence models to detect task phases.

Phase labels
------------
0 - get_key:          Navigate to + collect a key (type-1 room or wizard-spawned key).
1 - make_key:         Navigate to ore, material, collect both, then forge (type-2 room).
2 - get_key_location: Navigate to wizard NPC + engage (type-3 room).
3 - collect_reward:   Navigate to safe + open it.
4 - goto_next_room:   Navigate to door + open it + traverse.  Also used for
                      post-completion wandering.

Material labels
---------------
0 - none, 1 - diamond, 2 - ruby, 3 - sapphire
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    GET_KEY = 0
    MAKE_KEY = 1
    GET_KEY_LOCATION = 2
    COLLECT_REWARD = 3
    GOTO_NEXT_ROOM = 4

PHASE_NAMES = ["get_key", "make_key", "get_key_location", "collect_reward", "goto_next_room"]

MATERIAL_TO_ID = {"none": 0, "diamond": 1, "ruby": 2, "sapphire": 3}
MATERIAL_NAMES = ["none", "diamond", "ruby", "sapphire"]

# Action IDs (must match ModularMazeEnv)
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
USE_KEY, COLLECT_KEY, ENGAGE, FORGE_KEY = 4, 5, 6, 7
N_ACTIONS = 8

_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ---------------------------------------------------------------------------
# Oracle Policy
# ---------------------------------------------------------------------------

class OraclePolicy:
    """BFS-based oracle that produces near-optimal actions and phase labels.

    Usage::

        env = ModularMazeEnv(layout, ...)
        oracle = OraclePolicy(env)

        obs, info = env.reset(seed=0)
        oracle.reset(agent_pos=override_pos, rng=np.random.default_rng(42))

        for t in range(max_steps):
            action, phase, material_id = oracle.get_action_and_labels()
            noisy_action = apply_noise(action, ...)
            obs, reward, term, trunc, info = env.step(noisy_action)
            oracle.notify_step()
    """

    def __init__(self, env) -> None:
        self.env = env
        self._build_room_door_map()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_room_door_map(self) -> None:
        """Build mapping: (room_a, room_b) -> list of door objects."""
        cell_to_room = self.env._cell_to_room
        layout = self.env._base_layout

        # door -> pair of rooms it connects
        self.door_rooms: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # (room_a, room_b) -> list of doors (usually 1)
        self.room_pair_doors: Dict[Tuple[int, int], list] = {}

        for door in layout.doors:
            dr, dc = door.position
            neighbor_rooms: Set[int] = set()
            for ddr, ddc in _NEIGHBORS:
                nb = (dr + ddr, dc + ddc)
                if nb in cell_to_room:
                    neighbor_rooms.add(cell_to_room[nb])
            rooms = sorted(neighbor_rooms)
            if len(rooms) >= 2:
                pair = (rooms[0], rooms[1])
                self.door_rooms[door.position] = pair
                self.room_pair_doors.setdefault(pair, []).append(door)
                self.room_pair_doors.setdefault((pair[1], pair[0]), []).append(door)

    def reset(self, agent_pos: Tuple[int, int], rng: np.random.Generator) -> None:
        """Reset oracle state for a new episode.

        Call *after* ``env.reset()`` and any start-position override.
        """
        self.rng = rng
        self._safe_order: List[int] = []  # indices into layout.safes
        self._current_safe_ptr = 0
        self._wander_target: Optional[Tuple[int, int]] = None
        self._replan_safe_order()

    # ------------------------------------------------------------------
    # Safe ordering (nearest-first by BFS distance)
    # ------------------------------------------------------------------

    def _replan_safe_order(self) -> None:
        """Order remaining unopened safes by BFS distance from agent."""
        layout = self.env._layout
        agent_pos = self.env._agent_pos
        has_keys = bool(self.env._inventory)

        remaining: List[Tuple[float, int]] = []
        for i, safe in enumerate(layout.safes):
            if safe.opened:
                continue
            dist = self._bfs_distance(agent_pos, safe.position, can_open_doors=has_keys)
            remaining.append((dist, i))

        remaining.sort()
        self._safe_order = [idx for _, idx in remaining]
        self._current_safe_ptr = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get_action_and_labels(self) -> Tuple[int, int, int]:
        """Return ``(action, phase_id, material_id)`` for the current step."""
        layout = self.env._layout

        # Advance past already-opened safes; re-plan if any were skipped
        # (e.g. opened by noise or USE_KEY side-effect).
        need_replan = False
        while self._current_safe_ptr < len(self._safe_order):
            si = self._safe_order[self._current_safe_ptr]
            if not layout.safes[si].opened:
                break
            self._current_safe_ptr += 1
            need_replan = True

        if need_replan or self._current_safe_ptr >= len(self._safe_order):
            self._replan_safe_order()

        if self._current_safe_ptr >= len(self._safe_order):
            return self._wander()

        safe = layout.safes[self._safe_order[self._current_safe_ptr]]
        material = safe.unique_material
        mat_id = MATERIAL_TO_ID.get(material, 0)

        # Can we already open this safe?
        held_mats = {k.unique_material for k in layout.keys if k.collected}
        if material in held_mats:
            return self._go_open_safe(safe, mat_id)

        # Need to acquire key — determine strategy from layout objects.
        return self._acquire_key_step(safe, material, mat_id)

    def notify_step(self) -> None:
        """Call after env.step() so the oracle can react to state changes."""
        # Stateless relative to env — nothing to do.  The oracle re-derives
        # everything from the live env state each call to get_action_and_labels.
        pass

    # ------------------------------------------------------------------
    # Go open a safe (we already have the key)
    # ------------------------------------------------------------------

    def _go_open_safe(self, safe, mat_id: int) -> Tuple[int, int, int]:
        agent_pos = self.env._agent_pos
        if agent_pos == safe.position:
            return USE_KEY, Phase.COLLECT_REWARD, mat_id
        return self._navigate(safe.position, Phase.COLLECT_REWARD, mat_id)

    # ------------------------------------------------------------------
    # Key acquisition logic
    # ------------------------------------------------------------------

    def _acquire_key_step(self, safe, material: str, mat_id: int):
        """Decide the next action to acquire the key for *safe*."""
        layout = self.env._layout
        env = self.env

        # --- Type 1: key room — uncollected key with matching material ---
        matching_key = self._find_uncollected_key(material)
        if matching_key is not None:
            agent_pos = env._agent_pos
            if agent_pos == matching_key.position:
                return COLLECT_KEY, Phase.GET_KEY, mat_id
            return self._navigate(matching_key.position, Phase.GET_KEY, mat_id)

        # --- Type 3: wizard room — NPC with matching key_type ---
        # (check before forge since engaging a wizard spawns a key)
        matching_npc = self._find_unengaged_npc(material)
        if matching_npc is not None:
            agent_pos = env._agent_pos
            adj = self._adjacent_to(matching_npc.position, agent_pos)
            if adj:
                return ENGAGE, Phase.GET_KEY_LOCATION, mat_id
            # Navigate to cell adjacent to NPC
            target = self._pick_adjacent_cell(matching_npc.position)
            if target is None:
                return self.rng.integers(4), Phase.GET_KEY_LOCATION, mat_id
            return self._navigate(target, Phase.GET_KEY_LOCATION, mat_id)

        # --- Type 2: forge room — collect ore + material, then forge ---
        return self._forge_step(material, mat_id)

    def _forge_step(self, material: str, mat_id: int):
        """Handle the multi-step forge workflow."""
        env = self.env
        inv = env._material_inventory

        has_ore = "ore" in inv
        has_gem = material in inv

        if has_ore and has_gem:
            # Ready to forge
            return FORGE_KEY, Phase.MAKE_KEY, mat_id

        # Still need to collect something
        layout = env._layout

        if not has_ore:
            ore = self._find_uncollected_material("ore")
            if ore is not None:
                if env._agent_pos == ore.position:
                    return COLLECT_KEY, Phase.MAKE_KEY, mat_id
                return self._navigate(ore.position, Phase.MAKE_KEY, mat_id)

        if not has_gem:
            gem = self._find_uncollected_material(material)
            if gem is not None:
                if env._agent_pos == gem.position:
                    return COLLECT_KEY, Phase.MAKE_KEY, mat_id
                return self._navigate(gem.position, Phase.MAKE_KEY, mat_id)

        # Fallback: nothing found (shouldn't happen in well-formed layouts)
        return self.rng.integers(4), Phase.MAKE_KEY, mat_id

    # ------------------------------------------------------------------
    # Post-completion wandering
    # ------------------------------------------------------------------

    def _wander(self) -> Tuple[int, int, int]:
        """All safes opened — navigate to a random room."""
        env = self.env
        cell_to_room = env._cell_to_room
        agent_pos = env._agent_pos
        current_room = cell_to_room.get(agent_pos, env._last_valid_room)
        mat_id = MATERIAL_TO_ID["none"]

        # Pick or re-pick a wander target
        if (self._wander_target is None
                or agent_pos == self._wander_target
                or cell_to_room.get(self._wander_target) == current_room):
            other_rooms = [r for r in range(env._n_rooms) if r != current_room]
            if not other_rooms:
                return int(self.rng.integers(4)), Phase.GOTO_NEXT_ROOM, mat_id
            target_room = other_rooms[int(self.rng.integers(len(other_rooms)))]
            # Pick a random cell in that room
            room_cells = [pos for pos, rid in cell_to_room.items() if rid == target_room]
            self._wander_target = room_cells[int(self.rng.integers(len(room_cells)))]

        return self._navigate(self._wander_target, Phase.GOTO_NEXT_ROOM, mat_id)

    # ------------------------------------------------------------------
    # Navigation (handles cross-room door traversal)
    # ------------------------------------------------------------------

    def _navigate(self, target: Tuple[int, int], target_phase: int,
                  mat_id: int) -> Tuple[int, int, int]:
        """Navigate toward *target*, opening doors as needed.

        Phase is ``target_phase`` when agent is in the same room as target,
        ``GOTO_NEXT_ROOM`` otherwise.
        """
        env = self.env
        agent_pos = env._agent_pos
        has_keys = bool(env._inventory)

        path = self._bfs(agent_pos, target, can_open_doors=has_keys)
        if not path or len(path) < 2:
            # No path — random move as fallback
            return int(self.rng.integers(4)), target_phase, mat_id

        next_cell = path[1]

        # If next cell is a closed door, open it
        door = env._layout.get_door_at(*next_cell)
        if door is not None and not door.is_open:
            return USE_KEY, Phase.GOTO_NEXT_ROOM, mat_id

        # Determine phase: same room as target → target_phase, else transit
        agent_room = self._get_room(agent_pos)
        target_room = self._get_room(target)
        phase = target_phase if agent_room == target_room else Phase.GOTO_NEXT_ROOM

        move = self._pos_to_move(agent_pos, next_cell)
        return move, phase, mat_id

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    def _bfs(self, start: Tuple[int, int], goal: Tuple[int, int],
             can_open_doors: bool = False) -> List[Tuple[int, int]]:
        """Shortest path from *start* to *goal*.

        Walls and NPC cells are always impassable (unless the NPC cell is the
        goal — but the oracle never targets NPC cells directly; it targets
        adjacent cells instead).

        Closed doors are impassable unless *can_open_doors* is True.
        """
        if start == goal:
            return [start]

        layout = self.env._layout
        npc_positions = {npc.position for npc in layout.npcs}

        visited: Set[Tuple[int, int]] = {start}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        queue: deque[Tuple[int, int]] = deque([start])

        while queue:
            pos = queue.popleft()
            for dr, dc in _NEIGHBORS:
                nb = (pos[0] + dr, pos[1] + dc)
                if nb in visited:
                    continue

                # Wall check
                if layout.is_wall(*nb):
                    visited.add(nb)
                    continue

                # NPC blocks movement (unless nb is the goal — shouldn't happen)
                if nb in npc_positions and nb != goal:
                    visited.add(nb)
                    continue

                # Closed door check
                door = layout.get_door_at(*nb)
                if door is not None and not door.is_open and not can_open_doors:
                    visited.add(nb)
                    continue

                visited.add(nb)
                parent[nb] = pos
                if nb == goal:
                    return self._reconstruct(parent, start, goal)
                queue.append(nb)

        return []  # no path

    def _bfs_distance(self, start: Tuple[int, int], goal: Tuple[int, int],
                      can_open_doors: bool = False) -> float:
        path = self._bfs(start, goal, can_open_doors=can_open_doors)
        return len(path) - 1 if path else float("inf")

    @staticmethod
    def _reconstruct(parent, start, goal) -> List[Tuple[int, int]]:
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_room(self, pos: Tuple[int, int]) -> int:
        room = self.env._cell_to_room.get(pos)
        if room is None:
            return self.env._last_valid_room
        return room

    @staticmethod
    def _pos_to_move(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        for action, (adr, adc) in _DELTAS.items():
            if adr == dr and adc == dc:
                return action
        raise ValueError(f"Cannot move from {src} to {dst} in one step")

    @staticmethod
    def _adjacent_to(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _pick_adjacent_cell(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Pick a reachable floor cell adjacent to *pos*."""
        layout = self.env._layout
        agent_pos = self.env._agent_pos
        candidates = []
        for dr, dc in _NEIGHBORS:
            nb = (pos[0] + dr, pos[1] + dc)
            if not layout.is_wall(*nb):
                npc_there = any(npc.position == nb for npc in layout.npcs)
                if not npc_there:
                    candidates.append(nb)
        if not candidates:
            return None
        # Pick nearest by BFS distance
        best, best_d = None, float("inf")
        for c in candidates:
            d = self._bfs_distance(agent_pos, c, can_open_doors=bool(self.env._inventory))
            if d < best_d:
                best, best_d = c, d
        return best

    def _find_uncollected_key(self, material: str):
        """Find the nearest reachable uncollected key with matching unique_material."""
        agent_pos = self.env._agent_pos
        has_keys = bool(self.env._inventory)
        best, best_d = None, float("inf")
        for key in self.env._layout.keys:
            if not key.collected and key.unique_material == material:
                d = self._bfs_distance(agent_pos, key.position, can_open_doors=has_keys)
                if d < best_d:
                    best, best_d = key, d
        return best

    def _find_unengaged_npc(self, material: str):
        """Find the nearest reachable unengaged NPC with matching key_type."""
        agent_pos = self.env._agent_pos
        has_keys = bool(self.env._inventory)
        best, best_d = None, float("inf")
        for npc in self.env._layout.npcs:
            if not npc.engaged and npc.key_type == material:
                # NPC cells are blocked in BFS, so check distance to adjacent cells
                adj = self._pick_adjacent_cell(npc.position)
                if adj is not None:
                    d = self._bfs_distance(agent_pos, adj, can_open_doors=has_keys)
                    if d < best_d:
                        best, best_d = npc, d
        return best

    def _find_uncollected_material(self, name: str):
        """Find the nearest reachable uncollected material by name."""
        agent_pos = self.env._agent_pos
        has_keys = bool(self.env._inventory)
        best, best_d = None, float("inf")
        for mat in self.env._layout.materials:
            if not mat.collected and mat.name == name:
                d = self._bfs_distance(agent_pos, mat.position, can_open_doors=has_keys)
                if d < best_d:
                    best, best_d = mat, d
        return best


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def apply_directional_noise(action: int, epsilon: float,
                            rng: np.random.Generator) -> int:
    """With probability *epsilon*, replace movement actions with a random direction."""
    if action > 3:
        return action  # interaction actions are never corrupted
    if rng.random() < epsilon:
        return int(rng.integers(4))
    return action


def apply_uniform_noise(action: int, epsilon: float,
                        rng: np.random.Generator) -> int:
    """With probability *epsilon*, replace *any* action with a uniformly random one."""
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return action
