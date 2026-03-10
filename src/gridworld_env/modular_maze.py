"""
Modular Maze Environment.

A grid-world environment where:

* Any number of keys (labeled a-z in the ASCII layout) can be collected into a
  persistent inventory by standing on a key cell and taking the COLLECT_KEY
  action (action 5).  Keys are *not* consumed when used.
* Any number of safes (labeled 1-9) hold hidden rewards.  Safe cells are freely
  traversable.  To open a safe the agent must stand on it and take the USE_KEY
  action (action 4); the safe opens only if the inventory contains a key whose
  ID appears in the safe's ``unlocked_by`` list.
* Doors (labeled D) block movement entirely — the agent cannot step onto a door
  cell.  To open a door the agent must stand on an *adjacent* cell and take the
  USE_KEY action (action 4); this requires at least one key in the inventory.
  Once open the door cell becomes freely traversable.
* No Posner cueing or key-pair mechanics.

Observation modes
-----------------
symbolic (default)
    One-hot-encoded grid  +  normalised agent position  +  binary inventory
    vector.  Flattened to 1-D when ``flatten_obs=True`` (default).
symbolic_minimal
    Normalised agent position  +  binary inventory  +  per-key-instance
    availability bits  +  per-safe open/closed bits  +  per-door open/closed
    bits.  No grid.
pixels
    RGB array rendered with pygame (falls back to pure-numpy if unavailable).
both
    Dict with ``"pixels"`` and ``"symbolic"`` keys.

Cell types (for the symbolic grid)
-----------------------------------
EMPTY=0  WALL=1  AGENT=2  KEY=3  SAFE_CLOSED=4  SAFE_OPEN=5
DOOR_CLOSED=6  DOOR_OPEN=7
"""

from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gridworld_env.modular_layout import (
    ModularLayout,
    parse_modular_layout_file,
    parse_modular_layout_string,
)
from gridworld_env.modular_objects import ModularKey, NPC, Safe, SimpleDoor
from gridworld_env.world_layout import parse_world_layout_file


class _Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    USE_KEY = 4
    COLLECT_KEY = 5
    ENGAGE = 6
    FORGE_KEY = 7


class ModularCellType(IntEnum):
    """Cell type encoding for the symbolic grid observation."""
    EMPTY = 0
    WALL = 1
    AGENT = 2
    KEY = 3
    SAFE_CLOSED = 4
    SAFE_OPEN = 5
    DOOR_CLOSED = 6
    DOOR_OPEN = 7
    NPC = 8
    NPC_ENGAGED = 9
    MATERIAL = 10


_DELTAS = {
    _Action.UP: (-1, 0),
    _Action.DOWN: (1, 0),
    _Action.LEFT: (0, -1),
    _Action.RIGHT: (0, 1),
}

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _is_world_file(path: Path) -> bool:
    """Return True if *path* looks like a world topology file.

    Detected by checking whether the first non-blank, non-comment line
    of the file is exactly ``"rooms"``.
    """
    try:
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped == "rooms"
    except OSError:
        pass
    return False


class ModularMazeEnv(gym.Env):
    """Modular Maze Gymnasium environment.

    Parameters
    ----------
    layout:
        A :class:`~gridworld_env.modular_layout.ModularLayout`, a path to a
        ``.txt`` layout file, or an inline ASCII string.
    max_steps:
        Maximum steps per episode.  ``None`` means unlimited.
    step_reward:
        Reward added every step (typically a small negative value).
    collision_reward:
        Reward for walking into a wall.
    flatten_obs:
        Flatten observations to a 1-D array (default ``True``).
    obs_mode:
        One of ``"symbolic"``, ``"symbolic_minimal"``, ``"pixels"``,
        ``"both"``.
    render_mode:
        ``"human"`` (ASCII to stdout), ``"rgb_array"``, or ``None``.
    show_score:
        Include a score line in ASCII rendering.
    start_pos_mode:
        ``"fixed"`` uses the layout's S cell every episode;
        ``"random_in_room"`` samples a random empty cell in the starting room
        (flood-fill from S, stopping at walls and doors).
    terminate_on_all_safes_opened:
        End the episode (``terminated=True``) once every safe has been opened.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layout: Union[ModularLayout, str, Path],
        max_steps: Optional[int] = None,
        step_reward: float = -0.01,
        collision_reward: float = -0.1,
        flatten_obs: bool = True,
        obs_mode: str = "symbolic",
        render_mode: Optional[str] = None,
        show_score: bool = False,
        start_pos_mode: str = "fixed",
        terminate_on_all_safes_opened: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Parse layout
        # ------------------------------------------------------------------
        if isinstance(layout, ModularLayout):
            self._base_layout = layout
        elif isinstance(layout, Path):
            if _is_world_file(layout):
                self._base_layout = parse_world_layout_file(layout)
            else:
                self._base_layout = parse_modular_layout_file(layout)
        elif isinstance(layout, str) and Path(layout).exists():
            p = Path(layout)
            if _is_world_file(p):
                self._base_layout = parse_world_layout_file(p)
            else:
                self._base_layout = parse_modular_layout_file(p)
        elif isinstance(layout, str) and (
            layout.endswith(".txt") or "/" in layout or "\\" in layout
        ):
            raise FileNotFoundError(f"Layout file not found: '{layout}'.")
        else:
            self._base_layout = parse_modular_layout_string(layout)

        # ------------------------------------------------------------------
        # Validate options
        # ------------------------------------------------------------------
        if obs_mode not in (
            "symbolic", "symbolic_minimal", "pixels", "both", "macro", "room_pixels"
        ):
            raise ValueError(
                f"obs_mode must be one of 'symbolic', 'symbolic_minimal', "
                f"'pixels', 'both', 'macro', 'room_pixels'; got '{obs_mode}'"
            )
        if start_pos_mode not in ("fixed", "random_in_room"):
            raise ValueError(
                f"start_pos_mode must be 'fixed' or 'random_in_room'; "
                f"got '{start_pos_mode}'"
            )

        self.max_steps = max_steps
        self.step_reward = step_reward
        self.collision_reward = collision_reward
        self.flatten_obs = flatten_obs
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.show_score = show_score
        self.start_pos_mode = start_pos_mode
        self.terminate_on_all_safes_opened = terminate_on_all_safes_opened

        # ------------------------------------------------------------------
        # Derived constants (fixed across episodes)
        # ------------------------------------------------------------------
        # Sorted list of unique key IDs present in the layout
        self._unique_key_ids: List[str] = sorted(
            {k.id for k in self._base_layout.keys}
        )
        self._n_key_types: int = len(self._unique_key_ids)
        self._n_key_instances: int = len(self._base_layout.keys)
        self._n_safes: int = len(self._base_layout.safes)
        self._n_doors: int = len(self._base_layout.doors)

        # ------------------------------------------------------------------
        # Room topology (fixed across episodes)
        # ------------------------------------------------------------------
        self._compute_rooms()
        self._compute_room_bboxes()

        # ------------------------------------------------------------------
        # Episode state (initialised in reset())
        # ------------------------------------------------------------------
        self._layout: Optional[ModularLayout] = None
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._inventory: Set[str] = set()            # collected key IDs
        self._material_inventory: List[str] = []    # collected material names, FIFO order
        self._steps: int = 0
        self._last_valid_room: int = 0      # for room_pixels when on door cell

        # ------------------------------------------------------------------
        # Spaces
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(8)

        n_cell_types = len(ModularCellType)
        grid_shape = (self._base_layout.height, self._base_layout.width)
        max_dim = max(grid_shape)

        # ---- symbolic ----
        inv_size = max(self._n_key_types, 1)  # at least 1 to avoid empty box
        if flatten_obs:
            sym_size = grid_shape[0] * grid_shape[1] * n_cell_types + 2 + inv_size
            symbolic_space: spaces.Space = spaces.Box(
                low=0.0, high=1.0, shape=(sym_size,), dtype=np.float32
            )
        else:
            symbolic_space = spaces.Dict({
                "grid": spaces.Box(
                    low=0, high=n_cell_types - 1,
                    shape=grid_shape, dtype=np.int32,
                ),
                "agent_pos": spaces.Box(
                    low=0, high=max_dim, shape=(2,), dtype=np.int32,
                ),
                "inventory": spaces.Box(
                    low=0, high=1, shape=(inv_size,), dtype=np.int32,
                ),
            })

        # ---- symbolic_minimal ----
        min_size = 2 + inv_size + self._n_key_instances + self._n_safes + self._n_doors
        if flatten_obs:
            sym_min_space: spaces.Space = spaces.Box(
                low=0.0, high=1.0, shape=(min_size,), dtype=np.float32
            )
        else:
            min_dict: Dict[str, spaces.Space] = {
                "agent_pos": spaces.Box(
                    low=0, high=max_dim, shape=(2,), dtype=np.int32,
                ),
                "inventory": spaces.Box(
                    low=0, high=1, shape=(inv_size,), dtype=np.int32,
                ),
            }
            if self._n_key_instances > 0:
                min_dict["keys"] = spaces.Box(
                    low=0, high=1, shape=(self._n_key_instances,), dtype=np.int32,
                )
            if self._n_safes > 0:
                min_dict["safes"] = spaces.Box(
                    low=0, high=1, shape=(self._n_safes,), dtype=np.int32,
                )
            if self._n_doors > 0:
                min_dict["doors"] = spaces.Box(
                    low=0, high=1, shape=(self._n_doors,), dtype=np.int32,
                )
            sym_min_space = spaces.Dict(min_dict)

        # ---- pixels ----
        self._obs_cell_size = 32
        self._obs_status_height = 40
        cs = self._obs_cell_size
        sh = self._obs_status_height
        pixel_h = grid_shape[0] * cs + sh
        pixel_w = grid_shape[1] * cs
        pixel_space: spaces.Space = spaces.Box(
            low=0, high=255, shape=(pixel_h, pixel_w, 3), dtype=np.uint8
        )

        # ---- room_pixels ----
        # Shape is exactly the (uniform) room bounding-box size, plus status bar.
        room_pixel_h = self._room_h * cs + sh
        room_pixel_w = self._room_w * cs
        room_pixel_space: spaces.Space = spaces.Box(
            low=0, high=255, shape=(room_pixel_h, room_pixel_w, 3), dtype=np.uint8
        )

        # ---- macro ----
        # Room adjacency matrix: N_rooms × N_rooms binary (static topology).
        # Current room: scalar integer in [0, N_rooms] (N_rooms = on door cell).
        n_rooms = self._n_rooms
        if flatten_obs:
            # Flattened adjacency (N_rooms²) + normalised current-room scalar (1).
            macro_size = n_rooms * n_rooms + 1
            macro_space: spaces.Space = spaces.Box(
                low=0.0, high=1.0, shape=(macro_size,), dtype=np.float32
            )
        else:
            macro_space = spaces.Dict({
                "room_adjacency": spaces.Box(
                    low=0, high=1, shape=(n_rooms, n_rooms), dtype=np.float32,
                ),
                "current_room": spaces.Discrete(n_rooms + 1),
            })

        if obs_mode == "symbolic":
            self.observation_space = symbolic_space
        elif obs_mode == "symbolic_minimal":
            self.observation_space = sym_min_space
        elif obs_mode == "pixels":
            self.observation_space = pixel_space
        elif obs_mode == "room_pixels":
            self.observation_space = room_pixel_space
        elif obs_mode == "macro":
            self.observation_space = macro_space
        else:  # "both"
            self.observation_space = spaces.Dict({
                "pixels": pixel_space,
                "symbolic": symbolic_space,
            })

        # ------------------------------------------------------------------
        # Tabular indexing support
        # ------------------------------------------------------------------
        self._valid_positions: List[Tuple[int, int]] = sorted(
            (r, c)
            for r in range(self._base_layout.height)
            for c in range(self._base_layout.width)
            if not self._base_layout.is_wall(r, c)
        )
        self._position_to_index: Dict[Tuple[int, int], int] = {
            pos: i for i, pos in enumerate(self._valid_positions)
        }

        # Cache for random-start positions
        self._first_room_positions: Optional[List[Tuple[int, int]]] = None

        # Pygame surface (lazy-initialised)
        self._pygame_surface = None
        self._pygame_initialized = False

    # ======================================================================
    # Gymnasium interface
    # ======================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self._layout = self._base_layout.copy()

        if self.start_pos_mode == "random_in_room":
            positions = self._get_first_room_positions()
            idx = int(self.np_random.integers(len(positions)))
            self._agent_pos = positions[idx]
        else:
            self._agent_pos = self._layout.start_position

        self._inventory = set()
        self._material_inventory = []
        self._steps = 0
        self._last_valid_room = self._cell_to_room.get(
            self._layout.start_position, 0
        )

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self._layout is None:
            raise RuntimeError("Call reset() before step().")

        self._steps += 1
        reward = self.step_reward
        terminated = False
        truncated = False

        a = _Action(action)
        if a == _Action.USE_KEY:
            reward += self._use_key()
        elif a == _Action.COLLECT_KEY:
            self._collect_key()
        elif a == _Action.ENGAGE:
            self._engage_npc()
        elif a == _Action.FORGE_KEY:
            self._forge_key()
        else:
            move_reward, moved = self._handle_movement(a)
            reward += move_reward

        if self.terminate_on_all_safes_opened and self._all_safes_opened():
            terminated = True

        if self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ======================================================================
    # Movement
    # ======================================================================

    def _handle_movement(self, action: _Action) -> Tuple[float, bool]:
        """Attempt to move the agent.

        Closed doors block movement entirely; use USE_KEY from an adjacent cell
        to open a door first.

        Returns
        -------
        (additional_reward, moved)
        """
        row, col = self._agent_pos
        dr, dc = _DELTAS[action]
        new_pos = (row + dr, col + dc)
        nr, nc = new_pos

        # Wall check
        if self._layout.is_wall(nr, nc):
            return self.collision_reward, False

        # Closed door blocks movement
        door = self._layout.get_door_at(nr, nc)
        if door is not None and not door.is_open:
            return self.collision_reward, False

        # NPC cells block movement (engage from adjacent instead)
        if any(npc.position == new_pos for npc in self._layout.npcs):
            return self.collision_reward, False

        self._agent_pos = new_pos
        return 0.0, True

    # ======================================================================
    # Auto-collect
    # ======================================================================

    def _collect_key(self) -> None:
        """Execute the COLLECT_KEY action.

        Picks up the key or material at the agent's current position (if any
        and not already collected).  No-op if the cell contains neither.
        """
        row, col = self._agent_pos
        key = self._layout.get_key_at(row, col)
        if key is not None:
            key.collected = True
            self._inventory.add(key.id)
        material = self._layout.get_material_at(row, col)
        if material is not None:
            material.collected = True
            self._material_inventory.append(material.name)

    def _use_key(self) -> float:
        """Execute the USE_KEY action.

        * If the agent is standing on an unopened safe and holds a matching key,
          the safe is opened and its reward is returned.
        * If any adjacent cell contains a closed door and the agent holds at
          least one key, all such doors are opened.

        Both effects can occur simultaneously (e.g. standing on a safe next to
        a door).  If the agent has no keys the action is a no-op.

        Returns
        -------
        float
            Sum of rewards from any safes opened this step.
        """
        if not self._inventory:
            return 0.0

        row, col = self._agent_pos
        reward = 0.0

        # Open safe at current position
        safe = self._layout.get_safe_at(row, col)
        held_materials = {k.unique_material for k in self._layout.keys if k.collected}
        if safe is not None and safe.can_open_with(held_materials):
            safe.opened = True
            reward += safe.reward

        # Open all adjacent closed doors
        for dr, dc in _NEIGHBORS:
            door = self._layout.get_door_at(row + dr, col + dc)
            if door is not None and not door.is_open:
                door.is_open = True

        return reward

    def _engage_npc(self) -> None:
        """Execute the ENGAGE action.

        Marks all NPCs adjacent to the agent as engaged.  On the first
        engagement, if the NPC has a key_type set, a key of that type is
        spawned at a random unoccupied floor cell in the same room.
        """
        row, col = self._agent_pos
        for npc in self._layout.get_npc_adjacent(row, col):
            if not npc.engaged:
                npc.engaged = True
                if npc.key_type:
                    self._spawn_key_in_room(npc.position, npc.key_type)

    def _spawn_key_in_room(self, ref_pos: Tuple[int, int], key_type: str) -> None:
        """Spawn a key with the given unique_material at a random free floor cell
        in the same room as ref_pos.  No-op if no free cell is available."""
        room_id = self._cell_to_room.get(ref_pos)
        if room_id is None:
            return

        candidates = [pos for pos, rid in self._cell_to_room.items() if rid == room_id]

        occupied: Set[Tuple[int, int]] = {self._agent_pos}
        occupied |= {d.position for d in self._layout.doors}
        occupied |= {k.position for k in self._layout.keys if not k.collected}
        occupied |= {s.position for s in self._layout.safes}
        occupied |= {n.position for n in self._layout.npcs}
        occupied |= {m.position for m in self._layout.materials if not m.collected}

        free = [pos for pos in candidates if pos not in occupied]
        if not free:
            return

        spawn_pos = free[int(self.np_random.integers(len(free)))]
        self._layout.keys.append(ModularKey(
            id=f"wizard_{key_type}",
            position=spawn_pos,
            unique_material=key_type,
            collected=False,
        ))

    def _forge_key(self) -> None:
        """Execute the FORGE_KEY action.

        Consumes one ore and the first (FIFO) non-ore material from the
        material inventory to produce a key whose unique_material matches
        that non-ore material.  The forged key is immediately collected.
        No-op if ore is absent or no non-ore material is held.
        """
        inv = self._material_inventory
        if "ore" not in inv:
            return

        # First non-ore material in collection order (FIFO)
        target_mat = next((m for m in inv if m != "ore"), None)
        if target_mat is None:
            return  # only ore in inventory, nothing to pair it with

        inv.remove("ore")        # removes first occurrence
        inv.remove(target_mat)   # removes first occurrence (= FIFO material)

        forge_id = f"forge_{target_mat}"
        forged = ModularKey(
            id=forge_id,
            position=(0, 0),
            unique_material=target_mat,
            collected=True,
        )
        self._layout.keys.append(forged)
        self._inventory.add(forge_id)

    def _all_safes_opened(self) -> bool:
        return all(s.opened for s in self._layout.safes)

    # ======================================================================
    # Observations
    # ======================================================================

    def _get_observation(self) -> Any:
        if self.obs_mode == "symbolic":
            return self._symbolic_obs()
        elif self.obs_mode == "symbolic_minimal":
            return self._minimal_obs()
        elif self.obs_mode == "pixels":
            return self._pixel_obs()
        elif self.obs_mode == "room_pixels":
            return self._room_pixel_obs()
        elif self.obs_mode == "macro":
            return self._macro_obs()
        else:
            return {
                "pixels": self._pixel_obs(),
                "symbolic": self._symbolic_obs(),
            }

    # ---- symbolic --------------------------------------------------------

    def _symbolic_obs(self) -> Any:
        grid = self._build_grid()
        height, width = grid.shape
        inv_vec = self._inventory_vector()

        if self.flatten_obs:
            n_types = len(ModularCellType)
            one_hot = np.zeros((height, width, n_types), dtype=np.float32)
            for r in range(height):
                for c in range(width):
                    one_hot[r, c, grid[r, c]] = 1.0
            agent_pos = np.array([
                self._agent_pos[0] / max(1, height - 1),
                self._agent_pos[1] / max(1, width - 1),
            ], dtype=np.float32)
            return np.concatenate([one_hot.flatten(), agent_pos, inv_vec])
        else:
            return {
                "grid": grid,
                "agent_pos": np.array(self._agent_pos, dtype=np.int32),
                "inventory": inv_vec.astype(np.int32),
            }

    # ---- symbolic_minimal ------------------------------------------------

    def _minimal_obs(self) -> Any:
        height = self._base_layout.height
        width = self._base_layout.width

        agent_pos = np.array([
            self._agent_pos[0] / max(1, height - 1),
            self._agent_pos[1] / max(1, width - 1),
        ], dtype=np.float32)
        inv_vec = self._inventory_vector()
        keys_avail = np.array(
            [0.0 if k.collected else 1.0 for k in self._layout.keys],
            dtype=np.float32,
        )
        safes_state = np.array(
            [1.0 if s.opened else 0.0 for s in self._layout.safes],
            dtype=np.float32,
        )
        doors_state = np.array(
            [1.0 if d.is_open else 0.0 for d in self._layout.doors],
            dtype=np.float32,
        )

        if self.flatten_obs:
            parts = [agent_pos, inv_vec]
            if len(keys_avail) > 0:
                parts.append(keys_avail)
            if len(safes_state) > 0:
                parts.append(safes_state)
            if len(doors_state) > 0:
                parts.append(doors_state)
            return np.concatenate(parts)
        else:
            obs: Dict[str, Any] = {
                "agent_pos": np.array(self._agent_pos, dtype=np.int32),
                "inventory": inv_vec.astype(np.int32),
            }
            if self._n_key_instances > 0:
                obs["keys"] = keys_avail.astype(np.int32)
            if self._n_safes > 0:
                obs["safes"] = safes_state.astype(np.int32)
            if self._n_doors > 0:
                obs["doors"] = doors_state.astype(np.int32)
            return obs

    # ---- pixel -----------------------------------------------------------

    def _pixel_obs(self) -> np.ndarray:
        try:
            return self._render_pygame_surface()
        except Exception:
            return self._render_numpy_fallback()

    # ---- room_pixels -----------------------------------------------------

    def _room_pixel_obs(self) -> np.ndarray:
        """Return a pixel observation cropped to the agent's current room.

        All rooms have the same bounding-box size (enforced at init).
        When the agent occupies a door cell (between rooms), the last valid
        room is shown.
        """
        cs = self._obs_cell_size
        sh = self._obs_status_height

        # Resolve current room
        current_room = self._cell_to_room.get(self._agent_pos)
        if current_room is None:
            current_room = self._last_valid_room
        else:
            self._last_valid_room = current_room

        # Render the full grid, strip the status bar
        full_img = self._pixel_obs()
        grid_img = full_img[:-sh] if sh > 0 else full_img
        status_bar = full_img[-sh:] if sh > 0 else np.zeros((0, full_img.shape[1], 3), dtype=np.uint8)

        # Crop to room bounding box (always the same size across rooms)
        min_row, max_row, min_col, max_col = self._room_bboxes[current_room]
        y1, y2 = min_row * cs, (max_row + 1) * cs
        x1, x2 = min_col * cs, (max_col + 1) * cs
        room_crop = grid_img[y1:y2, x1:x2]

        if sh == 0:
            return room_crop

        # Crop status bar to the same width
        out_w = room_crop.shape[1]
        status_cropped = status_bar[:, :out_w]

        return np.concatenate([room_crop, status_cropped], axis=0)

    # ---- helpers ---------------------------------------------------------

    def _inventory_vector(self) -> np.ndarray:
        """Binary float32 vector: 1 if the agent holds that key type, else 0."""
        if self._n_key_types == 0:
            return np.zeros(1, dtype=np.float32)
        return np.array(
            [1.0 if kid in self._inventory else 0.0 for kid in self._unique_key_ids],
            dtype=np.float32,
        )

    def _build_grid(self) -> np.ndarray:
        """Build the integer cell-type grid for the current state."""
        layout = self._layout
        h, w = layout.height, layout.width
        grid = np.zeros((h, w), dtype=np.int32)

        for r in range(h):
            for c in range(w):
                if layout.is_wall(r, c):
                    grid[r, c] = ModularCellType.WALL

        for key in layout.keys:
            if not key.collected:
                grid[key.position[0], key.position[1]] = ModularCellType.KEY

        for safe in layout.safes:
            ct = ModularCellType.SAFE_OPEN if safe.opened else ModularCellType.SAFE_CLOSED
            grid[safe.position[0], safe.position[1]] = ct

        for door in layout.doors:
            ct = ModularCellType.DOOR_OPEN if door.is_open else ModularCellType.DOOR_CLOSED
            grid[door.position[0], door.position[1]] = ct

        for npc in layout.npcs:
            ct = ModularCellType.NPC_ENGAGED if npc.engaged else ModularCellType.NPC
            grid[npc.position[0], npc.position[1]] = ct

        for mat in layout.materials:
            if not mat.collected:
                grid[mat.position[0], mat.position[1]] = ModularCellType.MATERIAL

        grid[self._agent_pos[0], self._agent_pos[1]] = ModularCellType.AGENT

        return grid

    # ======================================================================
    # Info
    # ======================================================================

    def _get_info(self) -> Dict[str, Any]:
        layout = self._layout
        score = sum(s.reward for s in layout.safes if s.opened)
        return {
            "steps": self._steps,
            "inventory": sorted(self._inventory),
            "materials": sorted(self._material_inventory),
            "agent_pos": self._agent_pos,
            "safes_opened": sum(1 for s in layout.safes if s.opened),
            "safes_total": self._n_safes,
            "score": score,
        }

    # ======================================================================
    # Random-start helper
    # ======================================================================

    def _get_first_room_positions(self) -> List[Tuple[int, int]]:
        """Flood-fill from start, stopping at walls and doors."""
        if self._first_room_positions is not None:
            return self._first_room_positions

        layout = self._base_layout
        start = layout.start_position
        door_positions = {d.position for d in layout.doors}
        occupied = (
            {k.position for k in layout.keys}
            | {s.position for s in layout.safes}
        )

        visited: Set[Tuple[int, int]] = set()
        stack = [start]
        positions: List[Tuple[int, int]] = []

        while stack:
            pos = stack.pop()
            if pos in visited:
                continue
            visited.add(pos)
            r, c = pos
            if layout.is_wall(r, c):
                continue
            if pos in door_positions:
                continue
            if pos not in occupied:
                positions.append(pos)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                stack.append((r + dr, c + dc))

        self._first_room_positions = positions or [start]
        return self._first_room_positions

    # ======================================================================
    # Room topology
    # ======================================================================

    def _compute_rooms(self) -> None:
        """Partition floor cells into rooms via BFS, using walls and door cells
        as boundaries.

        Stores
        ------
        _n_rooms : int
        _cell_to_room : dict mapping (row, col) -> room_index.
            Door cells are absent (not members of any room).
        _room_key_indices : list of lists
            For each room, the indices (into ``_base_layout.keys``) of keys
            whose positions fall inside that room.
        _room_safe_indices : list of lists
            For each room, the indices (into ``_base_layout.safes``) of safes
            whose positions fall inside that room.
        """
        layout = self._base_layout

        # ------------------------------------------------------------------
        # Step 1: build cell_to_room mapping
        # ------------------------------------------------------------------
        if layout.room_cell_map is not None:
            # Declared room indices from a multi-file world layout
            cell_to_room: Dict[Tuple[int, int], int] = dict(layout.room_cell_map)
            room_ids = set(cell_to_room.values())
            self._n_rooms = (max(room_ids) + 1) if room_ids else 0
        else:
            # BFS flood-fill inference (single-file layout)
            door_positions = {d.position for d in layout.doors}
            visited: Set[Tuple[int, int]] = set()
            rooms_list: List[Set[Tuple[int, int]]] = []
            cell_to_room = {}

            for r in range(layout.height):
                for c in range(layout.width):
                    pos = (r, c)
                    if pos in visited or layout.is_wall(r, c) or pos in door_positions:
                        visited.add(pos)
                        continue
                    room_cells: Set[Tuple[int, int]] = set()
                    stack = [pos]
                    while stack:
                        p = stack.pop()
                        if p in visited:
                            continue
                        visited.add(p)
                        pr, pc = p
                        if layout.is_wall(pr, pc) or p in door_positions:
                            continue
                        room_cells.add(p)
                        for dr, dc in _NEIGHBORS:
                            nb = (pr + dr, pc + dc)
                            if nb not in visited:
                                stack.append(nb)
                    room_idx = len(rooms_list)
                    rooms_list.append(room_cells)
                    for p in room_cells:
                        cell_to_room[p] = room_idx

            self._n_rooms = len(rooms_list)

        self._cell_to_room: Dict[Tuple[int, int], int] = cell_to_room

        # ------------------------------------------------------------------
        # Step 2: static adjacency matrix
        # ------------------------------------------------------------------
        adj = np.zeros((self._n_rooms, self._n_rooms), dtype=np.float32)
        for door in layout.doors:
            dr, dc = door.position
            neighbour_rooms = set()
            for ddr, ddc in _NEIGHBORS:
                nb = (dr + ddr, dc + ddc)
                if nb in cell_to_room:
                    neighbour_rooms.add(cell_to_room[nb])
            neighbour_rooms_list = sorted(neighbour_rooms)
            for i in range(len(neighbour_rooms_list)):
                for j in range(i + 1, len(neighbour_rooms_list)):
                    a, b = neighbour_rooms_list[i], neighbour_rooms_list[j]
                    adj[a, b] = 1.0
                    adj[b, a] = 1.0
        self._room_adjacency: np.ndarray = adj

        # ------------------------------------------------------------------
        # Step 3: which keys / safes live in each room (for internal use)
        # ------------------------------------------------------------------
        self._room_key_indices: List[List[int]] = [[] for _ in range(self._n_rooms)]
        for idx, key in enumerate(layout.keys):
            room = cell_to_room.get(key.position, -1)
            if room >= 0:
                self._room_key_indices[room].append(idx)

        self._room_safe_indices: List[List[int]] = [[] for _ in range(self._n_rooms)]
        for idx, safe in enumerate(layout.safes):
            room = cell_to_room.get(safe.position, -1)
            if room >= 0:
                self._room_safe_indices[room].append(idx)

    # ======================================================================
    # Room bounding boxes (used by room_pixels obs mode)
    # ======================================================================

    def _compute_room_bboxes(self) -> None:
        """Compute the axis-aligned bounding box (in grid cells) of each room.

        When ``obs_mode == "room_pixels"``, all rooms must have the same
        bounding-box dimensions; a :class:`ValueError` is raised otherwise.

        Stores
        ------
        _room_bboxes : dict mapping room_id -> (min_row, max_row, min_col, max_col)
        _room_h      : int — room height in cells (validated equal across rooms)
        _room_w      : int — room width  in cells (validated equal across rooms)
        """
        from collections import defaultdict

        cell_groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for pos, rid in self._cell_to_room.items():
            cell_groups[rid].append(pos)

        h = self._base_layout.height
        w = self._base_layout.width
        self._room_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        for rid, cells in cell_groups.items():
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            # Expand by 1 in each direction so the surrounding walls (and door
            # cells on the border) are included in the room_pixels crop.
            self._room_bboxes[rid] = (
                max(0, min(rows) - 1),
                min(h - 1, max(rows) + 1),
                max(0, min(cols) - 1),
                min(w - 1, max(cols) + 1),
            )

        if self._room_bboxes:
            heights = {br - tr + 1 for tr, br, _, _ in self._room_bboxes.values()}
            widths  = {bc - lc + 1 for _, _, lc, bc in self._room_bboxes.values()}

            if self.obs_mode == "room_pixels" and (len(heights) > 1 or len(widths) > 1):
                sizes = {
                    rid: (br - tr + 1, bc - lc + 1)
                    for rid, (tr, br, lc, bc) in self._room_bboxes.items()
                }
                raise ValueError(
                    f"obs_mode='room_pixels' requires all rooms to have the same "
                    f"bounding-box dimensions, but found: {sizes}"
                )

            self._room_h = max(heights)
            self._room_w = max(widths)
        else:
            self._room_h = self._base_layout.height
            self._room_w = self._base_layout.width

    # ======================================================================
    # Macro observation
    # ======================================================================

    def _macro_obs(self) -> Any:
        """Build the macro observation.

        Returns
        -------
        Flattened float32 array (when ``flatten_obs=True``) or dict with keys
        ``"room_adjacency"`` and ``"current_room"``.

        Room adjacency
            N_rooms × N_rooms binary matrix (static): entry (i, j) = 1 if
            rooms i and j are directly connected through a door.

        Current room
            Integer room index (0 … N_rooms-1), or ``N_rooms`` when the agent
            occupies a door cell (not a member of any named room).
        """
        n_rooms = self._n_rooms
        current_room = self._cell_to_room.get(self._agent_pos, n_rooms)

        if self.flatten_obs:
            current_room_norm = np.array(
                [current_room / max(n_rooms, 1)], dtype=np.float32
            )
            return np.concatenate([
                self._room_adjacency.flatten(),
                current_room_norm,
            ])
        else:
            return {
                "room_adjacency": self._room_adjacency.copy(),
                "current_room": current_room,
            }

    # ======================================================================
    # Rendering
    # ======================================================================

    def render(self) -> Optional[np.ndarray]:
        """Render the environment according to ``render_mode``."""
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            self._render_ascii()
            return None
        elif self.render_mode == "rgb_array":
            try:
                return self._render_pygame_surface()
            except Exception:
                return self._render_numpy_fallback()
        return None

    # ---- ASCII -----------------------------------------------------------

    def _render_ascii(self) -> None:
        """Print an ASCII representation to stdout."""
        if self._layout is None:
            return

        layout = self._layout
        lines: List[str] = []

        if self.show_score:
            score = sum(s.reward for s in layout.safes if s.opened)
            lines.append(f"Score: {score:.2f}")

        inv_str = ", ".join(sorted(self._inventory)) or "empty"
        lines.append(f"Inventory: [{inv_str}]")

        opened_positions = {s.position for s in layout.safes if s.opened}

        for row in range(layout.height):
            line = ""
            for col in range(layout.width):
                pos = (row, col)

                if pos == self._agent_pos:
                    line += "@"
                    continue
                if layout.is_wall(row, col):
                    line += "#"
                    continue

                door = layout.get_door_at(row, col)
                if door is not None:
                    line += "_" if door.is_open else "D"
                    continue

                key = layout.get_key_at(row, col)
                if key is not None:
                    line += key.id
                    continue

                safe = layout.get_safe_at(row, col)
                if safe is not None:
                    line += safe.id
                    continue

                if pos in opened_positions:
                    line += "*"
                    continue

                npc = next((n for n in layout.npcs if n.position == pos), None)
                if npc is not None:
                    char = "W" if npc.npc_type == "wizard" else "N"
                    line += char.lower() if npc.engaged else char
                    continue

                mat = layout.get_material_at(row, col)
                if mat is not None:
                    line += mat.name[0].upper()
                    continue

                line += "."
            lines.append(line)

        print("\n".join(lines))
        print()

    # ---- Pygame ----------------------------------------------------------

    # Colour palette (RGB)
    _COLORS = {
        "bg":          (40,  40,  40),
        "wall":        (80,  80,  80),
        "floor":       (200, 200, 200),
        "agent":       (50,  150, 50),
        "key":         (255, 200, 0),
        "safe_closed": (139, 90,  43),
        "safe_open":   (180, 140, 90),
        "door_closed": (100, 60,  20),
        "door_open":   (160, 120, 80),
        "npc_wizard":          (160, 80,  220),
        "npc_wizard_engaged":  (220, 180, 255),
        "status_bg":           (30,  30,  30),
        "text":        (255, 255, 255),
    }

    # Material name → base RGB (matches color strings in modular_objects.py)
    _MATERIAL_COLORS = {
        "diamond":  (0,   210, 210),  # cyan
        "ruby":     (210,  50,  50),  # red
        "sapphire": (50,   80, 210),  # blue
        "ore":      (160, 160, 160),  # gray
    }

    def _material_rgb(self, material_name: str, brightness: float = 1.0) -> tuple:
        """Return an RGB tuple for a material, scaled by brightness (0–1)."""
        base = self._MATERIAL_COLORS.get(material_name, self._COLORS["key"])
        return tuple(min(255, int(v * brightness)) for v in base)

    def _render_pygame_surface(self) -> np.ndarray:
        """Render using pygame and return an RGB array."""
        import pygame

        cs = self._obs_cell_size
        sh = self._obs_status_height
        layout = self._layout
        h, w = layout.height, layout.width
        C = self._COLORS

        if not self._pygame_initialized:
            pygame.init()
            self._pygame_surface = pygame.Surface((w * cs, h * cs + sh))
            self._pygame_initialized = True

        surf = self._pygame_surface
        surf.fill(C["bg"])

        # Grid cells
        for row in range(h):
            for col in range(w):
                x, y = col * cs, row * cs
                rect = pygame.Rect(x, y, cs, cs)
                color = C["wall"] if layout.is_wall(row, col) else C["floor"]
                pygame.draw.rect(surf, color, rect)
                pygame.draw.rect(surf, C["bg"], rect, 1)  # grid lines

        # Doors
        for door in layout.doors:
            r, c = door.position
            color = C["door_open"] if door.is_open else C["door_closed"]
            pygame.draw.rect(surf, color, pygame.Rect(c * cs, r * cs, cs, cs))

        # Safes
        for safe in layout.safes:
            r, c = safe.position
            brightness = 0.9 if safe.opened else 0.55
            color = self._material_rgb(safe.unique_material, brightness)
            pygame.draw.rect(
                surf, color, pygame.Rect(c * cs + 2, r * cs + 2, cs - 4, cs - 4)
            )
            if not safe.opened:
                # Draw a small lock circle
                cx, cy = c * cs + cs // 2, r * cs + cs // 2
                pygame.draw.circle(surf, (60, 40, 10), (cx, cy), cs // 5)

        # Keys
        for key in layout.keys:
            if not key.collected:
                r, c = key.position
                kx = c * cs + cs // 2
                ky = r * cs + cs // 2
                key_color = self._material_rgb(key.unique_material)
                # Key bow (circle with hole)
                pygame.draw.circle(surf, key_color, (kx, ky - cs // 4), cs // 5)
                pygame.draw.circle(surf, C["floor"], (kx, ky - cs // 4), cs // 10)
                # Key shaft
                pygame.draw.rect(
                    surf, key_color,
                    pygame.Rect(kx - 3, ky - cs // 4, 6, cs // 2),
                )

        # NPCs
        for npc in layout.npcs:
            r, c = npc.position
            color_key = f"npc_{npc.npc_type}_engaged" if npc.engaged else f"npc_{npc.npc_type}"
            npc_color = C.get(color_key, (160, 80, 220))
            nx, ny = c * cs + cs // 2, r * cs + cs // 2
            pygame.draw.circle(surf, npc_color, (nx, ny), cs // 3)
            outline_color = (255, 255, 180) if npc.engaged else (255, 255, 255)
            pygame.draw.circle(surf, outline_color, (nx, ny), cs // 3, 2)

        # Materials
        for mat in layout.materials:
            if not mat.collected:
                r, c = mat.position
                mat_color = self._material_rgb(mat.name)
                mx, my = c * cs + cs // 2, r * cs + cs // 2
                half = cs // 3
                points = [(mx, my - half), (mx + half, my), (mx, my + half), (mx - half, my)]
                pygame.draw.polygon(surf, mat_color, points)

        # Agent
        ar, ac = self._agent_pos
        ax, ay = ac * cs + cs // 2, ar * cs + cs // 2
        pygame.draw.circle(surf, C["agent"], (ax, ay), cs // 3)

        # Status bar
        status_y = h * cs
        pygame.draw.rect(
            surf, C["status_bg"], pygame.Rect(0, status_y, w * cs, sh)
        )
        if self.show_score:
            score = sum(s.reward for s in layout.safes if s.opened)
            try:
                font = pygame.font.SysFont("monospace", 16)
                score_surf = font.render(f"Score: {score:.1f}", True, C["text"])
                surf.blit(score_surf, (10, status_y + 10))
            except Exception:
                pass

        return np.transpose(pygame.surfarray.array3d(surf), axes=(1, 0, 2))

    # ---- Numpy fallback --------------------------------------------------

    def _render_numpy_fallback(self) -> np.ndarray:
        """Minimal RGB rendering without pygame."""
        cs = self._obs_cell_size
        sh = self._obs_status_height
        layout = self._layout
        h, w = layout.height, layout.width
        C = self._COLORS

        img = np.zeros((h * cs + sh, w * cs, 3), dtype=np.uint8)

        # Background cells
        for r in range(h):
            for c in range(w):
                y1, y2 = r * cs, (r + 1) * cs
                x1, x2 = c * cs, (c + 1) * cs
                img[y1:y2, x1:x2] = C["wall"] if layout.is_wall(r, c) else C["floor"]

        # Doors
        for door in layout.doors:
            r, c = door.position
            color = C["door_open"] if door.is_open else C["door_closed"]
            img[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs] = color

        # Safes
        for safe in layout.safes:
            r, c = safe.position
            brightness = 0.9 if safe.opened else 0.55
            color = self._material_rgb(safe.unique_material, brightness)
            m = 2
            img[r * cs + m:(r + 1) * cs - m, c * cs + m:(c + 1) * cs - m] = color

        # Keys
        for key in layout.keys:
            if not key.collected:
                r, c = key.position
                m = cs // 4
                img[r * cs + m:(r + 1) * cs - m, c * cs + m:(c + 1) * cs - m] = self._material_rgb(key.unique_material)

        # NPCs
        for npc in layout.npcs:
            r, c = npc.position
            color_key = f"npc_{npc.npc_type}_engaged" if npc.engaged else f"npc_{npc.npc_type}"
            npc_color = C.get(color_key, (160, 80, 220))
            m = cs // 4
            img[r * cs + m:(r + 1) * cs - m, c * cs + m:(c + 1) * cs - m] = npc_color

        # Materials
        for mat in layout.materials:
            if not mat.collected:
                r, c = mat.position
                m = cs // 3
                img[r * cs + m:(r + 1) * cs - m, c * cs + m:(c + 1) * cs - m] = self._material_rgb(mat.name)

        # Agent
        ar, ac = self._agent_pos
        m = cs // 4
        img[ar * cs + m:(ar + 1) * cs - m, ac * cs + m:(ac + 1) * cs - m] = C["agent"]

        # Status bar
        img[h * cs:, :] = C["status_bg"]

        return img

    # ======================================================================
    # Cleanup
    # ======================================================================

    def close(self) -> None:
        """Release pygame resources if they were initialised."""
        if self._pygame_initialized:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False
            self._pygame_surface = None

    # ======================================================================
    # Properties
    # ======================================================================

    @property
    def agent_position(self) -> Tuple[int, int]:
        """Current agent position (row, col)."""
        return self._agent_pos

    @property
    def inventory(self) -> FrozenSet[str]:
        """Frozenset of key IDs currently held by the agent."""
        return frozenset(self._inventory)

    @property
    def current_room(self) -> int:
        """Index of the room the agent is currently in.

        Returns ``n_rooms`` when the agent is on an open door cell (not a
        member of any named room).
        """
        return self._cell_to_room.get(self._agent_pos, self._n_rooms)

    @property
    def n_rooms(self) -> int:
        """Number of rooms in the layout."""
        return self._n_rooms

    def get_action_meanings(self) -> List[str]:
        return ["UP", "DOWN", "LEFT", "RIGHT", "USE_KEY", "COLLECT_KEY"]

    # ======================================================================
    # Tabular RL support
    # ======================================================================

    @property
    def state_space_size(self) -> int:
        """Total number of discrete states for tabular RL."""
        n_pos = len(self._valid_positions)
        return (
            n_pos
            * (2 ** self._n_key_types)       # inventory
            * (2 ** self._n_key_instances)    # key availability on grid
            * (2 ** self._n_safes)            # safe states
            * (2 ** self._n_doors)            # door states
        )

    @property
    def state_index(self) -> int:
        """Unique integer index for the current state (tabular RL).

        Raises ``RuntimeError`` if the environment has not been reset.
        """
        if self._layout is None:
            raise RuntimeError("Call reset() first.")

        idx = self._position_to_index[self._agent_pos]

        for kid in self._unique_key_ids:
            idx = idx * 2 + (1 if kid in self._inventory else 0)

        for key in self._layout.keys:
            idx = idx * 2 + (0 if key.collected else 1)

        for safe in self._layout.safes:
            idx = idx * 2 + (1 if safe.opened else 0)

        for door in self._layout.doors:
            idx = idx * 2 + (1 if door.is_open else 0)

        return idx
