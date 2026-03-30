"""Distribution over layout variants sharing a fixed room structure.

Generates variants by randomizing positions of interactive elements:
- Floor objects (keys, safes, materials, NPCs) stay within their own room.
- Doors stay on the same shared-wall segment (between the same room pair).
- The grid is updated so door cells are re-walled / re-carved correctly.

Typical usage::

    from gridworld_env.procgen import generate_world_grid
    from gridworld_env.env_distribution import LayoutDistribution

    base = generate_world_grid(n_rooms=4, seed=0)
    dist = LayoutDistribution(base, n_variants=50, seed=42)

    layout_v3 = dist.get_layout(3)          # ModularLayout for variant 3
    dist.save_variants("variants.npz")       # compact position archive
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gridworld_env.modular_layout import ModularLayout
from gridworld_env.modular_objects import Material, ModularKey, NPC, Safe, SimpleDoor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _room_floor_cells(layout: ModularLayout) -> Dict[int, List[Tuple[int, int]]]:
    """Return mapping room_id -> list of (row, col) floor cells in that room."""
    result: Dict[int, List[Tuple[int, int]]] = {}
    if layout.room_cell_map is None:
        return result
    for pos, room_id in layout.room_cell_map.items():
        r, c = pos
        if not layout.grid[r][c]:
            result.setdefault(room_id, []).append(pos)
    return result


def _door_candidates(
    door: SimpleDoor,
    layout: ModularLayout,
) -> List[Tuple[int, int]]:
    """All valid wall cells on the same shared-wall segment as *door*.

    Candidates are wall cells (``grid[r][c] == True``) whose two floor
    neighbours (left/right for a vertical wall, up/down for a horizontal wall)
    belong to the same rooms as the neighbours of the current door position.
    This restricts relocation to the specific pair of rooms the door connects.
    """
    r, c = door.position
    grid = layout.grid
    rcm = layout.room_cell_map or {}
    H, W = layout.height, layout.width

    has_left  = c > 0       and not grid[r][c - 1]
    has_right = c < W - 1   and not grid[r][c + 1]
    has_up    = r > 0       and not grid[r - 1][c]
    has_down  = r < H - 1   and not grid[r + 1][c]

    candidates: List[Tuple[int, int]] = []

    if has_left and has_right:
        # Door is in a vertical wall (column c); passage is horizontal.
        room_L = rcm.get((r, c - 1))
        room_R = rcm.get((r, c + 1))
        for row in range(1, H - 1):
            if (grid[row][c]                           # still a wall cell
                    and rcm.get((row, c - 1)) == room_L
                    and rcm.get((row, c + 1)) == room_R):
                candidates.append((row, c))

    elif has_up and has_down:
        # Door is in a horizontal wall (row r); passage is vertical.
        room_T = rcm.get((r - 1, c))
        room_B = rcm.get((r + 1, c))
        for col in range(1, W - 1):
            if (grid[r][col]                           # still a wall cell
                    and rcm.get((r - 1, col)) == room_T
                    and rcm.get((r + 1, col)) == room_B):
                candidates.append((r, col))

    return candidates if candidates else [door.position]


# ---------------------------------------------------------------------------
# Core randomization
# ---------------------------------------------------------------------------

def randomize_layout(
    base: ModularLayout,
    rng: np.random.Generator,
) -> ModularLayout:
    """Return a deep copy of *base* with interactive element positions randomized.

    Invariants preserved
    --------------------
    - Room structure (walls, topology) is unchanged.
    - Each floor object stays within its original room.
    - Each door stays on the shared wall between its original room pair.
    - No two objects share the same cell.
    - The ``grid`` is updated so moved door cells are re-walled / re-carved.

    Parameters
    ----------
    base:
        A :class:`~gridworld_env.modular_layout.ModularLayout` produced by
        ``generate_world_grid`` or ``generate_world`` (must have
        ``room_cell_map`` populated).
    rng:
        NumPy random generator (caller controls reproducibility).
    """
    layout = copy.deepcopy(base)
    grid = layout.grid
    room_cells = _room_floor_cells(layout)

    # Track occupied floor cells (start position is fixed).
    occupied: set[Tuple[int, int]] = {layout.start_position}

    # ------------------------------------------------------------------
    # 1. Doors — stay on the same shared-wall segment, no two at same cell.
    # ------------------------------------------------------------------
    door_taken: set[Tuple[int, int]] = set()
    new_door_positions: List[Tuple[int, int]] = []

    for door in layout.doors:
        candidates = [p for p in _door_candidates(door, layout) if p not in door_taken]
        if not candidates:
            candidates = [door.position]
        idx = int(rng.integers(len(candidates)))
        new_pos = candidates[idx]
        new_door_positions.append(new_pos)
        door_taken.add(new_pos)

    for door, old_pos, new_pos in zip(
        layout.doors,
        [d.position for d in layout.doors],
        new_door_positions,
    ):
        if old_pos != new_pos:
            grid[old_pos[0]][old_pos[1]] = True   # restore old cell to wall
            grid[new_pos[0]][new_pos[1]] = False  # carve new cell as floor
        door.position = new_pos

    # ------------------------------------------------------------------
    # Helper: pick a random free floor cell inside room_id.
    # ------------------------------------------------------------------
    def pick(room_id: Optional[int]) -> Optional[Tuple[int, int]]:
        if room_id is None or room_id not in room_cells:
            return None
        free = [p for p in room_cells[room_id] if p not in occupied]
        if not free:
            return None
        chosen = free[int(rng.integers(len(free)))]
        occupied.add(chosen)
        return chosen

    def room_of(pos: Tuple[int, int]) -> Optional[int]:
        if layout.room_cell_map is None:
            return None
        return layout.room_cell_map.get(pos)

    # ------------------------------------------------------------------
    # 2. Keys
    # ------------------------------------------------------------------
    for key in layout.keys:
        new_pos = pick(room_of(key.position))
        if new_pos is not None:
            key.position = new_pos

    # ------------------------------------------------------------------
    # 3. Safes
    # ------------------------------------------------------------------
    for safe in layout.safes:
        new_pos = pick(room_of(safe.position))
        if new_pos is not None:
            safe.position = new_pos

    # ------------------------------------------------------------------
    # 4. Materials (ore, diamond, ruby, sapphire)
    # ------------------------------------------------------------------
    for mat in layout.materials:
        new_pos = pick(room_of(mat.position))
        if new_pos is not None:
            mat.position = new_pos

    # ------------------------------------------------------------------
    # 5. NPCs
    # ------------------------------------------------------------------
    for npc in layout.npcs:
        new_pos = pick(room_of(npc.position))
        if new_pos is not None:
            npc.position = new_pos

    return layout


# ---------------------------------------------------------------------------
# Compact variant record (positions only — for saving/loading)
# ---------------------------------------------------------------------------

@dataclass
class LayoutVariant:
    """Compact record of one layout variant: just the (row, col) positions."""
    key_positions:      np.ndarray   # (K, 2) int16
    safe_positions:     np.ndarray   # (S, 2) int16
    door_positions:     np.ndarray   # (D, 2) int16
    material_positions: np.ndarray   # (M, 2) int16
    npc_positions:      np.ndarray   # (N, 2) int16


def apply_variant(base: ModularLayout, variant: LayoutVariant) -> ModularLayout:
    """Reconstruct a layout by applying saved positions from *variant* to *base*.

    Use this to replay or render a previously-saved variant without re-running
    ``randomize_layout``.

    Parameters
    ----------
    base:
        The original base layout (same seed / parameters as when the variant
        was generated).
    variant:
        A :class:`LayoutVariant` loaded from ``variants.npz``.
    """
    layout = copy.deepcopy(base)
    grid = layout.grid

    for key, pos in zip(layout.keys, variant.key_positions):
        key.position = tuple(int(x) for x in pos)

    for safe, pos in zip(layout.safes, variant.safe_positions):
        safe.position = tuple(int(x) for x in pos)

    for door, old_pos, new_pos_arr in zip(
        layout.doors,
        [d.position for d in layout.doors],
        variant.door_positions,
    ):
        new_pos = tuple(int(x) for x in new_pos_arr)
        if old_pos != new_pos:
            grid[old_pos[0]][old_pos[1]] = True
            grid[new_pos[0]][new_pos[1]] = False
        door.position = new_pos

    for mat, pos in zip(layout.materials, variant.material_positions):
        mat.position = tuple(int(x) for x in pos)

    for npc, pos in zip(layout.npcs, variant.npc_positions):
        npc.position = tuple(int(x) for x in pos)

    return layout


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

class LayoutDistribution:
    """A set of layout variants sharing the same room structure.

    All variants are generated up-front from *base_layout* and cached in
    memory.  Object positions are shifted per variant; the room topology
    (walls, which rooms exist, which pairs are connected) is identical across
    all variants.

    Parameters
    ----------
    base_layout:
        Template layout from ``generate_world_grid`` or ``generate_world``.
        Must have ``room_cell_map`` set (both procgen functions do this).
    n_variants:
        Number of randomized variants to pre-generate.
    seed:
        RNG seed for reproducibility of the distribution itself.
    """

    def __init__(
        self,
        base_layout: ModularLayout,
        n_variants: int,
        seed: int = 0,
    ) -> None:
        self.base_layout = base_layout
        self.n_variants  = n_variants
        self.seed        = seed

        rng = np.random.default_rng(seed)
        self._variants: List[ModularLayout] = [
            randomize_layout(base_layout, rng) for _ in range(n_variants)
        ]

    def get_layout(self, variant_idx: int) -> ModularLayout:
        """Return the :class:`~gridworld_env.modular_layout.ModularLayout` for
        variant *variant_idx* (0-indexed)."""
        return self._variants[variant_idx]

    def get_variant_record(self, variant_idx: int) -> LayoutVariant:
        """Return a compact :class:`LayoutVariant` (positions only) for saving."""
        layout = self._variants[variant_idx]

        def arr(objs) -> np.ndarray:
            return np.array(
                [o.position for o in objs], dtype=np.int16
            ).reshape(-1, 2)

        return LayoutVariant(
            key_positions      = arr(layout.keys),
            safe_positions     = arr(layout.safes),
            door_positions     = arr(layout.doors),
            material_positions = arr(layout.materials),
            npc_positions      = arr(layout.npcs),
        )

    def save_variants(self, path: str | Path) -> None:
        """Save all variant positions to a compressed ``.npz`` file.

        The saved arrays can be loaded and applied with
        :func:`apply_variant` to reconstruct any variant layout.
        """
        records = [self.get_variant_record(i) for i in range(self.n_variants)]

        def stack(attr: str) -> np.ndarray:
            arrays = [getattr(r, attr) for r in records]
            # Handle empty arrays (e.g. no NPCs) — give shape (N, 0, 2)
            if all(a.shape[0] == 0 for a in arrays):
                return np.zeros((self.n_variants, 0, 2), dtype=np.int16)
            return np.stack(arrays)

        np.savez_compressed(
            path,
            key_positions      = stack("key_positions"),
            safe_positions     = stack("safe_positions"),
            door_positions     = stack("door_positions"),
            material_positions = stack("material_positions"),
            npc_positions      = stack("npc_positions"),
        )

    @staticmethod
    def load_variants(
        path: str | Path,
        base_layout: ModularLayout,
    ) -> List[ModularLayout]:
        """Load variant layouts from a previously-saved ``.npz`` file.

        Parameters
        ----------
        path:
            Path to the ``.npz`` produced by :meth:`save_variants`.
        base_layout:
            The original base layout used when the distribution was created.

        Returns
        -------
        List[ModularLayout]
            One layout per saved variant.
        """
        data = np.load(path)
        n = data["door_positions"].shape[0]
        layouts = []
        for i in range(n):
            variant = LayoutVariant(
                key_positions      = data["key_positions"][i],
                safe_positions     = data["safe_positions"][i],
                door_positions     = data["door_positions"][i],
                material_positions = data["material_positions"][i],
                npc_positions      = data["npc_positions"][i],
            )
            layouts.append(apply_variant(base_layout, variant))
        return layouts
