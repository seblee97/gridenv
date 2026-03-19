"""Procedural world generation for ModularMazeEnv using Binary Space Partitioning.

BSP divides a rectangular canvas into rooms by recursively splitting it.
Each split produces two child rectangles that share exactly one wall, so a door
can always be placed somewhere on that shared wall.  This lets rooms have truly
independent heights and widths — not constrained to a regular grid.

How BSP works
-------------
1. Start with a single rectangle covering the whole canvas.
2. Repeatedly pick a leaf rectangle and split it (horizontally or vertically)
   until we have N leaves.
3. Each leaf becomes one room (walls = border of the rectangle).
4. Walk the BSP tree: at every internal node, carve a door in the wall shared
   by its two children.  Because the split is exact, that wall always spans the
   full width (h-split) or height (v-split) of the parent — at least some cells
   on it border room interiors on both sides, guaranteeing a valid door position.

The result is a spanning tree of N rooms with N-1 doors and no corridors.

Room types
----------
1 – Key room:    the correct key for the safe is placed in the room.
2 – Forge room:  ore + the matching material are placed in the room;
                 the agent uses FORGE_KEY to create the key.
3 – Wizard room: a wizard NPC is placed; engaging the wizard spawns a key
                 of the correct type in the room.

Distractor option
-----------------
When ``distractor=True``, each room has a 50 % chance of an extra irrelevant
element (wrong-type wizard / material / key depending on room type).

Usage
-----
    from gridworld_env.procgen import generate_world
    from gridworld_env.modular_maze import ModularMazeEnv

    layout = generate_world(n_rooms=6, distractor=True, seed=42)
    env = ModularMazeEnv(layout=layout, obs_mode="pixels", render_mode="rgb_array")
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from gridworld_env.modular_layout import ModularLayout
from gridworld_env.modular_objects import (
    MATERIAL_TYPES,
    Material,
    ModularKey,
    NPC,
    Safe,
    SimpleDoor,
)

# Ordered list of material types available for safes / keys.
_MATERIALS: List[str] = ["diamond", "ruby", "sapphire"]


# ---------------------------------------------------------------------------
# BSP tree node
# ---------------------------------------------------------------------------

@dataclass
class _BSPNode:
    """A rectangular region in the BSP tree."""
    top: int
    left: int
    h: int   # height in rows (includes border walls)
    w: int   # width  in cols (includes border walls)
    split_horizontal: bool = False  # True → children are top / bottom halves
    split_pos: int = 0              # shared-wall row (h-split) or col (v-split)
    left_child: Optional["_BSPNode"] = None   # top child or left child
    right_child: Optional["_BSPNode"] = None  # bottom child or right child

    @property
    def is_leaf(self) -> bool:
        return self.left_child is None


def _try_split(node: _BSPNode, rng: np.random.Generator,
               min_h: int, min_w: int) -> bool:
    """Split *node* along a randomly chosen axis.

    The shared wall becomes the border of both children — no space is wasted.
    Returns True if a split was made.
    """
    can_h = node.h >= 2 * min_h
    can_v = node.w >= 2 * min_w
    if not can_h and not can_v:
        return False

    if can_h and can_v:
        do_h = bool(rng.integers(2))
    else:
        do_h = can_h

    if do_h:
        # Shared wall is a row; both children span the full column range.
        # top child rows  : [top, sp],      height = sp - top + 1  >= min_h
        # bottom child rows: [sp, top+h-1], height = top+h-1-sp+1  >= min_h
        lo = node.top + min_h - 1
        hi = node.top + node.h - min_h
        sp = int(rng.integers(lo, hi + 1))
        node.left_child  = _BSPNode(node.top, node.left,
                                    sp - node.top + 1, node.w)
        node.right_child = _BSPNode(sp, node.left,
                                    (node.top + node.h - 1) - sp + 1, node.w)
        node.split_horizontal = True
    else:
        # Shared wall is a column; both children span the full row range.
        lo = node.left + min_w - 1
        hi = node.left + node.w - min_w
        sp = int(rng.integers(lo, hi + 1))
        node.left_child  = _BSPNode(node.top, node.left,
                                    node.h, sp - node.left + 1)
        node.right_child = _BSPNode(node.top, sp,
                                    node.h, (node.left + node.w - 1) - sp + 1)
        node.split_horizontal = False

    node.split_pos = sp
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_world(
    n_rooms: int,
    distractor: bool = False,
    seed: Optional[int] = None,
    room_h_range: Tuple[int, int] = (5, 11),
    room_w_range: Tuple[int, int] = (5, 15),
) -> ModularLayout:
    """Generate a random N-room :class:`~gridworld_env.modular_layout.ModularLayout`
    using Binary Space Partitioning.

    Parameters
    ----------
    n_rooms:
        Number of rooms (1–26).  Each room contains one safe.
    distractor:
        If True, rooms may contain an extra irrelevant element.
    seed:
        RNG seed for reproducibility.
    room_h_range:
        ``(min, max)`` inclusive range for room heights (including border walls).
        Minimum value is 5.
    room_w_range:
        ``(min, max)`` inclusive range for room widths (including border walls).
        Minimum value is 5.

    Returns
    -------
    ModularLayout
        Ready to pass to ``ModularMazeEnv(layout=...)``.
    """
    if not 1 <= n_rooms <= 26:
        raise ValueError(f"n_rooms must be between 1 and 26, got {n_rooms}.")
    min_h, max_h = room_h_range
    min_w, max_w = room_w_range
    if min_h < 5 or min_w < 5:
        raise ValueError("Minimum room dimension is 5.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Canvas size: sized so that N rooms of average size fit comfortably.
    #    Rooms share walls, so stride = size - 1.
    # ------------------------------------------------------------------
    n_cols_g = math.ceil(math.sqrt(n_rooms))
    n_rows_g = math.ceil(n_rooms / n_cols_g)
    avg_h = (min_h + max_h + 1) // 2
    avg_w = (min_w + max_w + 1) // 2
    total_h = n_rows_g * (avg_h - 1) + 1
    total_w = n_cols_g * (avg_w - 1) + 1

    # ------------------------------------------------------------------
    # 2. BSP: split leaves until we have n_rooms.
    #    Prefer splitting larger rooms (area-weighted selection).
    # ------------------------------------------------------------------
    root = _BSPNode(0, 0, total_h, total_w)
    leaves: List[_BSPNode] = [root]

    for _ in range(n_rooms - 1):
        splittable = [l for l in leaves if l.h >= 2 * min_h or l.w >= 2 * min_w]
        if not splittable:
            break
        areas = np.array([l.h * l.w for l in splittable], dtype=float)
        idx = int(rng.choice(len(splittable), p=areas / areas.sum()))
        chosen = splittable[idx]
        _try_split(chosen, rng, min_h, min_w)
        leaves.remove(chosen)
        leaves.extend([chosen.left_child, chosen.right_child])

    n_rooms = len(leaves)

    # ------------------------------------------------------------------
    # 3. Build grid (all walls) and carve room interiors.
    # ------------------------------------------------------------------
    grid: List[List[bool]] = [[True] * total_w for _ in range(total_h)]
    room_interiors: Dict[int, List[Tuple[int, int]]] = {}
    room_cell_map: Dict[Tuple[int, int], int] = {}

    for i, leaf in enumerate(leaves):
        cells: List[Tuple[int, int]] = []
        for r in range(leaf.top + 1, leaf.top + leaf.h - 1):
            for c in range(leaf.left + 1, leaf.left + leaf.w - 1):
                grid[r][c] = False
                room_cell_map[(r, c)] = i
                cells.append((r, c))
        room_interiors[i] = cells

    # ------------------------------------------------------------------
    # 4. Place doors by traversing the BSP tree bottom-up.
    #
    #    At each internal node the two children share exactly one wall
    #    (a row for h-splits, a column for v-splits).  We pick a cell on
    #    that wall where BOTH neighbouring cells (one per child) are already
    #    carved, guaranteeing a passable door.
    # ------------------------------------------------------------------
    doors: List[SimpleDoor] = []
    door_positions: Set[Tuple[int, int]] = set()

    def _add_doors(node: _BSPNode) -> None:
        if node.is_leaf:
            return
        _add_doors(node.left_child)
        _add_doors(node.right_child)

        if node.split_horizontal:
            sr = node.split_pos  # shared wall row
            # Valid door columns: adjacent cells above AND below must be carved.
            valid = [
                c for c in range(node.left + 1, node.left + node.w - 1)
                if not grid[sr - 1][c] and not grid[sr + 1][c]
            ]
            if not valid:
                return
            door_pos = (sr, valid[int(rng.integers(len(valid)))])
        else:
            sc = node.split_pos  # shared wall column
            valid = [
                r for r in range(node.top + 1, node.top + node.h - 1)
                if not grid[r][sc - 1] and not grid[r][sc + 1]
            ]
            if not valid:
                return
            door_pos = (valid[int(rng.integers(len(valid)))], sc)

        grid[door_pos[0]][door_pos[1]] = False
        doors.append(SimpleDoor(position=door_pos))
        door_positions.add(door_pos)

    _add_doors(root)

    for dp in door_positions:
        room_cell_map.pop(dp, None)

    # ------------------------------------------------------------------
    # 5. Assign room types and safe materials.
    # ------------------------------------------------------------------
    base_materials = [_MATERIALS[i % len(_MATERIALS)] for i in range(n_rooms)]
    perm = rng.permutation(n_rooms)
    safe_materials: List[str] = [base_materials[i] for i in perm]
    room_types: List[int] = [int(rng.integers(1, 4)) for _ in range(n_rooms)]

    # ------------------------------------------------------------------
    # 6. Place objects in each room.
    # ------------------------------------------------------------------
    keys: List[ModularKey] = []
    safes: List[Safe] = []
    npcs: List[NPC] = []
    materials: List[Material] = []
    start_position: Optional[Tuple[int, int]] = None

    for room_idx in range(n_rooms):
        interior = list(room_interiors[room_idx])
        occupied: Set[Tuple[int, int]] = set()

        def pick(rng=rng, interior=interior, occupied=occupied) -> Tuple[int, int]:
            free = [p for p in interior if p not in occupied]
            pos = free[int(rng.integers(len(free)))]
            occupied.add(pos)
            return pos

        safe_mat = safe_materials[room_idx]
        rtype = room_types[room_idx]

        if room_idx == 0:
            start_position = pick()

        safes.append(Safe(
            id=str(room_idx + 1),
            position=pick(),
            unique_material=safe_mat,
            reward=1.0,
        ))

        if rtype == 1:
            keys.append(ModularKey(
                id=chr(ord("a") + room_idx),
                position=pick(),
                unique_material=safe_mat,
            ))
        elif rtype == 2:
            mat_info = MATERIAL_TYPES[safe_mat]
            materials.append(Material(
                name="ore",
                color=MATERIAL_TYPES["ore"]["color"],
                shape=MATERIAL_TYPES["ore"]["shape"],
                position=pick(),
            ))
            materials.append(Material(
                name=safe_mat,
                color=mat_info["color"],
                shape=mat_info["shape"],
                position=pick(),
            ))
        elif rtype == 3:
            npcs.append(NPC(
                npc_type="wizard",
                position=pick(),
                key_type=safe_mat,
            ))

        if distractor and rng.random() < 0.5:
            wrong_mat = _pick_other(rng, safe_mat)
            if rtype == 1:
                npcs.append(NPC(
                    npc_type="wizard",
                    position=pick(),
                    key_type=wrong_mat,
                ))
            elif rtype == 2:
                wm_info = MATERIAL_TYPES[wrong_mat]
                materials.append(Material(
                    name=wrong_mat,
                    color=wm_info["color"],
                    shape=wm_info["shape"],
                    position=pick(),
                ))
            elif rtype == 3:
                keys.append(ModularKey(
                    id=f"distractor_{room_idx}",
                    position=pick(),
                    unique_material=wrong_mat,
                ))

    assert start_position is not None

    return ModularLayout(
        grid=grid,
        width=total_w,
        height=total_h,
        start_position=start_position,
        keys=keys,
        safes=safes,
        doors=doors,
        npcs=npcs,
        materials=materials,
        room_cell_map=room_cell_map,
    )


# ---------------------------------------------------------------------------
# Grid-based procgen (equal-sized rooms)
# ---------------------------------------------------------------------------

def generate_world_grid(
    n_rooms: int,
    distractor: bool = False,
    seed: Optional[int] = None,
    room_h: int = 9,
    room_w: int = 11,
) -> ModularLayout:
    """Generate a random N-room :class:`~gridworld_env.modular_layout.ModularLayout`
    using a regular grid of equal-sized rooms.

    All rooms share the same ``room_h × room_w`` bounding box (including border
    walls), so the layout is compatible with ``obs_mode='room_pixels'`` and
    ``global_map_mode='overlay'`` + ``--partial-obs``.

    Rooms are arranged in a roughly square grid; doors are placed on the shared
    wall between every pair of horizontally or vertically adjacent rooms.

    Parameters
    ----------
    n_rooms:
        Number of rooms (1–26).
    distractor:
        If True, rooms may contain an extra irrelevant element.
    seed:
        RNG seed for reproducibility.
    room_h:
        Height of each room in cells, including border walls (minimum 5).
    room_w:
        Width of each room in cells, including border walls (minimum 5).
    """
    if not 1 <= n_rooms <= 26:
        raise ValueError(f"n_rooms must be between 1 and 26, got {n_rooms}.")
    if room_h < 5 or room_w < 5:
        raise ValueError("Minimum room dimension is 5.")

    rng = np.random.default_rng(seed)

    n_cols = math.ceil(math.sqrt(n_rooms))
    n_rows = math.ceil(n_rooms / n_cols)

    total_h = n_rows * (room_h - 1) + 1
    total_w = n_cols * (room_w - 1) + 1

    # Build grid (all walls), then carve interiors.
    grid: List[List[bool]] = [[True] * total_w for _ in range(total_h)]
    room_interiors: Dict[int, List[Tuple[int, int]]] = {}
    room_cell_map: Dict[Tuple[int, int], int] = {}

    def _room_top_left(room_idx: int) -> Tuple[int, int]:
        ri, ci = divmod(room_idx, n_cols)
        return ri * (room_h - 1), ci * (room_w - 1)

    for i in range(n_rooms):
        top, left = _room_top_left(i)
        cells: List[Tuple[int, int]] = []
        for r in range(top + 1, top + room_h - 1):
            for c in range(left + 1, left + room_w - 1):
                grid[r][c] = False
                room_cell_map[(r, c)] = i
                cells.append((r, c))
        room_interiors[i] = cells

    # Place doors on shared walls between adjacent room pairs.
    doors: List[SimpleDoor] = []
    door_positions: Set[Tuple[int, int]] = set()

    for i in range(n_rooms):
        top_i, left_i = _room_top_left(i)
        ri, ci = divmod(i, n_cols)

        # Right neighbour (same row, next column).
        j = i + 1
        if ci + 1 < n_cols and j < n_rooms:
            shared_col = left_i + room_w - 1
            valid_rows = list(range(top_i + 1, top_i + room_h - 1))
            dr = valid_rows[int(rng.integers(len(valid_rows)))]
            grid[dr][shared_col] = False
            doors.append(SimpleDoor(position=(dr, shared_col)))
            door_positions.add((dr, shared_col))

        # Bottom neighbour (next row, same column).
        j = i + n_cols
        if ri + 1 < n_rows and j < n_rooms:
            shared_row = top_i + room_h - 1
            valid_cols = list(range(left_i + 1, left_i + room_w - 1))
            dc = valid_cols[int(rng.integers(len(valid_cols)))]
            grid[shared_row][dc] = False
            doors.append(SimpleDoor(position=(shared_row, dc)))
            door_positions.add((shared_row, dc))

    for dp in door_positions:
        room_cell_map.pop(dp, None)

    # Assign room types and objects (identical logic to BSP generator).
    base_materials = [_MATERIALS[i % len(_MATERIALS)] for i in range(n_rooms)]
    perm = rng.permutation(n_rooms)
    safe_materials: List[str] = [base_materials[i] for i in perm]
    room_types: List[int] = [int(rng.integers(1, 4)) for _ in range(n_rooms)]

    keys: List[ModularKey] = []
    safes: List[Safe] = []
    npcs: List[NPC] = []
    materials: List[Material] = []
    start_position: Optional[Tuple[int, int]] = None

    for room_idx in range(n_rooms):
        interior = list(room_interiors[room_idx])
        occupied: Set[Tuple[int, int]] = set()

        def pick(rng=rng, interior=interior, occupied=occupied) -> Tuple[int, int]:
            free = [p for p in interior if p not in occupied]
            pos = free[int(rng.integers(len(free)))]
            occupied.add(pos)
            return pos

        safe_mat = safe_materials[room_idx]
        rtype = room_types[room_idx]

        if room_idx == 0:
            start_position = pick()

        safes.append(Safe(
            id=str(room_idx + 1),
            position=pick(),
            unique_material=safe_mat,
            reward=1.0,
        ))

        if rtype == 1:
            keys.append(ModularKey(
                id=chr(ord("a") + room_idx),
                position=pick(),
                unique_material=safe_mat,
            ))
        elif rtype == 2:
            mat_info = MATERIAL_TYPES[safe_mat]
            materials.append(Material(
                name="ore",
                color=MATERIAL_TYPES["ore"]["color"],
                shape=MATERIAL_TYPES["ore"]["shape"],
                position=pick(),
            ))
            materials.append(Material(
                name=safe_mat,
                color=mat_info["color"],
                shape=mat_info["shape"],
                position=pick(),
            ))
        elif rtype == 3:
            npcs.append(NPC(
                npc_type="wizard",
                position=pick(),
                key_type=safe_mat,
            ))

        if distractor and rng.random() < 0.5:
            wrong_mat = _pick_other(rng, safe_mat)
            if rtype == 1:
                npcs.append(NPC(
                    npc_type="wizard",
                    position=pick(),
                    key_type=wrong_mat,
                ))
            elif rtype == 2:
                wm_info = MATERIAL_TYPES[wrong_mat]
                materials.append(Material(
                    name=wrong_mat,
                    color=wm_info["color"],
                    shape=wm_info["shape"],
                    position=pick(),
                ))
            elif rtype == 3:
                keys.append(ModularKey(
                    id=f"distractor_{room_idx}",
                    position=pick(),
                    unique_material=wrong_mat,
                ))

    assert start_position is not None

    return ModularLayout(
        grid=grid,
        width=total_w,
        height=total_h,
        start_position=start_position,
        keys=keys,
        safes=safes,
        doors=doors,
        npcs=npcs,
        materials=materials,
        room_cell_map=room_cell_map,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_other(rng: np.random.Generator, exclude: str) -> str:
    choices = [m for m in _MATERIALS if m != exclude]
    return choices[int(rng.integers(len(choices)))]
