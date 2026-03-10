"""
World layout — multiple room files stitched together into one ModularLayout.

Room file format
----------------
Same as the single-file ModularLayout ASCII format, with one addition:

    +   Connector on a border (wall) cell: marks where a door will be placed
        when this room is connected to another room.  Only valid on the
        outermost row or column of the room grid.

Characters:
    #       Wall
    .       Empty floor
    S       Agent start position  (exactly one room must contain S)
    D       (not used in room files; doors are created by connections)
    a-z     Key   (key ID = the lowercase letter)
    1-9     Safe  (safe ID = the digit)
    +       Connection point on a border wall

World file format
-----------------

    rooms
    0 room_lobby.txt
    1 room_north.txt
    2 room_east.txt

    connections
    0.south -> 1.north
    0.east -> 2.west

Blank lines and lines starting with '#' are ignored.
Paths in the rooms section are resolved relative to the world file.

Global JSON companion (world.json, auto-loaded)
-----------------------------------------------
Same format as the single-layout .json companion:

    {
      "safes": {
        "1": {"unlocked_by": ["a"], "reward": 1.0}
      }
    }

Assembly
--------
Rooms share a border column (east-west connections) or border row
(north-south connections).  The '+' connector cell in each room file
determines the exact alignment.  After assembly the resulting grid has:

  * The interior of each room copied to the global grid.
  * A SimpleDoor object placed at each connection point.
  * room_cell_map set on the returned ModularLayout so ModularMazeEnv
    can use declared room indices instead of inferring them via BFS.
"""

import json
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from gridworld_env.modular_layout import ModularLayout
from gridworld_env.modular_objects import MATERIAL_TYPES, Material, ModularKey, NPC, Safe, SimpleDoor


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoomSpec:
    """A single parsed room file."""

    room_id: int
    grid: List[List[bool]]          # True = wall
    height: int
    width: int
    start_position: Optional[Tuple[int, int]]  # None if no 'S' in this room
    keys: List[Tuple[str, int, int]]           # (id, row, col) within room
    safes: List[Tuple[str, int, int]]          # (id, row, col) within room
    npcs: List[Tuple[str, int, int]]           # (npc_type, row, col) within room
    materials: List[Tuple[str, int, int]]      # (mat_type, row, col) within room
    connectors: Dict[str, int]
    """Mapping from side name to offset along that side.

    * ``'north'`` / ``'south'``: column index of the '+' in that border row.
    * ``'east'``  / ``'west'``:  row index   of the '+' in that border column.
    """


@dataclass
class Connection:
    """A directed edge declaring how two rooms are joined."""

    room_a: int
    side_a: str   # 'north' | 'south' | 'east' | 'west'
    room_b: int
    side_b: str


@dataclass
class WorldSpec:
    """Parsed world topology (before assembly)."""

    room_files: Dict[int, Path]    # room_id -> absolute path to room .txt
    connections: List[Connection]
    base_dir: Path                 # directory containing the world file


# ---------------------------------------------------------------------------
# Room file parser
# ---------------------------------------------------------------------------

_VALID_SIDES = ("north", "south", "east", "west")


def parse_room_file(path: Union[str, Path], room_id: int) -> RoomSpec:
    """Parse a single room ASCII file and return a :class:`RoomSpec`.

    Parameters
    ----------
    path:
        Path to the room ``.txt`` file.
    room_id:
        The integer ID declared for this room in the world file.
    """
    path = Path(path)
    with open(path) as f:
        content = f.read()

    content = textwrap.dedent(content)
    lines = [line for line in content.strip().split("\n") if line.strip()]
    if not lines:
        raise ValueError(f"Room file '{path}' is empty.")

    height = len(lines)
    width = max(len(line) for line in lines)
    lines = [line.ljust(width) for line in lines]

    grid: List[List[bool]] = [[False] * width for _ in range(height)]
    start_position: Optional[Tuple[int, int]] = None
    keys: List[Tuple[str, int, int]] = []
    safes: List[Tuple[str, int, int]] = []
    npcs: List[Tuple[str, int, int]] = []
    materials: List[Tuple[str, int, int]] = []   # (instance_char, row, col)
    connectors: Dict[str, int] = {}

    _NPC_CHARS: Dict[str, str] = {"W": "wizard"}
    _MATERIAL_CHARS: Dict[str, str] = {"V": "diamond", "R": "ruby", "P": "sapphire", "O": "ore"}

    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            on_border = (
                row == 0 or row == height - 1 or col == 0 or col == width - 1
            )

            if char == "#":
                grid[row][col] = True

            elif char == "+":
                if not on_border:
                    raise ValueError(
                        f"Room '{path}': '+' at ({row}, {col}) is not on the border."
                    )
                # Determine side (priority: north > south > west > east for corners)
                if row == 0:
                    side, offset = "north", col
                elif row == height - 1:
                    side, offset = "south", col
                elif col == 0:
                    side, offset = "west", row
                else:
                    side, offset = "east", row

                if side in connectors:
                    raise ValueError(
                        f"Room '{path}': multiple '+' connectors on the {side} side."
                    )
                connectors[side] = offset
                # '+' is a floor cell — grid[row][col] stays False

            elif char == "S":
                start_position = (row, col)

            elif char.islower() and char.isalpha():
                keys.append((char, row, col))

            elif char.isdigit() and char != "0":
                safes.append((char, row, col))

            elif char in _NPC_CHARS:
                npcs.append((_NPC_CHARS[char], row, col))

            elif char in _MATERIAL_CHARS:
                materials.append((_MATERIAL_CHARS[char], row, col))

            elif char == ".":
                pass  # empty floor

            else:
                raise ValueError(
                    f"Room '{path}': unexpected character '{char}' at ({row}, {col}). "
                    "Valid: # . S + "
                    "W (wizard NPC)  V (diamond)  R (ruby)  P (sapphire)  O (ore)  "
                    "a-z (keys)  1-9 (safes)"
                )

    return RoomSpec(
        room_id=room_id,
        grid=grid,
        height=height,
        width=width,
        start_position=start_position,
        keys=keys,
        safes=safes,
        npcs=npcs,
        materials=materials,
        connectors=connectors,
    )


# ---------------------------------------------------------------------------
# World file parser
# ---------------------------------------------------------------------------

def parse_world_file(path: Union[str, Path]) -> WorldSpec:
    """Parse a world topology ``.txt`` file.

    Parameters
    ----------
    path:
        Path to the world file.

    Returns
    -------
    :class:`WorldSpec`
    """
    path = Path(path)
    base_dir = path.parent

    with open(path) as f:
        lines = f.readlines()

    section: Optional[str] = None
    room_files: Dict[int, Path] = {}
    connections: List[Connection] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "rooms":
            section = "rooms"
            continue
        if line == "connections":
            section = "connections"
            continue

        if section == "rooms":
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"World file: invalid rooms entry: '{line}'")
            room_id = int(parts[0])
            room_path = base_dir / parts[1].strip()
            room_files[room_id] = room_path

        elif section == "connections":
            # Format: "A.side -> B.side"  (spaces optional)
            line_nospace = line.replace(" ", "")
            if "->" not in line_nospace:
                raise ValueError(f"World file: invalid connection line: '{line}'")
            left, right = line_nospace.split("->", 1)
            if "." not in left or "." not in right:
                raise ValueError(f"World file: invalid connection line: '{line}'")
            str_a, side_a = left.rsplit(".", 1)
            str_b, side_b = right.rsplit(".", 1)
            if side_a not in _VALID_SIDES or side_b not in _VALID_SIDES:
                raise ValueError(
                    f"World file: unknown side in '{line}'. "
                    f"Valid: {_VALID_SIDES}"
                )
            connections.append(Connection(
                room_a=int(str_a), side_a=side_a,
                room_b=int(str_b), side_b=side_b,
            ))

    if not room_files:
        raise ValueError("World file: no rooms defined.")

    return WorldSpec(
        room_files=room_files,
        connections=connections,
        base_dir=base_dir,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_world(
    world_spec: WorldSpec,
    json_config: Optional[Dict] = None,
) -> ModularLayout:
    """Stitch room specs into a single :class:`ModularLayout`.

    Parameters
    ----------
    world_spec:
        Parsed world topology.
    json_config:
        Optional global safe configuration dict (same structure as the
        companion ``.json`` for single-layout files).

    Returns
    -------
    :class:`ModularLayout` with ``room_cell_map`` populated.
    """
    json_config = json_config or {}
    key_configs: Dict[str, Dict] = json_config.get("keys", {})
    safe_configs: Dict[str, Dict] = json_config.get("safes", {})
    npc_configs: Dict[str, Dict] = json_config.get("npcs", {})  # keyed by npc_type

    # --- Parse all room files -------------------------------------------
    rooms: Dict[int, RoomSpec] = {
        rid: parse_room_file(path, rid)
        for rid, path in world_spec.room_files.items()
    }

    # --- Find the start room (must contain 'S') --------------------------
    start_room_id: Optional[int] = None
    for rid, room in rooms.items():
        if room.start_position is not None:
            start_room_id = rid
            break
    if start_room_id is None:
        raise ValueError("No room file contains a start position 'S'.")

    # --- BFS to assign global (row, col) top-left to each room ----------
    # Rooms share one border column (east-west) or one border row (north-south).
    global_top_left: Dict[int, Tuple[int, int]] = {start_room_id: (0, 0)}
    visited_rooms: Set[int] = {start_room_id}
    queue: deque = deque([start_room_id])

    while queue:
        cur_id = queue.popleft()
        cur = rooms[cur_id]
        rCur, cCur = global_top_left[cur_id]

        for conn in world_spec.connections:
            # Determine which end of the connection is cur_id
            if conn.room_a == cur_id:
                nb_id = conn.room_b
                side_cur = conn.side_a
                side_nb = conn.side_b
            elif conn.room_b == cur_id:
                nb_id = conn.room_a
                side_cur = conn.side_b
                side_nb = conn.side_a
            else:
                continue

            if nb_id in visited_rooms:
                continue

            nb = rooms[nb_id]
            rNb, cNb = _place_room(
                cur, rCur, cCur, side_cur,
                nb, side_nb,
            )
            global_top_left[nb_id] = (rNb, cNb)
            visited_rooms.add(nb_id)
            queue.append(nb_id)

    unreachable = set(rooms.keys()) - visited_rooms
    if unreachable:
        raise ValueError(
            f"Rooms {sorted(unreachable)} are not reachable from start room "
            f"{start_room_id} via the declared connections."
        )

    # --- Normalize: shift so all top-lefts are non-negative --------------
    min_row = min(r for r, _ in global_top_left.values())
    min_col = min(c for _, c in global_top_left.values())
    global_top_left = {
        rid: (r - min_row, c - min_col)
        for rid, (r, c) in global_top_left.items()
    }

    total_height = max(r + rooms[rid].height for rid, (r, _) in global_top_left.items())
    total_width  = max(c + rooms[rid].width  for rid, (_, c) in global_top_left.items())

    # --- Build combined grid (all walls initially) -----------------------
    combined_grid: List[List[bool]] = [
        [True] * total_width for _ in range(total_height)
    ]
    room_cell_map: Dict[Tuple[int, int], int] = {}

    for rid, room in rooms.items():
        rTop, cTop = global_top_left[rid]
        for row in range(room.height):
            for col in range(room.width):
                if not room.grid[row][col]:   # floor (includes '+')
                    gr, gc = rTop + row, cTop + col
                    combined_grid[gr][gc] = False
                    room_cell_map[(gr, gc)] = rid

    # --- Compute door positions and remove them from room_cell_map -------
    doors: List[SimpleDoor] = []
    for conn in world_spec.connections:
        a = rooms[conn.room_a]
        rA, cA = global_top_left[conn.room_a]
        door_pos = _door_position(a, rA, cA, conn.side_a)
        doors.append(SimpleDoor(position=door_pos))
        room_cell_map.pop(door_pos, None)   # door cells belong to no room

    # --- Collect keys, safes, NPCs, and materials with global coordinates -
    all_keys: List[ModularKey] = []
    all_safes: List[Safe] = []
    all_npcs: List[NPC] = []
    all_materials: List[Material] = []

    for rid, room in rooms.items():
        rTop, cTop = global_top_left[rid]
        for (kid, kr, kc) in room.keys:
            key_cfg = key_configs.get(kid, {})
            all_keys.append(ModularKey(
                id=kid,
                position=(rTop + kr, cTop + kc),
                unique_material=key_cfg.get("unique_material", ""),
            ))
        for (sid, sr, sc) in room.safes:
            cfg = safe_configs.get(sid, {})
            all_safes.append(Safe(
                id=sid,
                position=(rTop + sr, cTop + sc),
                unique_material=cfg.get("unique_material", ""),
                reward=float(cfg.get("reward", 0.0)),
            ))
        for (npc_type, nr, nc) in room.npcs:
            npc_cfg = npc_configs.get(npc_type, {})
            all_npcs.append(NPC(
                npc_type=npc_type,
                position=(rTop + nr, cTop + nc),
                key_type=npc_cfg.get("key_type", ""),
            ))
        for (mat_type, mr, mc) in room.materials:
            mat_info = MATERIAL_TYPES.get(mat_type, {"color": "", "shape": ""})
            all_materials.append(Material(
                name=mat_type,
                color=mat_info["color"],
                shape=mat_info["shape"],
                position=(rTop + mr, cTop + mc),
            ))

    # --- Global start position -------------------------------------------
    sr, sc = rooms[start_room_id].start_position
    rS, cS = global_top_left[start_room_id]
    global_start = (rS + sr, cS + sc)

    return ModularLayout(
        grid=combined_grid,
        width=total_width,
        height=total_height,
        start_position=global_start,
        keys=all_keys,
        safes=all_safes,
        doors=doors,
        npcs=all_npcs,
        materials=all_materials,
        room_cell_map=room_cell_map,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _place_room(
    a: RoomSpec, rA: int, cA: int, side_a: str,
    b: RoomSpec, side_b: str,
) -> Tuple[int, int]:
    """Compute the global top-left (row, col) of room B given that A's
    ``side_a`` wall connects to B's ``side_b`` wall."""

    if side_a == "east" and side_b == "west":
        rA_conn = _get_connector(a, "east")
        rB_conn = _get_connector(b, "west")
        return (rA + rA_conn - rB_conn, cA + a.width - 1)

    elif side_a == "west" and side_b == "east":
        rA_conn = _get_connector(a, "west")
        rB_conn = _get_connector(b, "east")
        return (rA + rA_conn - rB_conn, cA - b.width + 1)

    elif side_a == "south" and side_b == "north":
        cA_conn = _get_connector(a, "south")
        cB_conn = _get_connector(b, "north")
        return (rA + a.height - 1, cA + cA_conn - cB_conn)

    elif side_a == "north" and side_b == "south":
        cA_conn = _get_connector(a, "north")
        cB_conn = _get_connector(b, "south")
        return (rA - b.height + 1, cA + cA_conn - cB_conn)

    else:
        raise ValueError(
            f"Incompatible connection sides: {side_a!r} -> {side_b!r}. "
            "Valid pairs: east<->west, north<->south."
        )


def _door_position(
    a: RoomSpec, rA: int, cA: int, side_a: str
) -> Tuple[int, int]:
    """Global position of the door on room A's specified side."""
    if side_a == "east":
        return (rA + _get_connector(a, "east"), cA + a.width - 1)
    elif side_a == "west":
        return (rA + _get_connector(a, "west"), cA)
    elif side_a == "south":
        return (rA + a.height - 1, cA + _get_connector(a, "south"))
    elif side_a == "north":
        return (rA, cA + _get_connector(a, "north"))
    else:
        raise ValueError(f"Unknown side: {side_a!r}")


def _get_connector(room: RoomSpec, side: str) -> int:
    """Return the connector offset for ``side``, raising if absent."""
    if side not in room.connectors:
        raise ValueError(
            f"Room {room.room_id} has no connector on its {side!r} side."
        )
    return room.connectors[side]


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def parse_world_layout_file(path: Union[str, Path]) -> ModularLayout:
    """Load a world from a ``.txt`` file and its companion ``.json``.

    The companion JSON is read from the same directory with the same stem
    and a ``.json`` suffix.  If it does not exist no warning is issued
    (safes will have empty ``unlocked_by`` lists and zero reward).

    Parameters
    ----------
    path:
        Path to the world ``.txt`` file.

    Returns
    -------
    Assembled :class:`ModularLayout`.
    """
    path = Path(path)
    json_config: Dict = {}
    config_path = path.with_suffix(".json")
    if config_path.exists():
        with open(config_path) as f:
            json_config = json.load(f)

    world_spec = parse_world_file(path)
    return assemble_world(world_spec, json_config)
