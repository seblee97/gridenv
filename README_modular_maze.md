# Modular Maze Environment

A multi-room grid-world environment built on [Gymnasium](https://gymnasium.farama.org/).
The agent navigates connected rooms, collects keys and materials, opens safes, and
interacts with NPCs.  Worlds can be defined by hand-crafted ASCII layout files or
generated procedurally.

---

## Table of contents

1. [Concepts and objects](#1-concepts-and-objects)
2. [Actions](#2-actions)
3. [Opening safes](#3-opening-safes)
4. [Opening doors](#4-opening-doors)
5. [Room types (procgen)](#5-room-types-procgen)
6. [Observation modes](#6-observation-modes)
7. [Global map view](#7-global-map-view)
8. [Play mode](#8-play-mode)
9. [Layout file format](#9-layout-file-format)
10. [Multi-room world files](#10-multi-room-world-files)
11. [Python API](#11-python-api)

---

## 1. Concepts and objects

### Keys

Keys are collectible items identified by a lowercase letter (`a`–`z`).  Each key has
a **material type** (e.g. `diamond`, `ruby`, `sapphire`) which determines which safes
it can open.  Once collected, keys persist in the agent's inventory for the whole
episode and are **not consumed** when used.

### Materials

Raw materials are collectible items with a name, colour, and shape:

| Name       | Colour | Shape    | ASCII char |
|------------|--------|----------|------------|
| `diamond`  | cyan   | hexagon  | `V`        |
| `ruby`     | red    | octagon  | `R`        |
| `sapphire` | blue   | teardrop | `P`        |
| `ore`      | gray   | nugget   | `O`        |

Materials accumulate in a separate **material inventory**.  Ore combined with any
gem material can be **forged** into a key (see [actions](#2-actions)).

### Safes

Safes hold a hidden reward revealed when opened.  Each safe has a `unique_material`
property; to open it the agent must hold at least one key whose material type matches.
Safe cells are freely traversable — the agent does not need to stop on them first.

### Doors

Doors block passage between rooms.  They open automatically the first time the agent
takes the **USE_KEY** action while standing adjacent to the door.  No key is required
to open a door — proximity and the action are sufficient.  Once open, a door cell
is freely traversable for the rest of the episode.

### NPCs

Currently only the **wizard** type is implemented.  When the agent stands adjacent
to a wizard and takes the **ENGAGE** action:

- If the wizard has a `key_type` configured, a key of that material spawns at a
  random free position in the same room (first engagement only).
- Subsequent engagements with the same NPC have no further effect.

---

## 2. Actions

| ID | Name          | Key (play) | Effect |
|----|---------------|------------|--------|
| 0  | `UP`          | ↑ / W      | Move up one cell |
| 1  | `DOWN`        | ↓ / S      | Move down one cell |
| 2  | `LEFT`        | ← / A      | Move left one cell |
| 3  | `RIGHT`       | → / D      | Move right one cell |
| 4  | `USE_KEY`     | E          | Open an adjacent door, or open a safe the agent is standing on |
| 5  | `COLLECT_KEY` | F          | Pick up a key or material on the agent's current cell |
| 6  | `ENGAGE`      | G          | Interact with an adjacent NPC |
| 7  | `FORGE_KEY`   | Z          | Consume ore + a gem material from inventory to forge a key |

**Forge key rules:**
- Requires at least one `ore` and at least one gem material (`diamond`, `ruby`, or
  `sapphire`) in the material inventory.
- If multiple gem materials are held, the key is forged from the gem that was
  **collected first** (FIFO ordering).
- Both the ore and the gem material are consumed.  The resulting key goes directly
  into the key inventory (marked as collected) and can be used immediately.

---

## 3. Opening safes

```
safe.unique_material == "diamond"
  → needs any key with unique_material == "diamond"
```

Ways to obtain a diamond key (example):

| Method | How |
|--------|-----|
| **Key in room** | A diamond key (`V`-shaped gem) is lying in the room — collect it with F |
| **Forge it** | Collect ore (`O`) + diamond material (`V`), then press Z |
| **Wizard** | Engage (`G`) a wizard configured with `key_type: diamond` — a key spawns in the room |

---

## 4. Opening doors

Stand in any of the four cells adjacent to a closed door (`D`) and press **E**
(`USE_KEY`).  The door opens immediately.  No key is required or consumed.

---

## 5. Room types (procgen)

When generating worlds with `--procgen`, each room is randomly assigned one of three
types.  Every room contains one safe regardless of type.

### Type 1 — Key room
The correct key for the room's safe is placed directly in the room.  Collect it
with **F**, then press **E** on the safe.

### Type 2 — Forge room
The room contains:
- One **ore** nugget
- One **gem material** matching the safe's material type

The agent must collect both (F, F) and then forge a key (Z) to open the safe.

### Type 3 — Wizard room
A wizard NPC is placed in the room.  Engaging the wizard (G while adjacent) causes
a key of the correct type to spawn somewhere in the room.  Collect the key (F) and
use it (E).

### Distractor elements

When `--distractor` is enabled, each room has a 50% chance of also containing an
irrelevant element:

| Room type | Distractor added |
|-----------|-----------------|
| Type 1    | A wizard who spawns a key of the **wrong** material |
| Type 2    | An extra gem material of the **wrong** type |
| Type 3    | A key of the **wrong** material lying in the room |

---

## 6. Observation modes

Pass `obs_mode=` to `ModularMazeEnv(...)`.

### `symbolic` (default)

Flat 1-D array (or 3-D grid when `flatten_obs=False`) containing:
- **Grid**: one-hot cell-type encoding over the full map
- **Agent position**: normalised `(row, col)`
- **Key inventory**: binary vector, one bit per key slot

Cell type integers:

| Value | Cell type      |
|-------|----------------|
| 0     | Empty          |
| 1     | Wall           |
| 2     | Agent          |
| 3     | Key            |
| 4     | Safe (closed)  |
| 5     | Safe (open)    |
| 6     | Door (closed)  |
| 7     | Door (open)    |
| 8     | NPC            |
| 9     | NPC (engaged)  |
| 10    | Material       |

### `symbolic_minimal`

Compact 1-D vector with no grid — useful for larger maps:
- Normalised agent position
- Binary key inventory
- Per-key-instance availability bits
- Per-safe open/closed bits
- Per-door open/closed bits

### `pixels`

RGB array (`H × W × 3`, uint8) rendered with pygame (falls back to a pure-numpy
renderer if pygame is unavailable).  Returns the full map view.

### `room_pixels`

RGB array showing **only the room the agent is currently in** (partial
observability).  Triggered by `--partial-obs` in play mode; use `obs_mode="room_pixels"`
in code.

### `both`

Dict with two keys:
```python
obs["pixels"]   # full-map RGB array
obs["symbolic"] # flat symbolic vector
```

### `macro`

Room-level structural observation (no grid pixels):
- **Room adjacency matrix**: `N_rooms × N_rooms` binary matrix — entry `(i, j) = 1` if rooms `i` and `j` share a door
- **Current room**: integer index of the agent's current room

---

## 7. Global map view

An optional room-resolution minimap can be added on top of any observation mode via the `global_map_mode` parameter.  The map indicates **which room the agent is in** without revealing anything at the cell level.

### How it works

The map is a compact grid where each cell represents one room.  Room positions in the grid are inferred automatically from the spatial layout of room bounding boxes.  The agent's current room is highlighted in **cyan**; all other rooms are dark grey.

### `global_map_mode="overlay"`

The minimap is drawn directly in the **top-right corner** of the pixel observation.  The observation space shape is unchanged — the map is burned into the pixel image.

Requires `obs_mode` to be `"pixels"`, `"room_pixels"`, or `"both"`.

```python
env = ModularMazeEnv(layout, obs_mode="pixels", global_map_mode="overlay")
obs, _ = env.reset()   # obs is a normal (H, W, 3) array with map in corner
```

### `global_map_mode="image"`

The observation becomes a Dict with the primary observation and a **separate map image** sized to the room grid.  Intended for architectures that process the global map in a parallel stream.

```python
env = ModularMazeEnv(layout, obs_mode="pixels", global_map_mode="image")
obs, _ = env.reset()
obs["obs"]       # primary pixel observation, shape (H, W, 3)
obs["map_image"] # room-grid image, shape (map_rows*cs, map_cols*cs, 3)
```

### `global_map_mode="onehot"`

The observation becomes a Dict with the primary observation and a **one-hot room vector**.  Works with any `obs_mode`.

```python
env = ModularMazeEnv(layout, obs_mode="symbolic", global_map_mode="onehot")
obs, _ = env.reset()
obs["obs"]        # primary symbolic observation
obs["room_onehot"] # float32 vector, length = n_rooms, 1.0 at current room
```

### `map_cell_size` parameter

Controls the pixel size of each room square in the minimap for `"overlay"` and `"image"` modes (default `8`).

```python
env = ModularMazeEnv(layout, obs_mode="pixels", global_map_mode="image", map_cell_size=16)
```

---

## 8. Play mode

Run `play.py` from the project root.

### Loading a hand-crafted layout

```bash
python play.py src/gridworld_env/layouts/simple_room.txt
python play.py src/gridworld_env/layouts/level0/world.txt
python play.py src/gridworld_env/layouts/two_rooms.txt --cell-size 64
```

### Procedural generation

```bash
# 4 rooms, random seed
python play.py --procgen --n-rooms 4

# 6 rooms, reproducible seed
python play.py --procgen --n-rooms 6 --seed 42

# 8 rooms with distractors
python play.py --procgen --n-rooms 8 --distractor

# Partial observability (room-only view)
python play.py --procgen --n-rooms 6 --partial-obs
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `layout` | — | Path to a `.txt` layout or world file |
| `--procgen` | off | Generate world with BSP instead of loading a file |
| `--n-rooms N` | 4 | Number of rooms for procgen |
| `--distractor` | off | Add irrelevant elements to procgen rooms |
| `--seed S` | random | RNG seed for procgen |
| `--partial-obs` | off | Show only the current room |
| `--cell-size PX` | 48 | Pixel size per grid cell |
| `--fps N` | 60 | Frame rate cap |
| `--max-steps N` | unlimited | Episode length limit |
| `--step-reward R` | -0.01 | Reward per step |

### Controls

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move |
| E | Use key (open door / open safe) |
| F | Collect key or material |
| G | Engage adjacent NPC |
| Z | Forge key |
| R | Reset episode |
| ESC / Q | Quit |

The sidebar shows the current episode number, step count, score, key inventory,
material inventory, and safe progress.

---

## 9. Layout file format

A single-room layout is a plain-text ASCII grid:

```
###########
#S........#
#.........#
#..a......#
#.........#
#####D#####
#.........#
#..1......#
#.........#
###########
```

| Character | Meaning |
|-----------|---------|
| `#`       | Wall |
| `.`       | Empty floor |
| `S`       | Agent start position |
| `D`       | Door (opens on USE_KEY from adjacent cell) |
| `a`–`z`   | Key (ID = the letter) |
| `1`–`9`   | Safe (ID = the digit) |
| `W`       | Wizard NPC |
| `V`       | Diamond material |
| `R`       | Ruby material |
| `P`       | Sapphire material |
| `O`       | Ore material |

A companion `.json` file (same name, `.json` extension) configures safes and NPCs:

```json
{
  "safes": {
    "1": { "unique_material": "diamond", "reward": 1.0 },
    "2": { "unique_material": "ruby",    "reward": 2.0 }
  },
  "npcs": {
    "wizard": { "key_type": "diamond" }
  }
}
```

Safes without a config entry are openable by any key and give zero reward.

---

## 10. Multi-room world files

A world is a directory containing:

```
world.txt        # topology file listing rooms and connections
world.json       # global config (safes, NPCs)
room_a.txt       # individual room ASCII layouts
room_b.txt
...
```

`world.txt` format:

```
rooms
room_start  room_a.txt
room_north  room_b.txt

connections
room_start north room_north
```

Each `connections` entry specifies `<room_id> <direction> <room_id>`, where direction
is one of `north`, `south`, `east`, `west`.  Rooms are joined by a door cell placed
on the shared border wall.

---

## 11. Python API

```python
from gridworld_env.modular_maze import ModularMazeEnv

# From a layout file
env = ModularMazeEnv(
    layout="src/gridworld_env/layouts/simple_room.txt",
    obs_mode="pixels",          # "symbolic" | "symbolic_minimal" | "pixels" | "both"
    render_mode="rgb_array",
    max_steps=500,
    step_reward=-0.01,
)

# Procedurally generated world
from gridworld_env.procgen import generate_world

layout = generate_world(
    n_rooms=6,
    distractor=True,
    seed=42,
    room_h_range=(5, 11),   # room height range (inclusive, min 5)
    room_w_range=(5, 15),   # room width range  (inclusive, min 5)
)
env = ModularMazeEnv(layout=layout, obs_mode="pixels", render_mode="rgb_array")

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Info dict keys (ModularMazeEnv)
info["inventory"]      # list of key IDs currently held
info["materials"]      # list of material names currently held
info["safes_opened"]   # int: number of safes opened this episode
info["safes_total"]    # int: total safes in the layout
info["score"]          # float: cumulative reward
```

### `ModularMazeEnv` constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layout` | required | `ModularLayout`, file path, or ASCII string |
| `max_steps` | `None` | Episode length limit (`None` = unlimited) |
| `step_reward` | `-0.01` | Reward added each step |
| `collision_reward` | `-0.1` | Reward for walking into a wall |
| `flatten_obs` | `True` | Flatten symbolic observations to 1-D |
| `obs_mode` | `"symbolic"` | Observation mode (see [§6](#6-observation-modes)) |
| `render_mode` | `None` | `"human"`, `"rgb_array"`, or `None` |
| `start_pos_mode` | `"fixed"` | `"fixed"` or `"random_in_room"` |
| `terminate_on_all_safes_opened` | `True` | End episode when all safes are opened |
| `global_map_mode` | `None` | `"overlay"`, `"image"`, `"onehot"`, or `None` (see [§7](#7-global-map-view)) |
| `map_cell_size` | `8` | Pixel size per room cell in the minimap |
