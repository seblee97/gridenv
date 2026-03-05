"""
Layout parsing for the Modular Maze environment.

ASCII Layout Format
-------------------
#       Wall
.       Empty floor
S       Agent start position
D       Door  (blocks passage; opens when agent steps into it from adjacent cell)
a-z     Key   (key ID = the lowercase letter, e.g. 'a', 'b', 'c')
1-9     Safe  (safe ID = the digit character, e.g. '1', '2')

Configuration (companion .json file)
-------------------------------------
{
  "safes": {
    "1": {"unlocked_by": ["a"],        "reward": 1.0},
    "2": {"unlocked_by": ["a", "b"],   "reward": 0.0},
    "3": {"unlocked_by": ["b"],        "reward": 2.0}
  }
}

- "unlocked_by" lists key IDs (any one suffices to open the safe).
- "reward" is the hidden value revealed when the safe is opened.
- Safes not mentioned in the config are created with no key requirement and
  zero reward (effectively always openable but empty).

Keys do not require config entries; they are identified solely by their ASCII
character.

Example
-------
###########
#S........#
#.........#
#..a...b..#
#.........#
#####D#####
#.........#
#..1...2..#
#.........#
###########
"""

import copy
import json
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from gridworld_env.modular_objects import ModularKey, Safe, SimpleDoor


@dataclass
class ModularLayout:
    """Parsed layout for a Modular Maze environment.

    Attributes:
        grid: 2D list where True = wall.
        width: Grid width in cells.
        height: Grid height in cells.
        start_position: Agent's (row, col) starting position.
        keys: All keys placed in the layout.
        safes: All safes placed in the layout.
        doors: All doors placed in the layout.
    """

    grid: List[List[bool]]
    width: int
    height: int
    start_position: Tuple[int, int]
    keys: List[ModularKey] = field(default_factory=list)
    safes: List[Safe] = field(default_factory=list)
    doors: List[SimpleDoor] = field(default_factory=list)
    room_cell_map: Optional[Dict[Tuple[int, int], int]] = field(
        default=None, repr=False
    )
    """Optional mapping from floor cell (row, col) to declared room index.

    Set by :func:`~gridworld_env.world_layout.assemble_world` when a world
    is loaded from multiple room files.  When present,
    :class:`~gridworld_env.modular_maze.ModularMazeEnv` uses these declared
    indices instead of inferring room membership via BFS.
    """

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_wall(self, row: int, col: int) -> bool:
        """Return True if (row, col) is out-of-bounds or a wall."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return self.grid[row][col]

    def is_valid_position(self, row: int, col: int) -> bool:
        """Return True if (row, col) is in-bounds and not a wall."""
        return not self.is_wall(row, col)

    def get_door_at(self, row: int, col: int) -> Optional[SimpleDoor]:
        """Return the door at (row, col), or None."""
        for door in self.doors:
            if door.position == (row, col):
                return door
        return None

    def get_key_at(self, row: int, col: int) -> Optional[ModularKey]:
        """Return an uncollected key at (row, col), or None."""
        for key in self.keys:
            if key.position == (row, col) and not key.collected:
                return key
        return None

    def get_safe_at(self, row: int, col: int) -> Optional[Safe]:
        """Return an unopened safe at (row, col), or None."""
        for safe in self.safes:
            if safe.position == (row, col) and not safe.opened:
                return safe
        return None

    def copy(self) -> "ModularLayout":
        """Deep copy for episode isolation."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_modular_layout_string(
    layout_str: str,
    config: Optional[Dict] = None,
) -> ModularLayout:
    """Parse a Modular Maze layout from an ASCII string.

    Args:
        layout_str: ASCII grid, optionally followed by ``---`` and a JSON
            config block.
        config: Optional explicit config dict.  Merged with any embedded
            config; explicit values take precedence.

    Returns:
        Parsed :class:`ModularLayout`.

    Raises:
        ValueError: On unrecognised ASCII characters.
    """
    config = config or {}

    # Support embedded JSON config after a '---' separator
    if "---" in layout_str:
        layout_part, config_part = layout_str.split("---", 1)
        try:
            embedded = json.loads(config_part.strip())
            embedded.update(config)  # explicit config wins
            config = embedded
        except json.JSONDecodeError:
            pass
    else:
        layout_part = layout_str

    layout_part = textwrap.dedent(layout_part)
    lines = [line for line in layout_part.strip().split("\n") if line.strip()]

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    lines = [line.ljust(width) for line in lines]

    grid: List[List[bool]] = [[False] * width for _ in range(height)]
    start_position: Tuple[int, int] = (0, 0)
    keys: List[ModularKey] = []
    safes: List[Safe] = []
    doors: List[SimpleDoor] = []

    safe_configs: Dict[str, Dict] = config.get("safes", {})

    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            if char == "#":
                grid[row][col] = True
            elif char == "S":
                start_position = (row, col)
            elif char == "D":
                doors.append(SimpleDoor(position=(row, col)))
            elif char.islower() and char.isalpha():
                # Lowercase letter → key with that ID
                keys.append(ModularKey(id=char, position=(row, col)))
            elif char.isdigit() and char != "0":
                # Non-zero digit → safe with that ID
                safe_cfg = safe_configs.get(char, {})
                safes.append(Safe(
                    id=char,
                    position=(row, col),
                    unlocked_by=list(safe_cfg.get("unlocked_by", [])),
                    reward=float(safe_cfg.get("reward", 0.0)),
                ))
            elif char == ".":
                pass  # empty floor
            else:
                raise ValueError(
                    f"Unexpected character '{char}' at row {row}, col {col}. "
                    "Valid characters: #  .  S  D  a-z (keys)  1-9 (safes)"
                )

    return ModularLayout(
        grid=grid,
        width=width,
        height=height,
        start_position=start_position,
        keys=keys,
        safes=safes,
        doors=doors,
    )


def parse_modular_layout_file(filepath: Union[str, Path]) -> ModularLayout:
    """Load and parse a Modular Maze layout from a ``.txt`` file.

    Automatically loads a companion ``.json`` config file from the same
    directory if one exists.

    Args:
        filepath: Path to the ``.txt`` layout file.

    Returns:
        Parsed :class:`ModularLayout`.
    """
    filepath = Path(filepath)
    config_path = filepath.with_suffix(".json")
    config: Dict = {}

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        warnings.warn(
            f"No companion JSON config found at {config_path}. "
            "Safes will be created with empty unlocked_by lists and zero reward.",
            UserWarning,
            stacklevel=2,
        )

    with open(filepath) as f:
        content = f.read()

    return parse_modular_layout_string(content, config)
