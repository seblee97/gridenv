"""
Objects for the Modular Maze environment.

Keys (a-z) are collected into a persistent inventory and used to open safes.
Safes (1-9) have many-to-many relationships with keys and may contain hidden
rewards.  Simple doors block passage and open when the agent enters from an
adjacent cell (no key required).
"""

from dataclasses import dataclass
from typing import Set, Tuple


@dataclass
class ModularKey:
    """A collectible key identified by a single character (a-z).

    Attributes:
        id: Single-character key identifier (e.g. 'a', 'b').
        position: (row, col) grid position.
        unique_material: Name of the material associated with this key (e.g. 'diamond').
        collected: Whether the key has been picked up.
    """

    id: str
    position: Tuple[int, int]
    unique_material: str = ""
    collected: bool = False

    def __hash__(self) -> int:
        return hash((self.position, self.id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModularKey):
            return NotImplemented
        return self.position == other.position and self.id == other.id


@dataclass
class Safe:
    """A safe that stores a hidden reward and is opened by any key of the correct material type.

    Attributes:
        id: Single-character safe identifier (e.g. '1', '2').
        position: (row, col) grid position.
        unique_material: The material type a key must have to open this safe (e.g. 'diamond').
        reward: The reward value revealed when the safe is opened.
        opened: Whether the safe has already been opened.
    """

    id: str
    position: Tuple[int, int]
    unique_material: str = ""
    reward: float = 0.0
    opened: bool = False

    def can_open_with(self, key_materials: Set[str]) -> bool:
        """Return True if the agent holds at least one key whose unique_material matches this safe."""
        return self.unique_material in key_materials

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Safe):
            return NotImplemented
        return self.position == other.position and self.id == other.id


@dataclass
class Material:
    """A collectible material with a color, shape, and name.

    Attributes:
        name: Human-readable material name (e.g. 'diamond').
        color: Display color string (e.g. 'cyan').
        shape: Shape descriptor string (e.g. 'hexagon').
        position: (row, col) grid position.
        collected: Whether the material has been picked up.
    """

    name: str
    color: str
    shape: str
    position: Tuple[int, int]
    collected: bool = False

    def __hash__(self) -> int:
        return hash((self.position, self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Material):
            return NotImplemented
        return self.position == other.position and self.name == other.name


# Predefined material templates (position must be set when placing in a layout)
DIAMOND  = lambda position: Material(name="diamond",  color="cyan",  shape="hexagon",  position=position)
RUBY     = lambda position: Material(name="ruby",     color="red",   shape="octagon",  position=position)
SAPPHIRE = lambda position: Material(name="sapphire", color="blue",  shape="teardrop", position=position)
ORE      = lambda position: Material(name="ore",      color="gray",  shape="nugget",   position=position)

# Lookup table: material name → (color string, shape string)
MATERIAL_TYPES: dict = {
    "diamond":  {"color": "cyan",  "shape": "hexagon"},
    "ruby":     {"color": "red",   "shape": "octagon"},
    "sapphire": {"color": "blue",  "shape": "teardrop"},
    "ore":      {"color": "gray",  "shape": "nugget"},
}


@dataclass
class NPC:
    """A non-player character that the agent can engage with from an adjacent cell.

    Attributes:
        npc_type: Type identifier (e.g. 'wizard').
        position: (row, col) grid position.
        engaged: Whether the agent has engaged with this NPC.
        key_type: If set, engaging this NPC (for the first time) spawns a key
            with this unique_material in the same room.
    """

    npc_type: str
    position: Tuple[int, int]
    engaged: bool = False
    key_type: str = ""

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NPC):
            return NotImplemented
        return self.position == other.position


@dataclass
class SimpleDoor:
    """A door that blocks movement until the agent opens it.

    The door opens automatically when the agent attempts to move into the door
    cell from an adjacent cell.  No key is required.

    Attributes:
        position: (row, col) grid position.
        is_open: Whether the door has been opened.
    """

    position: Tuple[int, int]
    is_open: bool = False

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleDoor):
            return NotImplemented
        return self.position == other.position
