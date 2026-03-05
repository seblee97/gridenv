"""
Objects for the Modular Maze environment.

Keys (a-z) are collected into a persistent inventory and used to open safes.
Safes (1-9) have many-to-many relationships with keys and may contain hidden
rewards.  Simple doors block passage and open when the agent enters from an
adjacent cell (no key required).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class ModularKey:
    """A collectible key identified by a single character (a-z).

    Attributes:
        id: Single-character key identifier (e.g. 'a', 'b').
        position: (row, col) grid position.
        collected: Whether the key has been picked up.
    """

    id: str
    position: Tuple[int, int]
    collected: bool = False

    def __hash__(self) -> int:
        return hash((self.position, self.id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModularKey):
            return NotImplemented
        return self.position == other.position and self.id == other.id


@dataclass
class Safe:
    """A safe that stores a hidden reward and is opened by specific keys.

    Attributes:
        id: Single-character safe identifier (e.g. '1', '2').
        position: (row, col) grid position.
        unlocked_by: List of key IDs (any one of them suffices to open this safe).
        reward: The reward value revealed when the safe is opened.
        opened: Whether the safe has already been opened.
    """

    id: str
    position: Tuple[int, int]
    unlocked_by: List[str] = field(default_factory=list)
    reward: float = 0.0
    opened: bool = False

    def can_open_with(self, inventory: Set[str]) -> bool:
        """Return True if the inventory contains at least one key that opens this safe."""
        return bool(set(self.unlocked_by) & inventory)

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Safe):
            return NotImplemented
        return self.position == other.position and self.id == other.id


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
