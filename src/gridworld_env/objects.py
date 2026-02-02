"""
Game objects: Keys, Doors, and Rewards.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class KeyColor(Enum):
    """Available key colors."""
    RED = auto()
    BLUE = auto()

    @classmethod
    def from_string(cls, s: str) -> "KeyColor":
        """Parse key color from string."""
        mapping = {
            "r": cls.RED,
            "red": cls.RED,
            "b": cls.BLUE,
            "blue": cls.BLUE,
        }
        return mapping[s.lower()]

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Key:
    """A collectible key that can open doors."""
    position: Tuple[int, int]
    color: KeyColor
    collected: bool = False

    def __hash__(self) -> int:
        return hash((self.position, self.color))


@dataclass
class Door:
    """
    A door between rooms that requires a key to open.

    Attributes:
        position: The (row, col) position of the door.
        correct_key_color: The key color that preserves the reward in the next room.
        is_open: Whether the door has been opened.
        wrong_key_used: Whether the wrong key was used to open this door.
    """
    position: Tuple[int, int]
    correct_key_color: KeyColor
    is_open: bool = False
    wrong_key_used: bool = False

    def open_with_key(self, key_color: KeyColor) -> bool:
        """
        Attempt to open the door with a key.

        Returns True if successful, marks wrong_key_used if incorrect color.
        """
        self.is_open = True
        if key_color != self.correct_key_color:
            self.wrong_key_used = True
        return True

    def __hash__(self) -> int:
        return hash(self.position)


@dataclass
class Reward:
    """
    A reward that can be collected.

    Attributes:
        position: The (row, col) position of the reward.
        value: The reward value when collected.
        collected: Whether the reward has been collected.
        protected_by_door: Optional door that protects this reward.
        destroyed: Whether this reward was destroyed by wrong key choice.
    """
    position: Tuple[int, int]
    value: float = 1.0
    collected: bool = False
    protected_by_door: Optional[Door] = None
    destroyed: bool = False

    def is_available(self) -> bool:
        """Check if reward can still be collected."""
        return not self.collected and not self.destroyed

    def __hash__(self) -> int:
        return hash(self.position)


@dataclass
class KeyPair:
    """
    A pair of keys in a room - only one can be collected.

    Attributes:
        keys: Tuple of two keys (different colors).
        room_id: Identifier for the room containing these keys.
        collected_key: Which key was collected (None if neither).
        door: The door associated with this key pair (set during layout parsing).
    """
    keys: Tuple[Key, Key]
    room_id: int
    collected_key: Optional[Key] = None
    door: Optional["Door"] = None

    def collect(self, key: Key) -> bool:
        """
        Collect a key, making the other disappear.

        Returns True if collection was successful.
        """
        if self.collected_key is not None:
            return False
        if key not in self.keys:
            return False

        self.collected_key = key
        key.collected = True
        # Mark the other key as "collected" (disappeared)
        for k in self.keys:
            if k != key:
                k.collected = True
        return True

    def get_available_keys(self) -> list:
        """Get keys that haven't been collected yet."""
        if self.collected_key is not None:
            return []
        return [k for k in self.keys if not k.collected]
