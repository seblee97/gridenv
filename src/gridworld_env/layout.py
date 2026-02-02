"""
Layout parsing and representation for grid worlds.

ASCII Layout Format:
    # - Wall
    . - Empty floor
    S - Start position
    G - Goal/Reward
    R - Reward (with optional value suffix like R10 for value 10)
    D - Door (requires annotation for correct key color)
    r - Red key
    b - Blue key
    K - Key pair location (both colors, specified in config)

Layouts can be loaded from files or strings.

Example layout file (simple_room.txt):
    #######
    #S....#
    #.....#
    #..r..#
    #..b..#
    ###D###
    #..G..#
    #######

Configuration can be provided as YAML/JSON after the layout separated by ---
"""

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from gridworld_env.objects import Door, Key, KeyColor, KeyPair, Reward


@dataclass
class Layout:
    """
    Represents a parsed grid world layout.

    Attributes:
        grid: 2D array where True indicates a wall.
        width: Width of the grid.
        height: Height of the grid.
        start_position: Starting position (row, col).
        rewards: List of rewards in the environment.
        keys: List of individual keys.
        key_pairs: List of key pairs (mutually exclusive keys).
        doors: List of doors.
        door_reward_mapping: Maps doors to rewards they protect.
    """
    grid: List[List[bool]]
    width: int
    height: int
    start_position: Tuple[int, int]
    rewards: List[Reward] = field(default_factory=list)
    keys: List[Key] = field(default_factory=list)
    key_pairs: List[KeyPair] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    door_reward_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(
        default_factory=dict
    )

    def is_wall(self, row: int, col: int) -> bool:
        """Check if position is a wall."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return self.grid[row][col]

    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is valid (in bounds and not a wall)."""
        return not self.is_wall(row, col)

    def get_door_at(self, row: int, col: int) -> Optional[Door]:
        """Get door at position if one exists."""
        for door in self.doors:
            if door.position == (row, col):
                return door
        return None

    def get_key_at(self, row: int, col: int) -> Optional[Key]:
        """Get available key at position if one exists."""
        for key in self.keys:
            if key.position == (row, col) and not key.collected:
                return key
        return None

    def get_reward_at(self, row: int, col: int) -> Optional[Reward]:
        """Get available reward at position if one exists."""
        for reward in self.rewards:
            if reward.position == (row, col) and reward.is_available():
                return reward
        return None

    def get_key_pair_at(self, row: int, col: int) -> Optional[Tuple[KeyPair, Key]]:
        """Get key pair and specific key at position if available."""
        for key_pair in self.key_pairs:
            for key in key_pair.get_available_keys():
                if key.position == (row, col):
                    return (key_pair, key)
        return None

    def copy(self) -> "Layout":
        """Create a deep copy of the layout for episode reset."""
        import copy
        return copy.deepcopy(self)


def parse_layout_string(
    layout_str: str,
    config: Optional[Dict] = None
) -> Layout:
    """
    Parse a layout from an ASCII string.

    Args:
        layout_str: ASCII representation of the layout.
        config: Optional configuration dict with:
            - door_colors: Dict mapping door positions to correct key colors
            - reward_values: Dict mapping reward positions to values
            - protected_rewards: Dict mapping door positions to reward positions they protect

    Returns:
        Parsed Layout object.
    """
    config = config or {}

    # Split layout from embedded config if present
    if "---" in layout_str:
        layout_part, config_part = layout_str.split("---", 1)
        try:
            embedded_config = json.loads(config_part.strip())
            # Merge configs, explicit config takes precedence
            embedded_config.update(config)
            config = embedded_config
        except json.JSONDecodeError:
            pass
    else:
        layout_part = layout_str

    # Use textwrap.dedent to handle indented multiline strings
    layout_part = textwrap.dedent(layout_part)
    lines = [line for line in layout_part.strip().split("\n") if line.strip()]

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0

    # Pad lines to consistent width
    lines = [line.ljust(width) for line in lines]

    grid = [[False for _ in range(width)] for _ in range(height)]
    start_position = (0, 0)
    rewards = []
    keys = []
    doors = []
    key_positions = {}  # Track positions for key pair creation

    door_colors = config.get("door_colors", {})
    reward_values = config.get("reward_values", {})
    protected_rewards = config.get("protected_rewards", {})

    for row, line in enumerate(lines):
        col = 0
        while col < len(line):
            char = line[col]

            if char == "#":
                grid[row][col] = True
            elif char == "S":
                start_position = (row, col)
            elif char == "G":
                value = reward_values.get(f"{row},{col}", 1.0)
                rewards.append(Reward(position=(row, col), value=value))
            elif char == "R":
                # Check for value suffix (e.g., R10)
                value_str = ""
                while col + 1 < len(line) and line[col + 1].isdigit():
                    col += 1
                    value_str += line[col]
                value = float(value_str) if value_str else reward_values.get(
                    f"{row},{col - len(value_str)}", 1.0
                )
                pos = (row, col - len(value_str)) if value_str else (row, col)
                rewards.append(Reward(position=pos, value=value))
            elif char == "r":
                keys.append(Key(position=(row, col), color=KeyColor.RED))
                key_positions[(row, col)] = KeyColor.RED
            elif char == "b":
                keys.append(Key(position=(row, col), color=KeyColor.BLUE))
                key_positions[(row, col)] = KeyColor.BLUE
            elif char == "D":
                # Get correct color from config, default to RED
                pos_key = f"{row},{col}"
                color_str = door_colors.get(pos_key, "red")
                correct_color = KeyColor.from_string(color_str)
                doors.append(Door(position=(row, col), correct_key_color=correct_color))
            elif char == ".":
                pass  # Empty floor
            elif char == " ":
                grid[row][col] = True  # Treat space as wall

            col += 1

    # Build key pairs from adjacent keys of different colors
    key_pairs = _build_key_pairs(keys, config.get("key_pairs", []))

    # Build door-reward protection mapping
    door_reward_mapping = {}
    for door_pos_str, reward_positions in protected_rewards.items():
        door_pos = tuple(map(int, door_pos_str.split(",")))
        reward_pos_list = [
            tuple(map(int, pos.split(","))) for pos in reward_positions
        ]
        door_reward_mapping[door_pos] = reward_pos_list

    # Link rewards to their protecting doors
    for door in doors:
        protected_pos = door_reward_mapping.get(door.position, [])
        for reward in rewards:
            if reward.position in protected_pos:
                reward.protected_by_door = door

    # Link key pairs to their associated doors (sorted by grid position)
    sorted_doors = sorted(doors, key=lambda d: d.position)
    for kp in key_pairs:
        if kp.room_id < len(sorted_doors):
            kp.door = sorted_doors[kp.room_id]

    return Layout(
        grid=grid,
        width=width,
        height=height,
        start_position=start_position,
        rewards=rewards,
        keys=keys,
        key_pairs=key_pairs,
        doors=doors,
        door_reward_mapping=door_reward_mapping,
    )


def _build_key_pairs(
    keys: List[Key],
    pair_config: List[Dict]
) -> List[KeyPair]:
    """
    Build key pairs from configuration or proximity.

    Args:
        keys: List of all keys.
        pair_config: List of dicts with 'positions' key listing paired key positions.

    Returns:
        List of KeyPair objects.
    """
    key_pairs = []
    paired_keys = set()

    # First, process explicit pair configuration
    for pair_def in pair_config:
        positions = pair_def.get("positions", [])
        room_id = pair_def.get("room_id", len(key_pairs))

        pair_keys = []
        for pos_str in positions:
            pos = tuple(map(int, pos_str.split(",")))
            for key in keys:
                if key.position == pos and key not in paired_keys:
                    pair_keys.append(key)
                    paired_keys.add(key)
                    break

        if len(pair_keys) == 2:
            key_pairs.append(KeyPair(
                keys=(pair_keys[0], pair_keys[1]),
                room_id=room_id
            ))

    # Auto-pair remaining unpaired keys that are adjacent and different colors
    unpaired = [k for k in keys if k not in paired_keys]
    for i, key1 in enumerate(unpaired):
        if key1 in paired_keys:
            continue
        for key2 in unpaired[i + 1:]:
            if key2 in paired_keys:
                continue
            if key1.color != key2.color:
                # Check if adjacent (within 2 cells)
                dist = abs(key1.position[0] - key2.position[0]) + abs(
                    key1.position[1] - key2.position[1]
                )
                if dist <= 2:
                    key_pairs.append(KeyPair(
                        keys=(key1, key2),
                        room_id=len(key_pairs)
                    ))
                    paired_keys.add(key1)
                    paired_keys.add(key2)
                    break

    return key_pairs


def parse_layout_file(filepath: Union[str, Path]) -> Layout:
    """
    Load and parse a layout from a file.

    The file can contain just the ASCII layout, or include configuration
    after a --- separator.

    Args:
        filepath: Path to the layout file.

    Returns:
        Parsed Layout object.
    """
    filepath = Path(filepath)

    # Check for companion config file
    config_path = filepath.with_suffix(".json")
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    with open(filepath) as f:
        content = f.read()

    return parse_layout_string(content, config)
