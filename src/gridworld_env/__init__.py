"""
GridWorld Environment - A configurable RL environment with keys, doors, and Posner cueing.
"""

from gridworld_env.continual import TaskConfig, TaskSequenceWrapper
from gridworld_env.environment import GridWorldEnv
from gridworld_env.layout import Layout, parse_layout_file, parse_layout_string
from gridworld_env.modular_layout import (
    ModularLayout,
    parse_modular_layout_file,
    parse_modular_layout_string,
)
from gridworld_env.modular_maze import ModularCellType, ModularMazeEnv
from gridworld_env.modular_objects import ModularKey, Safe, SimpleDoor
from gridworld_env.objects import Door, Key, KeyColor, Reward
from gridworld_env.world_layout import (
    Connection,
    RoomSpec,
    WorldSpec,
    assemble_world,
    parse_room_file,
    parse_world_file,
    parse_world_layout_file,
)

__version__ = "0.1.0"
__all__ = [
    # Original environment
    "GridWorldEnv",
    "Layout",
    "parse_layout_file",
    "parse_layout_string",
    "Key",
    "Door",
    "Reward",
    "KeyColor",
    "TaskConfig",
    "TaskSequenceWrapper",
    # Modular Maze environment
    "ModularMazeEnv",
    "ModularLayout",
    "ModularCellType",
    "parse_modular_layout_file",
    "parse_modular_layout_string",
    "ModularKey",
    "Safe",
    "SimpleDoor",
    # World layout (multi-file)
    "RoomSpec",
    "Connection",
    "WorldSpec",
    "parse_room_file",
    "parse_world_file",
    "assemble_world",
    "parse_world_layout_file",
]
