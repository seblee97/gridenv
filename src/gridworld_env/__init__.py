"""
GridWorld Environment - A configurable RL environment with keys, doors, and Posner cueing.
"""

from gridworld_env.environment import GridWorldEnv
from gridworld_env.layout import Layout, parse_layout_file, parse_layout_string
from gridworld_env.objects import Key, Door, Reward, KeyColor

__version__ = "0.1.0"
__all__ = [
    "GridWorldEnv",
    "Layout",
    "parse_layout_file",
    "parse_layout_string",
    "Key",
    "Door",
    "Reward",
    "KeyColor",
]
