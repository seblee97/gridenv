"""
GridWorld Environment - Main Gymnasium environment implementation.
"""

from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gridworld_env.layout import Layout, parse_layout_file, parse_layout_string
from gridworld_env.objects import Door, Key, KeyColor, KeyPair, Reward


class Action(IntEnum):
    """Available actions in the environment."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType(IntEnum):
    """Types of cells for observation encoding."""
    EMPTY = 0
    WALL = 1
    AGENT = 2
    KEY_RED = 3
    KEY_BLUE = 4
    DOOR_CLOSED = 5
    DOOR_OPEN = 6
    REWARD = 7
    REWARD_DESTROYED = 8


class GridWorldEnv(gym.Env):
    """
    A configurable grid world environment with keys, doors, and Posner cueing.

    Features:
    - Customizable layouts via ASCII files or strings
    - Walls that block movement
    - Rewards collected automatically when stepped on
    - Keys collected automatically when stepped on (red/blue)
    - Doors block movement unless agent has correct key (auto-opens)
    - Key pairs where collecting one makes the other disappear
    - Wrong key destroys rewards behind the door
    - Posner mode: multi-feature cues where one feature indicates the correct key

    Observation Space:
        If flatten_obs=True (default): Box of shape (obs_dim,) containing flattened grid
        If flatten_obs=False: Dict with 'grid', 'agent_pos', 'held_key', 'posner_cue'

    Action Space:
        Discrete(4): UP, DOWN, LEFT, RIGHT

    Args:
        layout: Layout object, path to layout file, or ASCII string.
        posner_mode: Enable Posner cueing mode.
        posner_validity: Probability that the cue at posner_cue_index is correct (0.0-1.0).
        posner_num_features: Number of features in the Posner cue vector.
        posner_cue_index: Index of the feature that is the true cue (0 to num_features-1).
        max_steps: Maximum steps per episode (None for unlimited).
        step_reward: Reward given each step (usually negative for time pressure).
        collision_reward: Reward for walking into walls.
        flatten_obs: Whether to flatten observations to 1D array.
        render_mode: 'human', 'rgb_array', or None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layout: Union[Layout, str, Path],
        posner_mode: bool = False,
        posner_validity: float = 0.8,
        posner_num_features: int = 1,
        posner_cue_index: int = 0,
        max_steps: Optional[int] = None,
        step_reward: float = -0.01,
        collision_reward: float = -0.1,
        flatten_obs: bool = True,
        render_mode: Optional[str] = None,
        debug_cues: bool = False,
        show_score: bool = False,
    ):
        super().__init__()

        # Parse layout if needed
        if isinstance(layout, Layout):
            self._base_layout = layout
        elif isinstance(layout, Path) or (
            isinstance(layout, str) and Path(layout).exists()
        ):
            self._base_layout = parse_layout_file(layout)
        else:
            self._base_layout = parse_layout_string(layout)

        self.posner_mode = posner_mode
        self.posner_validity = posner_validity
        self.posner_num_features = posner_num_features
        self.posner_cue_index = posner_cue_index
        self.max_steps = max_steps
        self.step_reward = step_reward
        self.collision_reward = collision_reward
        self.flatten_obs = flatten_obs
        self.render_mode = render_mode
        self.debug_cues = debug_cues
        self.show_score = show_score

        # Validate posner_cue_index
        if posner_cue_index < 0 or posner_cue_index >= posner_num_features:
            raise ValueError(
                f"posner_cue_index ({posner_cue_index}) must be in range "
                f"[0, posner_num_features ({posner_num_features}))"
            )

        # Current episode state
        self._layout: Optional[Layout] = None
        self._agent_pos: Tuple[int, int] = (0, 0)
        self._held_key: Optional[KeyColor] = None
        self._posner_cues: Optional[List[KeyColor]] = None  # Current room's cue vector
        self._correct_key: Optional[KeyColor] = None  # Current room's correct key
        self._steps: int = 0

        # Per-room Posner cue state
        self._room_cues: Dict[int, List[KeyColor]] = {}  # room_id -> cue vector
        self._room_correct_keys: Dict[int, KeyColor] = {}  # room_id -> correct key
        self._current_room_index: int = 0  # Index into key_pairs list

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)

        # Observation space
        grid_shape = (self._base_layout.height, self._base_layout.width)
        num_cell_types = len(CellType)

        # Posner cue: each feature is one-hot encoded (none=0, red=1, blue=2)
        # So each feature takes 3 values, total = num_features * 3
        posner_cue_size = posner_num_features * 3 if posner_mode else 3

        if flatten_obs:
            # Flattened: grid (one-hot) + agent_pos + held_key + posner_cues
            grid_size = grid_shape[0] * grid_shape[1] * num_cell_types
            extra_features = 2 + 3 + posner_cue_size  # agent_pos(2) + held_key(3) + posner_cues
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(grid_size + extra_features,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict({
                "grid": spaces.Box(
                    low=0,
                    high=num_cell_types - 1,
                    shape=grid_shape,
                    dtype=np.int32,
                ),
                "agent_pos": spaces.Box(
                    low=0,
                    high=max(grid_shape),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "held_key": spaces.Discrete(3),  # 0=none, 1=red, 2=blue
                "posner_cue": spaces.Box(
                    low=0,
                    high=2,
                    shape=(posner_num_features,) if posner_mode else (1,),
                    dtype=np.int32,
                ),  # Each feature: 0=none, 1=red, 2=blue
            })

        # Rendering
        self._renderer = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Initial observation and info dict.
        """
        super().reset(seed=seed)

        # Deep copy the base layout for this episode
        self._layout = self._base_layout.copy()
        self._agent_pos = self._layout.start_position
        self._held_key = None
        self._steps = 0

        # Reset per-room Posner cue state
        self._room_cues = {}
        self._room_correct_keys = {}
        self._current_room_index = 0
        self._posner_cues = None
        self._correct_key = None

        if self.posner_mode and self._layout.key_pairs:
            # Generate cues for ALL key pairs/rooms upfront
            self._generate_all_room_cues()

            # Set current cues to first room
            if self._layout.key_pairs:
                first_room_id = self._layout.key_pairs[0].room_id
                self._posner_cues = self._room_cues.get(first_room_id)
                self._correct_key = self._room_correct_keys.get(first_room_id)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (0-3: UP, DOWN, LEFT, RIGHT).

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._layout is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._steps += 1
        reward = self.step_reward
        terminated = False
        truncated = False

        action = Action(action)
        move_reward, moved = self._handle_movement(action)
        reward += move_reward

        # If we moved, auto-collect keys and rewards at new position
        if moved:
            reward += self._auto_collect()

        # Check termination conditions
        if self._all_rewards_collected_or_destroyed():
            terminated = True

        if self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _handle_movement(self, action: Action) -> Tuple[float, bool]:
        """
        Handle movement actions.

        Returns:
            Tuple of (additional reward, whether agent moved)
        """
        row, col = self._agent_pos

        if action == Action.UP:
            new_pos = (row - 1, col)
        elif action == Action.DOWN:
            new_pos = (row + 1, col)
        elif action == Action.LEFT:
            new_pos = (row, col - 1)
        elif action == Action.RIGHT:
            new_pos = (row, col + 1)
        else:
            return 0.0, False

        # Check if new position is a wall
        if self._layout.is_wall(new_pos[0], new_pos[1]):
            return self.collision_reward, False

        # Check if there's a closed door
        door = self._layout.get_door_at(new_pos[0], new_pos[1])
        if door is not None and not door.is_open:
            # Check if agent has the correct key
            if self._held_key is not None:
                # Open the door with the held key
                door.open_with_key(self._held_key)

                # If wrong key, destroy protected rewards
                if door.wrong_key_used:
                    for r in self._layout.rewards:
                        if r.protected_by_door == door:
                            r.destroyed = True

                # Key is consumed
                self._held_key = None

                # Move through the now-open door
                self._agent_pos = new_pos
                return 0.0, True
            else:
                # No key, blocked by door
                return self.collision_reward, False

        self._agent_pos = new_pos
        return 0.0, True

    def _auto_collect(self) -> float:
        """
        Automatically collect keys and rewards at current position.

        Returns:
            Reward value collected.
        """
        row, col = self._agent_pos
        reward = 0.0

        # Collect reward if present
        env_reward = self._layout.get_reward_at(row, col)
        if env_reward is not None and env_reward.is_available():
            env_reward.collected = True
            reward += env_reward.value

        # Collect key if present (and not already holding one)
        if self._held_key is None:
            key_pair_result = self._layout.get_key_pair_at(row, col)
            if key_pair_result is not None:
                key_pair, key = key_pair_result
                key_pair.collect(key)
                self._held_key = key.color

                # Advance to next room's cues when key from key pair is collected
                if self.posner_mode:
                    self._advance_to_next_room()
            else:
                key = self._layout.get_key_at(row, col)
                if key is not None:
                    key.collected = True
                    self._held_key = key.color

        return reward

    def _all_rewards_collected_or_destroyed(self) -> bool:
        """Check if all rewards have been collected or destroyed."""
        for reward in self._layout.rewards:
            if reward.is_available():
                return False
        return True

    def _generate_all_room_cues(self) -> None:
        """Generate Posner cues for all rooms/key pairs at episode start."""
        key_colors = [KeyColor.RED, KeyColor.BLUE]

        for key_pair in self._layout.key_pairs:
            room_id = key_pair.room_id

            # Find the door that this key pair's room leads to
            # We need to match key pair to its corresponding door
            correct_key = self._find_correct_key_for_room(key_pair)

            if correct_key is None:
                # No door found for this room, pick randomly
                correct_key = self.np_random.choice(key_colors)

            self._room_correct_keys[room_id] = correct_key

            # Generate multi-feature cue vector for this room
            cues = []
            for i in range(self.posner_num_features):
                if i == self.posner_cue_index:
                    # This is the true cue - apply validity
                    if self.np_random.random() < self.posner_validity:
                        cues.append(correct_key)
                    else:
                        # Invalid cue - pick the wrong color
                        wrong_color = KeyColor.BLUE if correct_key == KeyColor.RED else KeyColor.RED
                        cues.append(wrong_color)
                else:
                    # Distractor feature - random color
                    cues.append(self.np_random.choice(key_colors))

            self._room_cues[room_id] = cues

    def _find_correct_key_for_room(self, key_pair: "KeyPair") -> Optional[KeyColor]:
        """
        Find the correct key color for a room based on its adjacent door.

        This searches for a door that the agent would reach after collecting
        a key from this key pair, based on proximity.
        """
        # Get positions of keys in this pair
        key_positions = [k.position for k in key_pair.keys]

        # Find the closest door to these key positions
        closest_door = None
        min_distance = float('inf')

        for door in self._layout.doors:
            if door.is_open:
                continue  # Skip already open doors

            for key_pos in key_positions:
                # Manhattan distance
                dist = abs(door.position[0] - key_pos[0]) + abs(door.position[1] - key_pos[1])
                if dist < min_distance:
                    min_distance = dist
                    closest_door = door

        if closest_door is not None:
            return closest_door.correct_key_color

        return None

    def _advance_to_next_room(self) -> None:
        """
        Advance to the next room's cues after a key is collected.

        Called when a key from the current room's key pair is collected.
        """
        self._current_room_index += 1

        if self._current_room_index < len(self._layout.key_pairs):
            # Move to next room's cues
            next_pair = self._layout.key_pairs[self._current_room_index]
            room_id = next_pair.room_id
            self._posner_cues = self._room_cues.get(room_id)
            self._correct_key = self._room_correct_keys.get(room_id)
        else:
            # No more rooms - clear cues
            self._posner_cues = None
            self._correct_key = None

    def _get_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Build the current observation."""
        grid = self._build_grid_observation()

        if self.flatten_obs:
            # One-hot encode the grid
            height, width = grid.shape
            num_types = len(CellType)
            one_hot = np.zeros((height, width, num_types), dtype=np.float32)
            for i in range(height):
                for j in range(width):
                    one_hot[i, j, grid[i, j]] = 1.0

            flat_grid = one_hot.flatten()

            # Agent position (normalized)
            agent_pos = np.array([
                self._agent_pos[0] / max(1, height - 1),
                self._agent_pos[1] / max(1, width - 1),
            ], dtype=np.float32)

            # Held key (one-hot: none, red, blue)
            held_key = np.zeros(3, dtype=np.float32)
            if self._held_key is None:
                held_key[0] = 1.0
            elif self._held_key == KeyColor.RED:
                held_key[1] = 1.0
            else:
                held_key[2] = 1.0

            # Posner cues (one-hot for each feature: none, red, blue)
            if self.posner_mode:
                posner_cue = np.zeros(self.posner_num_features * 3, dtype=np.float32)
                if self._posner_cues is not None:
                    for i, cue in enumerate(self._posner_cues):
                        offset = i * 3
                        if cue == KeyColor.RED:
                            posner_cue[offset + 1] = 1.0
                        else:  # BLUE
                            posner_cue[offset + 2] = 1.0
                else:
                    # No cue - mark all as "none"
                    for i in range(self.posner_num_features):
                        posner_cue[i * 3] = 1.0
            else:
                # Non-posner mode: single cue slot
                posner_cue = np.zeros(3, dtype=np.float32)
                posner_cue[0] = 1.0  # none

            return np.concatenate([flat_grid, agent_pos, held_key, posner_cue])
        else:
            held_key_val = 0
            if self._held_key == KeyColor.RED:
                held_key_val = 1
            elif self._held_key == KeyColor.BLUE:
                held_key_val = 2

            if self.posner_mode and self._posner_cues is not None:
                posner_cue_vals = np.array([
                    1 if c == KeyColor.RED else 2 for c in self._posner_cues
                ], dtype=np.int32)
            else:
                posner_cue_vals = np.array([0], dtype=np.int32)

            return {
                "grid": grid,
                "agent_pos": np.array(self._agent_pos, dtype=np.int32),
                "held_key": held_key_val,
                "posner_cue": posner_cue_vals,
            }

    def _build_grid_observation(self) -> np.ndarray:
        """Build grid observation showing cell types."""
        height = self._layout.height
        width = self._layout.width
        grid = np.zeros((height, width), dtype=np.int32)

        for row in range(height):
            for col in range(width):
                if self._layout.is_wall(row, col):
                    grid[row, col] = CellType.WALL
                else:
                    grid[row, col] = CellType.EMPTY

        # Add keys
        for key in self._layout.keys:
            if not key.collected:
                row, col = key.position
                if key.color == KeyColor.RED:
                    grid[row, col] = CellType.KEY_RED
                else:
                    grid[row, col] = CellType.KEY_BLUE

        # Add keys from key pairs
        for key_pair in self._layout.key_pairs:
            for key in key_pair.get_available_keys():
                row, col = key.position
                if key.color == KeyColor.RED:
                    grid[row, col] = CellType.KEY_RED
                else:
                    grid[row, col] = CellType.KEY_BLUE

        # Add doors
        for door in self._layout.doors:
            row, col = door.position
            if door.is_open:
                grid[row, col] = CellType.DOOR_OPEN
            else:
                grid[row, col] = CellType.DOOR_CLOSED

        # Add rewards
        for reward in self._layout.rewards:
            row, col = reward.position
            if reward.destroyed:
                grid[row, col] = CellType.REWARD_DESTROYED
            elif not reward.collected:
                grid[row, col] = CellType.REWARD

        # Add agent
        row, col = self._agent_pos
        grid[row, col] = CellType.AGENT

        return grid

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict."""
        collected_rewards = sum(
            r.value for r in self._layout.rewards if r.collected
        )
        destroyed_rewards = sum(
            r.value for r in self._layout.rewards if r.destroyed
        )
        available_rewards = sum(
            r.value for r in self._layout.rewards if r.is_available()
        )

        info = {
            "steps": self._steps,
            "collected_rewards": collected_rewards,
            "destroyed_rewards": destroyed_rewards,
            "available_rewards": available_rewards,
            "held_key": str(self._held_key) if self._held_key else None,
            "agent_pos": self._agent_pos,
        }

        if self.posner_mode:
            # Report all cues and which index is the true one
            if self._posner_cues is not None:
                info["posner_cues"] = [str(c) for c in self._posner_cues]
                info["posner_cue_index"] = self.posner_cue_index
                true_cue = self._posner_cues[self.posner_cue_index]
                info["posner_cue"] = str(true_cue)
                info["posner_cue_valid"] = (true_cue == self._correct_key)
            else:
                info["posner_cues"] = None
                info["posner_cue_index"] = self.posner_cue_index
                info["posner_cue"] = None
                info["posner_cue_valid"] = None

        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from gridworld_env.rendering import Renderer
            self._renderer = Renderer(
                self._base_layout.width,
                self._base_layout.height,
                self.render_mode,
            )

        # Compute debug info if enabled
        debug_info = None
        if self.debug_cues and self._posner_cues is not None:
            true_cue = self._posner_cues[self.posner_cue_index]
            debug_info = {
                "cue_index": self.posner_cue_index,
                "cue_valid": true_cue == self._correct_key,
            }

        # Compute score if enabled
        score = None
        if self.show_score:
            score = sum(r.value for r in self._layout.rewards if r.collected)

        return self._renderer.render(
            self._layout,
            self._agent_pos,
            self._held_key,
            self._posner_cues,
            debug_info,
            score,
        )

    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names."""
        return ["UP", "DOWN", "LEFT", "RIGHT"]

    @property
    def agent_position(self) -> Tuple[int, int]:
        """Current agent position."""
        return self._agent_pos

    @property
    def held_key(self) -> Optional[KeyColor]:
        """Currently held key color."""
        return self._held_key

    @property
    def posner_cue(self) -> Optional[KeyColor]:
        """The true Posner cue (at posner_cue_index)."""
        if self._posner_cues is not None:
            return self._posner_cues[self.posner_cue_index]
        return None

    @property
    def posner_cues(self) -> Optional[List[KeyColor]]:
        """All Posner cue features."""
        return self._posner_cues
