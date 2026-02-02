"""
Continual learning support: task sequences for the GridWorld environment.

Provides TaskSequenceWrapper, a Gymnasium wrapper that presents a sequence of
tasks to the agent. Each task can vary the layout (same grid size), Posner cue
contingencies, and scalar reward parameters.

Example usage::

    from gridworld_env import GridWorldEnv
    from gridworld_env.continual import TaskSequenceWrapper, TaskConfig

    env = GridWorldEnv(layout_a, posner_mode=True, posner_num_features=3)

    tasks = [
        TaskConfig(layout=layout_a, posner_validity=0.8, posner_cue_index=0),
        TaskConfig(layout=layout_b, posner_validity=0.5, posner_cue_index=2),
        TaskConfig(layout=layout_a, posner_validity=0.8, posner_cue_index=0),
    ]

    wrapped = TaskSequenceWrapper(env, tasks, episodes_per_task=100, cycle=False)

    obs, info = wrapped.reset()
    # info["task_index"] == 0
    # After 100 episodes, automatically advances to task 1
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gridworld_env.environment import GridWorldEnv, _UNSET
from gridworld_env.layout import Layout, parse_layout_file, parse_layout_string


@dataclass
class TaskConfig:
    """Configuration for a single task in a continual learning sequence.

    Only ``layout`` is required. All other fields default to ``None``,
    meaning "keep the base environment's current value".

    Attributes:
        layout: Layout object, file path, or ASCII string. Must have the same
            grid dimensions as the base environment.
        posner_validity: Cue validity for this task (0.0--1.0).
        posner_cue_index: Which feature index is the true cue.
        step_reward: Per-step reward (usually negative).
        collision_reward: Penalty for walking into walls.
        max_steps: Max steps per episode. ``None`` means unlimited.
            Omit (or set to ``_UNSET``) to keep the base env's value.
        start_pos_mode: ``"fixed"`` or ``"random_in_room"``.
        metadata: Arbitrary user-defined metadata for logging.
    """

    layout: Union[Layout, str, Path]
    posner_validity: Optional[float] = None
    posner_cue_index: Optional[int] = None
    step_reward: Optional[float] = None
    collision_reward: Optional[float] = None
    max_steps: object = _UNSET
    start_pos_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _dict_to_task_config(d: dict) -> TaskConfig:
    """Convert a plain dict to a TaskConfig, passing only recognised keys."""
    known = {f.name for f in TaskConfig.__dataclass_fields__.values()}
    return TaskConfig(**{k: v for k, v in d.items() if k in known})


def _parse_layout(layout: Union[Layout, str, Path]) -> Layout:
    """Parse a layout from various input types."""
    if isinstance(layout, Layout):
        return layout
    if isinstance(layout, Path) or (
        isinstance(layout, str) and Path(layout).exists()
    ):
        return parse_layout_file(layout)
    return parse_layout_string(layout)


class TaskSequenceWrapper(gym.Wrapper):
    """Gymnasium wrapper that presents a sequence of tasks for continual learning.

    All tasks must share the same grid dimensions. The observation and action
    spaces are set once at construction and remain fixed. For
    ``obs_mode="symbolic_minimal"``, observations are zero-padded to the
    maximum object count across all tasks so that the shape is constant.

    Args:
        env: A ``GridWorldEnv`` instance.
        task_configs: Sequence of :class:`TaskConfig` objects (or plain dicts).
        episodes_per_task: Auto-advance to the next task after this many
            episodes. ``None`` disables auto-advance (use :meth:`advance_task`
            or :meth:`set_task` manually).
        cycle: If ``True``, loop back to task 0 after the last task.
    """

    def __init__(
        self,
        env: GridWorldEnv,
        task_configs: List[Union[TaskConfig, dict]],
        episodes_per_task: Optional[int] = None,
        cycle: bool = False,
    ):
        super().__init__(env)

        if len(task_configs) == 0:
            raise ValueError("task_configs must contain at least one task")

        # Normalise dicts to TaskConfig
        self._task_configs: List[TaskConfig] = [
            _dict_to_task_config(tc) if isinstance(tc, dict) else tc
            for tc in task_configs
        ]

        # Pre-parse all layouts and validate grid sizes
        base_h = env._base_layout.height
        base_w = env._base_layout.width
        self._parsed_layouts: List[Layout] = []
        for i, tc in enumerate(self._task_configs):
            layout = _parse_layout(tc.layout)
            if layout.height != base_h or layout.width != base_w:
                raise ValueError(
                    f"Task {i} layout has grid size ({layout.height}, {layout.width}), "
                    f"expected ({base_h}, {base_w})"
                )
            self._parsed_layouts.append(layout)

        # Compute max object counts across all tasks (for obs padding)
        self._max_n_keys = 0
        self._max_n_keypair_keys = 0
        self._max_n_doors = 0
        self._max_n_rewards = 0
        for layout in self._parsed_layouts:
            paired = set()
            for kp in layout.key_pairs:
                for k in kp.keys:
                    paired.add(k.position)
            n_keys = sum(1 for k in layout.keys if k.position not in paired)
            n_kp = sum(len(kp.keys) for kp in layout.key_pairs)
            self._max_n_keys = max(self._max_n_keys, n_keys)
            self._max_n_keypair_keys = max(self._max_n_keypair_keys, n_kp)
            self._max_n_doors = max(self._max_n_doors, len(layout.doors))
            self._max_n_rewards = max(self._max_n_rewards, len(layout.rewards))

        # Rebuild observation space for symbolic_minimal if object counts vary
        self._needs_padding = (env.obs_mode == "symbolic_minimal")
        if self._needs_padding:
            self.observation_space = self._build_padded_obs_space()

        # Tracking state
        self._task_index: int = 0
        self._episodes_on_task: int = 0
        self._total_episodes: int = 0
        self._episodes_per_task = episodes_per_task
        self._cycle = cycle
        self._sequence_complete = False

        # Apply first task
        self._apply_task(0)

    # ------------------------------------------------------------------
    # Observation space construction
    # ------------------------------------------------------------------

    def _build_padded_obs_space(self) -> gym.Space:
        """Build an observation space sized to the max object counts."""
        env = self.env
        posner_cue_size = (
            env.posner_num_features * 3 if env.posner_mode else 3
        )
        max_obj = (
            self._max_n_keys
            + self._max_n_keypair_keys
            + self._max_n_doors
            + self._max_n_rewards
        )

        if env.flatten_obs:
            dim = 2 + 3 + max_obj + posner_cue_size
            return spaces.Box(
                low=0.0, high=1.0, shape=(dim,), dtype=np.float32,
            )
        else:
            grid_shape = (env._base_layout.height, env._base_layout.width)
            d: Dict[str, gym.Space] = {
                "agent_pos": spaces.Box(
                    low=0, high=max(grid_shape), shape=(2,), dtype=np.int32,
                ),
                "held_key": spaces.Discrete(3),
                "posner_cue": spaces.Box(
                    low=0,
                    high=2,
                    shape=(
                        (env.posner_num_features,)
                        if env.posner_mode
                        else (1,)
                    ),
                    dtype=np.int32,
                ),
            }
            if self._max_n_keys > 0:
                d["keys"] = spaces.Box(
                    low=0, high=1, shape=(self._max_n_keys,), dtype=np.int32,
                )
            if self._max_n_keypair_keys > 0:
                d["keypair_keys"] = spaces.Box(
                    low=0, high=1,
                    shape=(self._max_n_keypair_keys,), dtype=np.int32,
                )
            if self._max_n_doors > 0:
                d["doors"] = spaces.Box(
                    low=0, high=1, shape=(self._max_n_doors,), dtype=np.int32,
                )
            if self._max_n_rewards > 0:
                d["rewards"] = spaces.Box(
                    low=0, high=1,
                    shape=(self._max_n_rewards,), dtype=np.int32,
                )
            return spaces.Dict(d)

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def _apply_task(self, task_index: int) -> None:
        """Apply a task configuration to the underlying environment."""
        config = self._task_configs[task_index]
        layout = self._parsed_layouts[task_index]

        kwargs: Dict[str, Any] = {}
        if config.posner_validity is not None:
            kwargs["posner_validity"] = config.posner_validity
        if config.posner_cue_index is not None:
            kwargs["posner_cue_index"] = config.posner_cue_index
        if config.step_reward is not None:
            kwargs["step_reward"] = config.step_reward
        if config.collision_reward is not None:
            kwargs["collision_reward"] = config.collision_reward
        if config.max_steps is not _UNSET:
            kwargs["max_steps"] = config.max_steps
        if config.start_pos_mode is not None:
            kwargs["start_pos_mode"] = config.start_pos_mode

        self.env._reconfigure(layout, **kwargs)
        self._task_index = task_index
        self._episodes_on_task = 0

    def advance_task(self) -> bool:
        """Advance to the next task in the sequence.

        Returns:
            ``True`` if advanced successfully. ``False`` if the sequence is
            complete (last task reached and ``cycle=False``).
        """
        next_index = self._task_index + 1
        if next_index >= len(self._task_configs):
            if self._cycle:
                next_index = 0
            else:
                self._sequence_complete = True
                return False
        self._apply_task(next_index)
        self._sequence_complete = False
        return True

    def set_task(self, task_index: int) -> None:
        """Jump to a specific task by index.

        Args:
            task_index: Index into the task_configs list.
        """
        if task_index < 0 or task_index >= len(self._task_configs):
            raise IndexError(
                f"task_index {task_index} out of range "
                f"[0, {len(self._task_configs)})"
            )
        self._apply_task(task_index)
        self._sequence_complete = False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Auto-advance if episode budget for current task is exhausted
        if (
            self._episodes_per_task is not None
            and self._episodes_on_task >= self._episodes_per_task
        ):
            self.advance_task()

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._pad_observation(obs)

        self._episodes_on_task += 1
        self._total_episodes += 1

        info.update(self._task_info())
        return obs, info

    def step(
        self, action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._pad_observation(obs)
        info.update(self._task_info())
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation padding
    # ------------------------------------------------------------------

    def _pad_observation(self, obs):
        """Pad observation to match the wrapper's observation space."""
        if not self._needs_padding:
            return obs
        if self.env.flatten_obs:
            return self._pad_flat_minimal(obs)
        return self._pad_dict_minimal(obs)

    def _pad_flat_minimal(self, obs: np.ndarray) -> np.ndarray:
        """Pad a flat symbolic_minimal observation to max object counts."""
        env = self.env
        cur_nk = env._n_keys
        cur_nkp = env._n_keypair_keys
        cur_nd = env._n_doors
        cur_nr = env._n_rewards

        # Fast path: no padding needed
        if (
            cur_nk == self._max_n_keys
            and cur_nkp == self._max_n_keypair_keys
            and cur_nd == self._max_n_doors
            and cur_nr == self._max_n_rewards
        ):
            return obs

        posner_cue_size = (
            env.posner_num_features * 3 if env.posner_mode else 3
        )

        # Slice into sections
        idx = 0
        agent_pos = obs[idx : idx + 2]; idx += 2
        held_key = obs[idx : idx + 3]; idx += 3
        keys = obs[idx : idx + cur_nk]; idx += cur_nk
        kp_keys = obs[idx : idx + cur_nkp]; idx += cur_nkp
        doors = obs[idx : idx + cur_nd]; idx += cur_nd
        rewards = obs[idx : idx + cur_nr]; idx += cur_nr
        posner_cue = obs[idx : idx + posner_cue_size]

        def _pad(arr: np.ndarray, target: int) -> np.ndarray:
            if len(arr) >= target:
                return arr
            return np.concatenate(
                [arr, np.zeros(target - len(arr), dtype=np.float32)]
            )

        return np.concatenate([
            agent_pos,
            held_key,
            _pad(keys, self._max_n_keys),
            _pad(kp_keys, self._max_n_keypair_keys),
            _pad(doors, self._max_n_doors),
            _pad(rewards, self._max_n_rewards),
            posner_cue,
        ])

    def _pad_dict_minimal(self, obs: dict) -> dict:
        """Pad a dict symbolic_minimal observation to max object counts."""
        padded = dict(obs)

        def _ensure(name: str, max_len: int) -> None:
            if max_len == 0:
                return
            if name not in padded:
                padded[name] = np.zeros(max_len, dtype=np.int32)
            elif len(padded[name]) < max_len:
                padded[name] = np.concatenate([
                    padded[name],
                    np.zeros(max_len - len(padded[name]), dtype=np.int32),
                ])

        _ensure("keys", self._max_n_keys)
        _ensure("keypair_keys", self._max_n_keypair_keys)
        _ensure("doors", self._max_n_doors)
        _ensure("rewards", self._max_n_rewards)
        return padded

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def _task_info(self) -> Dict[str, Any]:
        """Task metadata to merge into the info dict."""
        return {
            "task_index": self._task_index,
            "episodes_on_task": self._episodes_on_task,
            "total_episodes": self._total_episodes,
            "num_tasks": len(self._task_configs),
            "sequence_complete": self._sequence_complete,
            "task_metadata": self._task_configs[self._task_index].metadata,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_index(self) -> int:
        """Current task index."""
        return self._task_index

    @property
    def episodes_on_task(self) -> int:
        """Number of episodes completed on the current task."""
        return self._episodes_on_task

    @property
    def total_episodes(self) -> int:
        """Total episodes across all tasks."""
        return self._total_episodes

    @property
    def num_tasks(self) -> int:
        """Number of tasks in the sequence."""
        return len(self._task_configs)

    @property
    def sequence_complete(self) -> bool:
        """Whether all tasks have been exhausted (when ``cycle=False``)."""
        return self._sequence_complete

    @property
    def current_task_config(self) -> TaskConfig:
        """The :class:`TaskConfig` for the current task."""
        return self._task_configs[self._task_index]
