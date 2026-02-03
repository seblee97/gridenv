"""Tests for continual learning task sequence support."""

import pytest
import numpy as np

from gridworld_env.continual import TaskConfig, TaskSequenceWrapper
from gridworld_env.environment import GridWorldEnv, _UNSET
from gridworld_env.layout import parse_layout_string


# ---------------------------------------------------------------------------
# Shared layouts (same grid size: 7x7)
# ---------------------------------------------------------------------------

LAYOUT_A_STR = """\
#######
#S....#
#.r...#
#.b...#
###D###
#..G..#
#######"""

LAYOUT_A_CONFIG = {
    "default_correct_keys": {"4,3": "red"},
    "key_pairs": [{"positions": ["2,2", "3,2"], "room_id": 0}],
    "protected_rewards": {"4,3": ["5,3"]},
}

LAYOUT_B_STR = """\
#######
#S....#
#.....#
#.....#
#.....#
#....G#
#######"""

# Different grid size for mismatch testing
LAYOUT_SMALL_STR = """\
#####
#S.G#
#####"""


@pytest.fixture
def layout_a():
    return parse_layout_string(LAYOUT_A_STR, LAYOUT_A_CONFIG)


@pytest.fixture
def layout_b():
    return parse_layout_string(LAYOUT_B_STR)


@pytest.fixture
def base_env(layout_a):
    return GridWorldEnv(layout_a, posner_mode=True, posner_num_features=3)


@pytest.fixture
def two_task_configs(layout_a, layout_b):
    return [
        TaskConfig(layout=layout_a, posner_validity=0.8, posner_cue_index=0),
        TaskConfig(layout=layout_b, posner_validity=0.5, posner_cue_index=2),
    ]


# ===================================================================
# _reconfigure() tests
# ===================================================================


class TestReconfigure:
    """Tests for GridWorldEnv._reconfigure()."""

    def test_reconfigure_changes_layout(self, base_env, layout_b):
        base_env._reconfigure(layout_b)
        assert base_env._base_layout.rewards[0].position == layout_b.rewards[0].position

    def test_reconfigure_preserves_grid_size(self, base_env, layout_b):
        old_h, old_w = base_env._base_layout.height, base_env._base_layout.width
        base_env._reconfigure(layout_b)
        assert base_env._base_layout.height == old_h
        assert base_env._base_layout.width == old_w

    def test_reconfigure_mismatched_grid_raises(self, base_env):
        small = parse_layout_string(LAYOUT_SMALL_STR)
        with pytest.raises(ValueError, match="grid size"):
            base_env._reconfigure(small)

    def test_reconfigure_updates_object_counts(self, base_env, layout_b):
        base_env._reconfigure(layout_b)
        # layout_b has no keys, no doors, no key pairs
        assert base_env._n_keys == 0
        assert base_env._n_keypair_keys == 0
        assert base_env._n_doors == 0
        assert base_env._n_rewards == 1

    def test_reconfigure_updates_scalar_params(self, base_env, layout_b):
        base_env._reconfigure(
            layout_b,
            posner_validity=0.3,
            posner_cue_index=1,
            step_reward=-0.05,
            collision_reward=-0.2,
            max_steps=50,
            start_pos_mode="random_in_room",
        )
        assert base_env.posner_validity == 0.3
        assert base_env.posner_cue_index == 1
        assert base_env.step_reward == -0.05
        assert base_env.collision_reward == -0.2
        assert base_env.max_steps == 50
        assert base_env.start_pos_mode == "random_in_room"

    def test_reconfigure_invalidates_caches(self, base_env, layout_b):
        # Prime the cache
        base_env._get_first_room_positions()
        assert base_env._first_room_positions is not None
        base_env._reconfigure(layout_b)
        assert base_env._first_room_positions is None

    def test_reconfigure_then_reset_works(self, base_env, layout_b):
        base_env._reconfigure(layout_b)
        obs, info = base_env.reset(seed=42)
        assert obs is not None

    def test_reconfigure_then_step_works(self, base_env, layout_b):
        base_env._reconfigure(layout_b)
        base_env.reset(seed=42)
        obs, reward, terminated, truncated, info = base_env.step(0)
        assert obs is not None

    def test_reconfigure_invalid_posner_cue_index_raises(self, base_env, layout_b):
        with pytest.raises(ValueError, match="posner_cue_index"):
            base_env._reconfigure(layout_b, posner_cue_index=10)

    def test_reconfigure_invalid_start_pos_mode_raises(self, base_env, layout_b):
        with pytest.raises(ValueError, match="start_pos_mode"):
            base_env._reconfigure(layout_b, start_pos_mode="invalid")

    def test_reconfigure_max_steps_none_means_unlimited(self, base_env, layout_b):
        base_env.max_steps = 100
        base_env._reconfigure(layout_b, max_steps=None)
        assert base_env.max_steps is None

    def test_reconfigure_omit_max_steps_keeps_current(self, base_env, layout_b):
        base_env.max_steps = 100
        base_env._reconfigure(layout_b)
        assert base_env.max_steps == 100

    def test_reconfigure_recomputes_valid_positions(self, base_env, layout_b):
        old_positions = list(base_env._valid_positions)
        base_env._reconfigure(layout_b)
        # layout_b has a different wall structure, so positions should differ
        assert base_env._valid_positions != old_positions


# ===================================================================
# TaskSequenceWrapper basics
# ===================================================================


class TestTaskSequenceBasics:
    """Tests for TaskSequenceWrapper construction and basic operations."""

    def test_init_single_task(self, base_env, layout_a):
        tasks = [TaskConfig(layout=layout_a)]
        w = TaskSequenceWrapper(base_env, tasks)
        assert w.num_tasks == 1
        assert w.task_index == 0

    def test_init_multiple_tasks(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        assert w.num_tasks == 2

    def test_empty_task_list_raises(self, base_env):
        with pytest.raises(ValueError, match="at least one"):
            TaskSequenceWrapper(base_env, [])

    def test_grid_size_mismatch_raises(self, base_env):
        small = parse_layout_string(LAYOUT_SMALL_STR)
        with pytest.raises(ValueError, match="grid size"):
            TaskSequenceWrapper(base_env, [TaskConfig(layout=small)])

    def test_task_index_starts_at_zero(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        assert w.task_index == 0

    def test_reset_returns_augmented_info(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        _, info = w.reset(seed=42)
        assert info["task_index"] == 0
        assert info["episodes_on_task"] == 1
        assert info["total_episodes"] == 1
        assert info["num_tasks"] == 2
        assert info["sequence_complete"] is False

    def test_step_returns_augmented_info(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.reset(seed=42)
        _, _, _, _, info = w.step(0)
        assert "task_index" in info
        assert "episodes_on_task" in info

    def test_dict_task_configs(self, base_env, layout_a, layout_b):
        tasks = [
            {"layout": layout_a, "posner_validity": 0.8},
            {"layout": layout_b, "posner_validity": 0.5},
        ]
        w = TaskSequenceWrapper(base_env, tasks)
        assert w.num_tasks == 2

    def test_task_metadata_in_info(self, base_env, layout_a):
        tasks = [TaskConfig(layout=layout_a, metadata={"name": "phase1"})]
        w = TaskSequenceWrapper(base_env, tasks)
        _, info = w.reset(seed=42)
        assert info["task_metadata"] == {"name": "phase1"}


# ===================================================================
# Manual advance
# ===================================================================


class TestManualAdvance:
    """Tests for manual task advancement."""

    def test_advance_task_increments_index(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.reset(seed=42)
        assert w.task_index == 0
        assert w.advance_task() is True
        assert w.task_index == 1

    def test_advance_past_end_returns_false(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.advance_task()  # -> task 1
        assert w.advance_task() is False
        assert w.sequence_complete is True

    def test_advance_with_cycle_wraps(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs, cycle=True)
        w.advance_task()  # -> task 1
        assert w.advance_task() is True
        assert w.task_index == 0

    def test_set_task_jumps(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.set_task(1)
        assert w.task_index == 1

    def test_set_task_out_of_range_raises(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        with pytest.raises(IndexError):
            w.set_task(5)
        with pytest.raises(IndexError):
            w.set_task(-1)

    def test_set_task_clears_sequence_complete(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.advance_task()
        w.advance_task()  # sequence_complete = True
        assert w.sequence_complete is True
        w.set_task(0)
        assert w.sequence_complete is False

    def test_reset_after_advance_uses_new_task(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(base_env, two_task_configs)
        w.reset(seed=42)
        w.advance_task()
        _, info = w.reset(seed=42)
        assert info["task_index"] == 1


# ===================================================================
# Auto-advance
# ===================================================================


class TestAutoAdvance:
    """Tests for automatic task advancement."""

    def test_auto_advance_after_n_episodes(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(
            base_env, two_task_configs, episodes_per_task=3
        )
        for _ in range(3):
            w.reset(seed=42)
        assert w.task_index == 0  # still on task 0 during the 3rd episode

        # 4th reset triggers advance
        _, info = w.reset(seed=42)
        assert info["task_index"] == 1

    def test_auto_advance_multiple_tasks(self, base_env, layout_a, layout_b):
        tasks = [
            TaskConfig(layout=layout_a),
            TaskConfig(layout=layout_b),
            TaskConfig(layout=layout_a),
        ]
        w = TaskSequenceWrapper(
            base_env, tasks, episodes_per_task=2
        )
        # 2 episodes on task 0
        w.reset(seed=42)
        w.reset(seed=42)
        # 3rd reset advances to task 1
        _, info = w.reset(seed=42)
        assert info["task_index"] == 1
        # 1 more episode on task 1
        w.reset(seed=42)
        # 5th reset advances to task 2
        _, info = w.reset(seed=42)
        assert info["task_index"] == 2

    def test_auto_advance_with_cycle(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(
            base_env, two_task_configs, episodes_per_task=1, cycle=True
        )
        _, info = w.reset(seed=42)
        assert info["task_index"] == 0
        _, info = w.reset(seed=42)
        assert info["task_index"] == 1
        _, info = w.reset(seed=42)
        assert info["task_index"] == 0  # cycled back

    def test_auto_advance_sequence_complete(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(
            base_env, two_task_configs, episodes_per_task=1
        )
        w.reset(seed=42)  # task 0, episode 1
        w.reset(seed=42)  # advance to task 1, episode 1
        _, info = w.reset(seed=42)  # try advance past end, stay on task 1
        assert info["task_index"] == 1
        assert info["sequence_complete"] is True

    def test_episodes_on_task_resets(self, base_env, two_task_configs):
        w = TaskSequenceWrapper(
            base_env, two_task_configs, episodes_per_task=2
        )
        w.reset(seed=42)
        assert w.episodes_on_task == 1
        w.reset(seed=42)
        assert w.episodes_on_task == 2
        # Next reset advances
        w.reset(seed=42)
        assert w.episodes_on_task == 1  # reset for new task


# ===================================================================
# Observation padding (symbolic_minimal)
# ===================================================================


class TestObservationPadding:
    """Tests for observation padding with symbolic_minimal mode."""

    @pytest.fixture
    def minimal_env(self, layout_a):
        return GridWorldEnv(
            layout_a,
            posner_mode=True,
            posner_num_features=3,
            obs_mode="symbolic_minimal",
            flatten_obs=True,
        )

    @pytest.fixture
    def minimal_tasks(self, layout_a, layout_b):
        return [
            TaskConfig(layout=layout_a),
            TaskConfig(layout=layout_b),
        ]

    def test_flat_minimal_padding_shape(self, minimal_env, minimal_tasks):
        w = TaskSequenceWrapper(minimal_env, minimal_tasks)
        obs, _ = w.reset(seed=42)
        assert obs.shape == w.observation_space.shape

    def test_flat_minimal_shape_consistent_across_tasks(
        self, minimal_env, minimal_tasks
    ):
        w = TaskSequenceWrapper(minimal_env, minimal_tasks)
        obs_a, _ = w.reset(seed=42)
        w.advance_task()
        obs_b, _ = w.reset(seed=42)
        assert obs_a.shape == obs_b.shape

    def test_flat_minimal_obs_in_space(self, minimal_env, minimal_tasks):
        w = TaskSequenceWrapper(minimal_env, minimal_tasks)
        for i in range(len(minimal_tasks)):
            if i > 0:
                w.advance_task()
            obs, _ = w.reset(seed=42)
            assert w.observation_space.contains(obs), (
                f"Observation not in space for task {i}"
            )

    def test_flat_minimal_step_obs_shape(self, minimal_env, minimal_tasks):
        w = TaskSequenceWrapper(minimal_env, minimal_tasks)
        w.reset(seed=42)
        obs, _, _, _, _ = w.step(0)
        assert obs.shape == w.observation_space.shape

    def test_symbolic_mode_no_padding(self, layout_a, layout_b):
        env = GridWorldEnv(layout_a, posner_mode=True, posner_num_features=3)
        tasks = [TaskConfig(layout=layout_a), TaskConfig(layout=layout_b)]
        w = TaskSequenceWrapper(env, tasks)
        obs, _ = w.reset(seed=42)
        assert obs.shape == w.observation_space.shape

    def test_dict_minimal_padding(self, layout_a, layout_b):
        env = GridWorldEnv(
            layout_a,
            posner_mode=True,
            posner_num_features=3,
            obs_mode="symbolic_minimal",
            flatten_obs=False,
        )
        tasks = [TaskConfig(layout=layout_a), TaskConfig(layout=layout_b)]
        w = TaskSequenceWrapper(env, tasks)

        obs_a, _ = w.reset(seed=42)
        w.advance_task()
        obs_b, _ = w.reset(seed=42)

        # Both should have same dict keys with same array sizes
        for key in ("keypair_keys", "doors", "rewards"):
            if key in obs_a:
                assert key in obs_b
                assert len(obs_a[key]) == len(obs_b[key])


# ===================================================================
# Task switching behaviour
# ===================================================================


class TestTaskSwitchingBehavior:
    """Tests verifying that task parameters actually take effect."""

    def test_different_posner_validity_per_task(self, layout_a, layout_b):
        env = GridWorldEnv(
            layout_a, posner_mode=True, posner_num_features=1
        )
        tasks = [
            TaskConfig(layout=layout_a, posner_validity=1.0, posner_cue_index=0),
            TaskConfig(layout=layout_a, posner_validity=0.0, posner_cue_index=0),
        ]
        w = TaskSequenceWrapper(env, tasks)

        # Task 0: validity=1.0 → cue always valid
        valid_count = 0
        n = 50
        for _ in range(n):
            _, info = w.reset()
            if info.get("posner_cue_valid"):
                valid_count += 1
        assert valid_count == n, "With validity=1.0, all cues should be valid"

        # Task 1: validity=0.0 → cue never valid
        w.advance_task()
        valid_count = 0
        for _ in range(n):
            _, info = w.reset()
            if info.get("posner_cue_valid"):
                valid_count += 1
        assert valid_count == 0, "With validity=0.0, no cues should be valid"

    def test_different_step_reward_per_task(self, base_env, layout_a, layout_b):
        tasks = [
            TaskConfig(layout=layout_a, step_reward=-0.01),
            TaskConfig(layout=layout_b, step_reward=-0.5),
        ]
        w = TaskSequenceWrapper(base_env, tasks)

        w.reset(seed=42)
        _, r1, _, _, _ = w.step(0)

        w.advance_task()
        w.reset(seed=42)
        _, r2, _, _, _ = w.step(0)

        # r2 should be more negative because step_reward is -0.5
        assert r2 < r1

    def test_state_space_changes_per_task(self, layout_a, layout_b):
        env = GridWorldEnv(layout_a, posner_mode=False)
        tasks = [
            TaskConfig(layout=layout_a),
            TaskConfig(layout=layout_b),
        ]
        w = TaskSequenceWrapper(env, tasks)

        w.reset(seed=42)
        size_a = env.state_space_size

        w.advance_task()
        w.reset(seed=42)
        size_b = env.state_space_size

        # layout_a has keys/doors/rewards → more states
        # layout_b has only 1 reward → fewer states
        assert size_a != size_b
