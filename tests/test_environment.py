"""Tests for GridWorld environment."""

import numpy as np
import pytest

from gridworld_env import GridWorldEnv, KeyColor
from gridworld_env.environment import Action


class TestEnvironmentBasics:
    """Basic environment tests."""

    @pytest.fixture
    def simple_env(self):
        """Create a simple test environment."""
        layout = """
        #####
        #S..#
        #...#
        #..G#
        #####
        """
        return GridWorldEnv(layout)

    def test_reset(self, simple_env):
        """Test environment reset."""
        obs, info = simple_env.reset(seed=42)

        assert obs is not None
        assert isinstance(info, dict)
        assert simple_env.agent_position == (1, 1)

    def test_observation_shape_flat(self, simple_env):
        """Test flattened observation shape."""
        obs, _ = simple_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_observation_shape_dict(self):
        """Test dict observation shape."""
        layout = """
        #####
        #S..#
        #...#
        #####
        """
        env = GridWorldEnv(layout, flatten_obs=False)
        obs, _ = env.reset()

        assert isinstance(obs, dict)
        assert "grid" in obs
        assert "agent_pos" in obs
        assert "held_key" in obs
        assert "posner_cue" in obs

    def test_movement(self, simple_env):
        """Test basic movement."""
        simple_env.reset(seed=42)

        # Move right
        obs, reward, term, trunc, info = simple_env.step(Action.RIGHT)
        assert simple_env.agent_position == (1, 2)

        # Move down
        obs, reward, term, trunc, info = simple_env.step(Action.DOWN)
        assert simple_env.agent_position == (2, 2)

        # Move left
        obs, reward, term, trunc, info = simple_env.step(Action.LEFT)
        assert simple_env.agent_position == (2, 1)

        # Move up
        obs, reward, term, trunc, info = simple_env.step(Action.UP)
        assert simple_env.agent_position == (1, 1)

    def test_wall_collision(self, simple_env):
        """Test that walls block movement."""
        simple_env.reset(seed=42)

        # Try to move into wall (up from start)
        obs, reward, term, trunc, info = simple_env.step(Action.UP)
        assert simple_env.agent_position == (1, 1)  # Didn't move
        assert reward < 0  # Collision penalty

    def test_auto_collect_reward(self, simple_env):
        """Test automatic reward collection when stepping on it."""
        simple_env.reset(seed=42)

        # Move to reward position (3, 3)
        simple_env.step(Action.RIGHT)
        simple_env.step(Action.RIGHT)
        simple_env.step(Action.DOWN)
        obs, reward, term, trunc, info = simple_env.step(Action.DOWN)

        # Reward should be auto-collected
        assert reward > 0
        assert term  # Episode ends when all rewards collected

    def test_action_space(self, simple_env):
        """Test that action space is Discrete(4)."""
        assert simple_env.action_space.n == 4


class TestKeysAndDoors:
    """Tests for key and door mechanics."""

    @pytest.fixture
    def key_door_env(self):
        """Create environment with keys and door."""
        layout = """
        #######
        #S....#
        #.r...#
        #.b...#
        ###D###
        #..G..#
        #######
        """
        config = {
            "door_colors": {"4,3": "red"},
            "key_pairs": [{"positions": ["2,2", "3,2"], "room_id": 0}],
            "protected_rewards": {"4,3": ["5,3"]},
        }
        from gridworld_env.layout import parse_layout_string
        parsed_layout = parse_layout_string(layout, config)
        return GridWorldEnv(parsed_layout)

    def test_auto_collect_key(self, key_door_env):
        """Test automatic key collection when stepping on it."""
        key_door_env.reset(seed=42)

        # Move to red key position
        key_door_env.step(Action.DOWN)   # (1,1) -> (2,1)
        key_door_env.step(Action.RIGHT)  # (2,1) -> (2,2) red key

        # Key should be auto-collected
        assert key_door_env.held_key == KeyColor.RED

    def test_key_pair_other_disappears(self, key_door_env):
        """Test that collecting one key makes the other disappear."""
        key_door_env.reset(seed=42)

        # Move to red key (auto-collects)
        key_door_env.step(Action.DOWN)
        key_door_env.step(Action.RIGHT)
        assert key_door_env.held_key == KeyColor.RED

        # Move to where blue key was
        key_door_env.step(Action.DOWN)  # (2,2) -> (3,2)

        # Should still have red key (blue key disappeared)
        assert key_door_env.held_key == KeyColor.RED

    def test_door_blocks_without_key(self, key_door_env):
        """Test that closed door blocks movement without key."""
        key_door_env.reset(seed=42)

        # Try to walk to door without key
        key_door_env.step(Action.RIGHT)  # (1,1) -> (1,2)
        key_door_env.step(Action.RIGHT)  # (1,2) -> (1,3)
        key_door_env.step(Action.DOWN)   # (1,3) -> (2,3)
        key_door_env.step(Action.DOWN)   # (2,3) -> (3,3)
        key_door_env.step(Action.DOWN)   # Try to enter door - blocked

        # Should be blocked at (3,3)
        assert key_door_env.agent_position == (3, 3)

    def test_door_opens_with_correct_key(self, key_door_env):
        """Test that door opens automatically with correct key."""
        key_door_env.reset(seed=42)

        # Get red key (correct)
        key_door_env.step(Action.DOWN)   # (1,1) -> (2,1)
        key_door_env.step(Action.RIGHT)  # (2,1) -> (2,2) - auto-collect red key
        assert key_door_env.held_key == KeyColor.RED

        # Go to door
        key_door_env.step(Action.RIGHT)  # (2,2) -> (2,3)
        key_door_env.step(Action.DOWN)   # (2,3) -> (3,3)

        # Move into door - should auto-open and move through
        key_door_env.step(Action.DOWN)   # (3,3) -> (4,3) door opens, agent enters

        # Key should be consumed
        assert key_door_env.held_key is None
        assert key_door_env.agent_position == (4, 3)

        # Move to reward and collect
        obs, reward, term, trunc, info = key_door_env.step(Action.DOWN)  # (4,3) -> (5,3)

        assert reward > 0  # Reward was preserved and auto-collected
        assert info["destroyed_rewards"] == 0

    def test_door_opens_with_wrong_key_destroys_reward(self, key_door_env):
        """Test that wrong key destroys protected reward."""
        key_door_env.reset(seed=42)

        # Get blue key (wrong)
        key_door_env.step(Action.DOWN)   # (1,1) -> (2,1)
        key_door_env.step(Action.DOWN)   # (2,1) -> (3,1)
        key_door_env.step(Action.RIGHT)  # (3,1) -> (3,2) - auto-collect blue key
        assert key_door_env.held_key == KeyColor.BLUE

        # Go to door
        key_door_env.step(Action.RIGHT)  # (3,2) -> (3,3)

        # Move into door - should auto-open with wrong key
        key_door_env.step(Action.DOWN)   # (3,3) -> (4,3)
        assert key_door_env.held_key is None  # Key consumed

        # Move to reward position
        obs, reward, term, trunc, info = key_door_env.step(Action.DOWN)

        # Reward was destroyed by wrong key
        assert info["destroyed_rewards"] > 0


class TestPosnerMode:
    """Tests for Posner cueing mode."""

    @pytest.fixture
    def posner_env(self):
        """Create environment with Posner mode enabled (single feature)."""
        layout = """
        #######
        #S....#
        #.r...#
        #.b...#
        ###D###
        #..G..#
        #######
        """
        config = {
            "door_colors": {"4,3": "red"},
            "key_pairs": [{"positions": ["2,2", "3,2"], "room_id": 0}],
        }
        from gridworld_env.layout import parse_layout_string
        parsed_layout = parse_layout_string(layout, config)
        return GridWorldEnv(
            parsed_layout,
            posner_mode=True,
            posner_validity=0.8,
        )

    @pytest.fixture
    def multi_feature_env(self):
        """Create environment with multi-feature Posner cues."""
        layout = """
        #######
        #S....#
        #.r...#
        #.b...#
        ###D###
        #..G..#
        #######
        """
        config = {
            "door_colors": {"4,3": "red"},
            "key_pairs": [{"positions": ["2,2", "3,2"], "room_id": 0}],
        }
        from gridworld_env.layout import parse_layout_string
        parsed_layout = parse_layout_string(layout, config)
        return GridWorldEnv(
            parsed_layout,
            posner_mode=True,
            posner_validity=0.8,
            posner_num_features=5,
            posner_cue_index=2,  # True cue is at index 2
        )

    def test_posner_cue_present(self, posner_env):
        """Test that Posner cue is generated."""
        obs, info = posner_env.reset(seed=42)

        assert posner_env.posner_cue is not None
        assert posner_env.posner_cue in [KeyColor.RED, KeyColor.BLUE]
        assert "posner_cue" in info
        assert "posner_cue_valid" in info

    def test_posner_validity_distribution(self, posner_env):
        """Test that cue validity follows expected distribution."""
        valid_count = 0
        n_trials = 1000

        for seed in range(n_trials):
            obs, info = posner_env.reset(seed=seed)
            if info["posner_cue_valid"]:
                valid_count += 1

        # Should be approximately 80% valid
        validity_rate = valid_count / n_trials
        assert 0.7 < validity_rate < 0.9

    def test_posner_cue_in_observation(self, posner_env):
        """Test that Posner cue is included in observation."""
        obs, _ = posner_env.reset(seed=42)

        # For flattened obs, cue should be in last 3 elements (one-hot)
        assert isinstance(obs, np.ndarray)
        posner_cue_vec = obs[-3:]
        assert posner_cue_vec.sum() == 1.0  # One-hot encoded

    def test_multi_feature_cues_present(self, multi_feature_env):
        """Test that multi-feature cues are generated."""
        obs, info = multi_feature_env.reset(seed=42)

        assert multi_feature_env.posner_cues is not None
        assert len(multi_feature_env.posner_cues) == 5
        assert all(c in [KeyColor.RED, KeyColor.BLUE] for c in multi_feature_env.posner_cues)
        assert "posner_cues" in info
        assert info["posner_cue_index"] == 2

    def test_multi_feature_cue_validity(self, multi_feature_env):
        """Test that only the true cue (at posner_cue_index) follows validity."""
        valid_count = 0
        n_trials = 1000

        for seed in range(n_trials):
            obs, info = multi_feature_env.reset(seed=seed)
            if info["posner_cue_valid"]:
                valid_count += 1

        # True cue at index 2 should be approximately 80% valid
        validity_rate = valid_count / n_trials
        assert 0.7 < validity_rate < 0.9

    def test_multi_feature_observation_shape(self, multi_feature_env):
        """Test that observation includes all cue features."""
        obs, _ = multi_feature_env.reset(seed=42)

        # For flattened obs, last 15 elements are the 5 cue features (5 * 3 one-hot)
        assert isinstance(obs, np.ndarray)
        posner_cue_vec = obs[-15:]

        # Each feature should be one-hot (sum of each group of 3 should be 1)
        for i in range(5):
            feature_vec = posner_cue_vec[i*3:(i+1)*3]
            assert feature_vec.sum() == 1.0

    def test_multi_feature_dict_observation(self):
        """Test dict observation with multi-feature cues."""
        layout = """
        #######
        #S....#
        #.r...#
        #.b...#
        ###D###
        #..G..#
        #######
        """
        config = {
            "door_colors": {"4,3": "red"},
            "key_pairs": [{"positions": ["2,2", "3,2"], "room_id": 0}],
        }
        from gridworld_env.layout import parse_layout_string
        parsed_layout = parse_layout_string(layout, config)
        env = GridWorldEnv(
            parsed_layout,
            posner_mode=True,
            posner_validity=0.8,
            posner_num_features=3,
            posner_cue_index=1,
            flatten_obs=False,
        )
        obs, info = env.reset(seed=42)

        assert isinstance(obs, dict)
        assert "posner_cue" in obs
        assert obs["posner_cue"].shape == (3,)
        assert all(v in [1, 2] for v in obs["posner_cue"])  # 1=red, 2=blue

    def test_invalid_cue_index_raises(self):
        """Test that invalid posner_cue_index raises ValueError."""
        layout = """
        ###
        #S#
        ###
        """
        with pytest.raises(ValueError):
            GridWorldEnv(
                layout,
                posner_mode=True,
                posner_num_features=3,
                posner_cue_index=5,  # Invalid: out of range
            )


class TestMaxSteps:
    """Tests for max steps truncation."""

    def test_truncation(self):
        """Test episode truncates at max steps."""
        layout = """
        #####
        #S..#
        #...#
        #..G#
        #####
        """
        env = GridWorldEnv(layout, max_steps=5)
        env.reset(seed=42)

        for i in range(5):
            obs, reward, term, trunc, info = env.step(Action.RIGHT)

        assert trunc
        assert info["steps"] == 5


class TestInfo:
    """Tests for info dict."""

    def test_info_contents(self):
        """Test info dict contains expected keys."""
        layout = """
        #####
        #S.G#
        #####
        """
        env = GridWorldEnv(layout)
        obs, info = env.reset()

        assert "steps" in info
        assert "collected_rewards" in info
        assert "destroyed_rewards" in info
        assert "available_rewards" in info
        assert "held_key" in info
        assert "agent_pos" in info
