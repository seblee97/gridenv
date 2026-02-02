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

    def test_wrong_key_destroys_reward_immediately(self, key_door_env):
        """Test that collecting the wrong key destroys protected reward immediately."""
        key_door_env.reset(seed=42)

        # Get blue key (wrong) — reward should be destroyed on collection
        key_door_env.step(Action.DOWN)   # (1,1) -> (2,1)
        key_door_env.step(Action.DOWN)   # (2,1) -> (3,1)
        _, _, term, _, info = key_door_env.step(Action.RIGHT)  # (3,1) -> (3,2)
        assert key_door_env.held_key == KeyColor.BLUE

        # Reward is already destroyed before reaching the door
        assert info["destroyed_rewards"] > 0
        assert term  # Episode ends (all rewards resolved)


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


class TestStartPositionMode:
    """Tests for start position modes."""

    @pytest.fixture
    def room_layout(self):
        """Layout with a first room, door, and second room."""
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
        return parse_layout_string(layout, config)

    def test_fixed_mode_uses_start_position(self, room_layout):
        """Test that fixed mode always starts at layout S position."""
        env = GridWorldEnv(room_layout, start_pos_mode="fixed")
        for seed in range(10):
            env.reset(seed=seed)
            assert env.agent_position == (1, 1)

    def test_random_in_room_stays_in_first_room(self, room_layout):
        """Test that random_in_room only places agent in the first room."""
        env = GridWorldEnv(room_layout, start_pos_mode="random_in_room")
        positions = set()
        for seed in range(100):
            env.reset(seed=seed)
            pos = env.agent_position
            positions.add(pos)
            row, col = pos
            # Must be in the first room (rows 1-3, cols 1-5), not a wall or door
            assert 1 <= row <= 3 and 1 <= col <= 5, f"Agent at {pos} is outside first room"

        # With 100 seeds, should see more than one unique position
        assert len(positions) > 1

    def test_random_in_room_avoids_objects(self, room_layout):
        """Test that random start positions exclude key/reward cells."""
        env = GridWorldEnv(room_layout, start_pos_mode="random_in_room")
        object_positions = {(2, 2), (3, 2)}  # key positions
        for seed in range(200):
            env.reset(seed=seed)
            assert env.agent_position not in object_positions, (
                f"Agent placed on object at {env.agent_position}"
            )

    def test_invalid_start_pos_mode_raises(self):
        """Test that an invalid start_pos_mode raises ValueError."""
        layout = """
        ###
        #S#
        ###
        """
        with pytest.raises(ValueError):
            GridWorldEnv(layout, start_pos_mode="invalid")


class TestObsMode:
    """Tests for obs_mode parameter."""

    SIMPLE_LAYOUT = """
    #####
    #S..#
    #...#
    #..G#
    #####
    """

    def test_default_is_symbolic(self):
        """Default obs_mode is symbolic (backward compatible)."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_invalid_obs_mode_raises(self):
        """Invalid obs_mode raises ValueError."""
        with pytest.raises(ValueError):
            GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="invalid")

    def test_pixel_obs_shape_and_dtype(self):
        """Pixel observations have correct shape and dtype."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels")
        obs, _ = env.reset(seed=42)
        # 5 rows * 32 + 40 status bar = 200, 5 cols * 32 = 160
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (200, 160, 3)
        assert obs.dtype == np.uint8

    def test_pixel_obs_values_in_range(self):
        """Pixel observation values are valid uint8."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels")
        obs, _ = env.reset(seed=42)
        assert obs.min() >= 0
        assert obs.max() <= 255

    def test_pixel_obs_works_without_render_mode(self):
        """Pixel observations work with render_mode=None."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels", render_mode=None)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (200, 160, 3)

    def test_pixel_obs_changes_after_step(self):
        """Pixel observation changes when agent moves."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels")
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(Action.RIGHT)
        assert not np.array_equal(obs1, obs2)

    def test_both_mode_returns_dict(self):
        """Both mode returns dict with pixels and symbolic keys."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="both")
        obs, _ = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "pixels" in obs
        assert "symbolic" in obs

    def test_both_mode_pixel_shape(self):
        """Pixels in both mode have correct shape."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="both")
        obs, _ = env.reset(seed=42)
        assert obs["pixels"].shape == (200, 160, 3)
        assert obs["pixels"].dtype == np.uint8

    def test_both_mode_symbolic_flat(self):
        """Symbolic in both mode is flat array when flatten_obs=True."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="both", flatten_obs=True)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs["symbolic"], np.ndarray)
        assert obs["symbolic"].dtype == np.float32

    def test_both_mode_symbolic_dict(self):
        """Symbolic in both mode is dict when flatten_obs=False."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="both", flatten_obs=False)
        obs, _ = env.reset(seed=42)
        assert isinstance(obs["symbolic"], dict)
        assert "grid" in obs["symbolic"]
        assert "agent_pos" in obs["symbolic"]

    def test_observation_space_symbolic(self):
        """Observation space for symbolic mode is Box (flat) or Dict."""
        from gymnasium import spaces
        env_flat = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="symbolic", flatten_obs=True)
        assert isinstance(env_flat.observation_space, spaces.Box)

        env_dict = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="symbolic", flatten_obs=False)
        assert isinstance(env_dict.observation_space, spaces.Dict)

    def test_observation_space_pixels(self):
        """Observation space for pixel mode is Box with uint8."""
        from gymnasium import spaces
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels")
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.dtype == np.uint8
        assert env.observation_space.shape == (200, 160, 3)

    def test_observation_space_both(self):
        """Observation space for both mode is Dict with pixels and symbolic."""
        from gymnasium import spaces
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="both")
        assert isinstance(env.observation_space, spaces.Dict)
        assert "pixels" in env.observation_space.spaces
        assert "symbolic" in env.observation_space.spaces

    def test_observation_in_space(self):
        """Observations are contained in their observation_space for all modes."""
        for obs_mode in ("symbolic", "pixels", "both"):
            env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode=obs_mode)
            obs, _ = env.reset(seed=42)
            assert env.observation_space.contains(obs), (
                f"Observation not in space for obs_mode='{obs_mode}'"
            )

    def test_close_cleans_obs_renderer(self):
        """close() cleans up the observation renderer."""
        env = GridWorldEnv(self.SIMPLE_LAYOUT, obs_mode="pixels")
        env.reset(seed=42)
        assert env._obs_renderer is not None
        env.close()
        assert env._obs_renderer is None
