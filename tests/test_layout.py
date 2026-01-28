"""Tests for layout parsing."""

import pytest

from gridworld_env.layout import parse_layout_string, Layout
from gridworld_env.objects import KeyColor


class TestLayoutParsing:
    """Tests for parsing ASCII layouts."""

    def test_simple_grid(self):
        """Test parsing a simple grid with walls."""
        layout_str = """
        #####
        #S..#
        #...#
        #####
        """
        layout = parse_layout_string(layout_str)

        assert layout.width == 5
        assert layout.height == 4
        assert layout.start_position == (1, 1)
        assert layout.is_wall(0, 0)
        assert layout.is_wall(0, 4)
        assert layout.is_wall(3, 2)
        assert not layout.is_wall(1, 2)
        assert not layout.is_wall(2, 2)

    def test_rewards(self):
        """Test parsing rewards."""
        layout_str = """
        #####
        #S..#
        #.G.#
        #####
        """
        layout = parse_layout_string(layout_str)

        assert len(layout.rewards) == 1
        assert layout.rewards[0].position == (2, 2)
        assert layout.rewards[0].value == 1.0

    def test_reward_with_value(self):
        """Test parsing rewards with custom values."""
        layout_str = """
        #####
        #S..#
        #.G.#
        #####
        """
        config = {"reward_values": {"2,2": 5.0}}
        layout = parse_layout_string(layout_str, config)

        assert layout.rewards[0].value == 5.0

    def test_keys(self):
        """Test parsing keys."""
        layout_str = """
        #####
        #S..#
        #rb.#
        #####
        """
        layout = parse_layout_string(layout_str)

        assert len(layout.keys) == 2
        red_key = next(k for k in layout.keys if k.color == KeyColor.RED)
        blue_key = next(k for k in layout.keys if k.color == KeyColor.BLUE)
        assert red_key.position == (2, 1)
        assert blue_key.position == (2, 2)

    def test_doors(self):
        """Test parsing doors."""
        layout_str = """
        #####
        #S..#
        ##D##
        #...#
        #####
        """
        config = {"door_colors": {"2,2": "blue"}}
        layout = parse_layout_string(layout_str, config)

        assert len(layout.doors) == 1
        assert layout.doors[0].position == (2, 2)
        assert layout.doors[0].correct_key_color == KeyColor.BLUE
        assert not layout.doors[0].is_open

    def test_key_pairs(self):
        """Test automatic key pair creation."""
        layout_str = """
        #######
        #S....#
        #.r...#
        #.b...#
        #######
        """
        layout = parse_layout_string(layout_str)

        # Keys should be auto-paired since they're adjacent and different colors
        assert len(layout.key_pairs) == 1
        pair = layout.key_pairs[0]
        colors = {k.color for k in pair.keys}
        assert colors == {KeyColor.RED, KeyColor.BLUE}

    def test_protected_rewards(self):
        """Test door-reward protection mapping."""
        layout_str = """
        #######
        #S....#
        ###D###
        #..G..#
        #######
        """
        config = {
            "door_colors": {"2,3": "red"},
            "protected_rewards": {"2,3": ["3,3"]},
        }
        layout = parse_layout_string(layout_str, config)

        door = layout.doors[0]
        reward = layout.rewards[0]
        assert reward.protected_by_door == door


class TestLayoutMethods:
    """Tests for Layout class methods."""

    def test_is_valid_position(self):
        """Test position validation."""
        layout_str = """
        #####
        #S..#
        #...#
        #####
        """
        layout = parse_layout_string(layout_str)

        assert layout.is_valid_position(1, 1)
        assert layout.is_valid_position(2, 3)
        assert not layout.is_valid_position(0, 0)  # Wall
        assert not layout.is_valid_position(-1, 0)  # Out of bounds
        assert not layout.is_valid_position(5, 0)  # Out of bounds

    def test_get_door_at(self):
        """Test getting door at position."""
        layout_str = """
        #####
        #S..#
        ##D##
        #...#
        #####
        """
        layout = parse_layout_string(layout_str)

        door = layout.get_door_at(2, 2)
        assert door is not None
        assert door.position == (2, 2)

        assert layout.get_door_at(1, 1) is None

    def test_get_key_at(self):
        """Test getting key at position."""
        layout_str = """
        #####
        #Sr.#
        #...#
        #####
        """
        layout = parse_layout_string(layout_str)

        key = layout.get_key_at(1, 2)
        assert key is not None
        assert key.color == KeyColor.RED

        assert layout.get_key_at(1, 1) is None

    def test_copy(self):
        """Test deep copy of layout."""
        layout_str = """
        #####
        #SG.#
        #...#
        #####
        """
        layout = parse_layout_string(layout_str)
        layout_copy = layout.copy()

        # Modify original
        layout.rewards[0].collected = True

        # Copy should be unchanged
        assert not layout_copy.rewards[0].collected
