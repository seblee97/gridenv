"""Tests for game objects."""

import pytest

from gridworld_env.objects import Key, KeyColor, Door, Reward, KeyPair


class TestKeyColor:
    """Tests for KeyColor enum."""

    def test_from_string(self):
        """Test parsing colors from strings."""
        assert KeyColor.from_string("red") == KeyColor.RED
        assert KeyColor.from_string("RED") == KeyColor.RED
        assert KeyColor.from_string("r") == KeyColor.RED
        assert KeyColor.from_string("blue") == KeyColor.BLUE
        assert KeyColor.from_string("BLUE") == KeyColor.BLUE
        assert KeyColor.from_string("b") == KeyColor.BLUE

    def test_str(self):
        """Test string representation."""
        assert str(KeyColor.RED) == "red"
        assert str(KeyColor.BLUE) == "blue"


class TestKey:
    """Tests for Key class."""

    def test_creation(self):
        """Test key creation."""
        key = Key(position=(1, 2), color=KeyColor.RED)
        assert key.position == (1, 2)
        assert key.color == KeyColor.RED
        assert not key.collected

    def test_hash(self):
        """Test key is hashable."""
        key = Key(position=(1, 2), color=KeyColor.RED)
        key_set = {key}
        assert key in key_set


class TestDoor:
    """Tests for Door class."""

    def test_creation(self):
        """Test door creation."""
        door = Door(position=(2, 3), correct_key_color=KeyColor.BLUE)
        assert door.position == (2, 3)
        assert door.correct_key_color == KeyColor.BLUE
        assert not door.is_open
        assert not door.wrong_key_used

    def test_open_with_correct_key(self):
        """Test opening with correct key."""
        door = Door(position=(2, 3), correct_key_color=KeyColor.BLUE)
        result = door.open_with_key(KeyColor.BLUE)

        assert result
        assert door.is_open
        assert not door.wrong_key_used

    def test_open_with_wrong_key(self):
        """Test opening with wrong key."""
        door = Door(position=(2, 3), correct_key_color=KeyColor.BLUE)
        result = door.open_with_key(KeyColor.RED)

        assert result  # Door still opens
        assert door.is_open
        assert door.wrong_key_used


class TestReward:
    """Tests for Reward class."""

    def test_creation(self):
        """Test reward creation."""
        reward = Reward(position=(3, 4), value=5.0)
        assert reward.position == (3, 4)
        assert reward.value == 5.0
        assert not reward.collected
        assert not reward.destroyed

    def test_is_available(self):
        """Test availability check."""
        reward = Reward(position=(3, 4))
        assert reward.is_available()

        reward.collected = True
        assert not reward.is_available()

        reward.collected = False
        reward.destroyed = True
        assert not reward.is_available()

    def test_protected_by_door(self):
        """Test door protection."""
        door = Door(position=(2, 3), correct_key_color=KeyColor.RED)
        reward = Reward(position=(4, 3), protected_by_door=door)

        assert reward.protected_by_door == door


class TestKeyPair:
    """Tests for KeyPair class."""

    def test_creation(self):
        """Test key pair creation."""
        key1 = Key(position=(1, 1), color=KeyColor.RED)
        key2 = Key(position=(2, 1), color=KeyColor.BLUE)
        pair = KeyPair(keys=(key1, key2), room_id=0)

        assert pair.keys == (key1, key2)
        assert pair.room_id == 0
        assert pair.collected_key is None

    def test_get_available_keys(self):
        """Test getting available keys."""
        key1 = Key(position=(1, 1), color=KeyColor.RED)
        key2 = Key(position=(2, 1), color=KeyColor.BLUE)
        pair = KeyPair(keys=(key1, key2), room_id=0)

        available = pair.get_available_keys()
        assert len(available) == 2
        assert key1 in available
        assert key2 in available

    def test_collect_makes_other_disappear(self):
        """Test that collecting one key makes the other disappear."""
        key1 = Key(position=(1, 1), color=KeyColor.RED)
        key2 = Key(position=(2, 1), color=KeyColor.BLUE)
        pair = KeyPair(keys=(key1, key2), room_id=0)

        result = pair.collect(key1)
        assert result
        assert pair.collected_key == key1
        assert key1.collected
        assert key2.collected  # Other key also marked collected

        # No more available keys
        assert pair.get_available_keys() == []

    def test_cannot_collect_twice(self):
        """Test that only one key can be collected."""
        key1 = Key(position=(1, 1), color=KeyColor.RED)
        key2 = Key(position=(2, 1), color=KeyColor.BLUE)
        pair = KeyPair(keys=(key1, key2), room_id=0)

        pair.collect(key1)
        result = pair.collect(key2)

        assert not result
        assert pair.collected_key == key1
