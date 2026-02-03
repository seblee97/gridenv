# GridWorld Environment

A configurable grid world reinforcement learning environment with keys, doors, and Posner cueing mode.

## Features

- **Configurable layouts** via ASCII files or strings
- **Walls** that block agent movement
- **Rewards** that can be placed anywhere and collected
- **Keys** of different colors (red/blue)
- **Doors** that require keys to open
- **Key pairs** where collecting one key makes the other disappear
- **Wrong key penalty**: using the wrong key destroys rewards behind the door
- **Posner mode**: cues indicate which key to collect (with configurable validity)

## Installation

```bash
# From local directory
pip install -e /path/to/gridworld_env

# With rendering support
pip install -e "/path/to/gridworld_env[render]"

# With dev dependencies
pip install -e "/path/to/gridworld_env[dev]"
```

## Quick Start

```python
from gridworld_env import GridWorldEnv

# Create from ASCII string
layout = """
#######
#S....#
#.r...#
#.b...#
###D###
#..G..#
#######
"""

env = GridWorldEnv(layout)
obs, info = env.reset()

# Take actions
obs, reward, terminated, truncated, info = env.step(0)  # Move up
```

## Layout Format

Layouts are defined using ASCII characters:

| Character | Meaning |
|-----------|---------|
| `#` | Wall |
| `.` | Empty floor |
| `S` | Start position |
| `G` | Goal/Reward (value 1.0) |
| `R` | Reward (can have value suffix like `R10`) |
| `D` | Door |
| `r` | Red key |
| `b` | Blue key |

### Layout Files

Create a `.txt` file with the ASCII layout and an optional `.json` config file:

**simple_room.txt**
```
#########
#S......#
#...r...#
#...b...#
####D####
#...G...#
#########
```

**simple_room.json**
```json
{
    "default_correct_keys": {
        "4,4": "red"
    },
    "key_pairs": [
        {
            "positions": ["2,4", "3,4"],
            "room_id": 0
        }
    ],
    "protected_rewards": {
        "4,4": ["5,4"]
    }
}
```

Load with:
```python
env = GridWorldEnv("path/to/simple_room.txt")
```

## Configuration

### Environment Parameters

```python
env = GridWorldEnv(
    layout,                    # Layout object, file path, or ASCII string
    posner_mode=False,         # Enable Posner cueing
    posner_validity=0.8,       # Probability cue is correct (0.0-1.0)
    random_key_colors=False,   # Randomize key colors per episode (overrides .txt)
    random_correct_key=False,  # Randomize correct key per episode (overrides .json)
    max_steps=None,            # Maximum steps per episode
    step_reward=-0.01,         # Reward per step (negative for time pressure)
    collision_reward=-0.1,     # Penalty for hitting walls
    flatten_obs=True,          # Flatten observations to 1D
    render_mode=None,          # 'human', 'rgb_array', or None
)
```

### Layout Configuration

The JSON config file supports:

- **default_correct_keys**: Map door positions to default correct key colors.
  Used when `random_correct_key=False`. Overridden per-episode when `random_correct_key=True`.
  ```json
  {"4,4": "red", "8,4": "blue"}
  ```

- **key_pairs**: Define which keys are mutually exclusive
  ```json
  [{"positions": ["2,4", "3,4"], "room_id": 0}]
  ```

- **protected_rewards**: Map doors to rewards they protect
  ```json
  {"4,4": ["5,4", "5,5"]}
  ```

- **reward_values**: Custom reward values
  ```json
  {"5,4": 10.0}
  ```

## Actions

| Action | Value | Description |
|--------|-------|-------------|
| UP | 0 | Move up |
| DOWN | 1 | Move down |
| LEFT | 2 | Move left |
| RIGHT | 3 | Move right |

Keys and rewards are collected automatically when the agent steps onto them. Doors are opened automatically when the agent moves into them while holding a key.

## Observations

### Flattened (default)
When `flatten_obs=True`, observations are a 1D float32 array containing:
- One-hot encoded grid cells
- Normalized agent position (2 values)
- Held key one-hot (3 values: none/red/blue)
- Posner cue one-hot (3 values: none/red/blue)

### Dictionary
When `flatten_obs=False`, observations are a dict:
```python
{
    "grid": np.array,      # 2D grid of cell types
    "agent_pos": np.array, # [row, col]
    "held_key": int,       # 0=none, 1=red, 2=blue
    "posner_cue": int,     # 0=none, 1=red, 2=blue
}
```

## Posner Mode

In Posner mode, a cue is shown at the start of each episode indicating which key to collect. The cue has a configurable validity (probability of being correct).

```python
env = GridWorldEnv(
    layout,
    posner_mode=True,
    posner_validity=0.8,  # 80% chance cue is correct
)

obs, info = env.reset()
print(f"Cue: {info['posner_cue']}")
print(f"Valid: {info['posner_cue_valid']}")
```

## Key-Door Mechanics

1. **Key pairs**: In each room, two keys (red and blue) can be placed. Collecting one makes the other disappear.

2. **Doors**: Require any key to open, but each door has a "correct" key color.

3. **Wrong key penalty**: If the wrong key is used, rewards behind that door are destroyed.

## Example: Multi-Room Navigation

```python
from gridworld_env import GridWorldEnv

layout = """
###############
#S............#
#..r..........#
#..b..........#
######D########
#.............#
#.....r.......#
#.....b.......#
######D########
#......G......#
###############
"""

config = {
    "default_correct_keys": {"4,6": "blue", "8,6": "red"},
    "key_pairs": [
        {"positions": ["2,3", "3,3"], "room_id": 0},
        {"positions": ["6,6", "7,6"], "room_id": 1}
    ],
    "protected_rewards": {"8,6": ["9,7"]}
}

from gridworld_env.layout import parse_layout_string
parsed = parse_layout_string(layout, config)
env = GridWorldEnv(parsed, posner_mode=True, posner_validity=0.75)

# Run episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward}")
print(f"Collected: {info['collected_rewards']}")
print(f"Destroyed: {info['destroyed_rewards']}")
```

## Rendering

```python
# ASCII rendering (always available)
env = GridWorldEnv(layout, render_mode="human")
env.reset()
env.render()  # Prints ASCII to console

# Pygame rendering (requires pygame)
env = GridWorldEnv(layout, render_mode="human")
env.reset()
env.render()  # Opens pygame window

# RGB array for recording
env = GridWorldEnv(layout, render_mode="rgb_array")
frame = env.render()  # Returns numpy array
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
