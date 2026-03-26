#!/usr/bin/env python
"""Render oracle trajectories as annotated videos with model predictions.

Overlays ground truth AND predicted phase/material/room labels alongside
the game screen (not on top of it).

Usage
-----
    python scripts/render_trajectories_with_predictions.py \
        --dataset datasets/oracle/directional \
        --checkpoint-dir ../recurrent-predictive-learning/checkpoints/maze_directional_pred \
        --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import imageio.v3 as iio

from gridworld_env.modular_maze import ModularMazeEnv
from gridworld_env.procgen import generate_world_grid
from gridworld_env.oracle_policy import PHASE_NAMES, MATERIAL_NAMES

# Add RPL to path for model loading
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "recurrent-predictive-learning"))

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "USE_KEY", "COLLECT_KEY", "ENGAGE", "FORGE_KEY"]

PHASE_COLORS = {
    "get_key":          (60, 180, 75),
    "make_key":         (245, 130, 48),
    "get_key_location": (145, 30, 180),
    "collect_reward":   (255, 215, 0),
    "goto_next_room":   (70, 130, 180),
}

MATERIAL_COLORS = {
    "none":      (128, 128, 128),
    "diamond":   (0, 210, 210),
    "ruby":      (210, 50, 50),
    "sapphire":  (50, 80, 210),
}


def load_model_and_readout(checkpoint_dir: Path, device: torch.device):
    """Load the trained RPL model and offline readout."""
    from repl.networks.utils import prepare_model
    from repl.networks.networks import LinearReadout
    from repl.scripts.eval_utils import prepare_readout
    from repl.scripts.utils import get_data_specs

    # Reconstruct model using the same factory as cli.py
    input_size, num_classes = get_data_specs(dataset="maze")
    model = prepare_model(
        encoder_kind="mlp",
        integrator_kind="lstm",
        predictor_kind="mlp",
        input_size=input_size,
        enc_dim=64,
        ctx_dim=256,
        pred_n_hidden_layers=1,
        pred_hidden_dim=256,
        pred_steps=1,
        enc_n_layers=2,
    )

    model_state = torch.load(checkpoint_dir / "model_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Reconstruct readout using the same factory
    readout = prepare_readout(
        task="multitask",
        downstream_input="ctx",
        num_classes=num_classes,
        seq_len=100,
        enc_output_dim=64,
        ctx_dim=256,
        pred_steps=1,
        model=model,
    )
    readout_state = torch.load(checkpoint_dir / "offline_ctx_readout.pt", map_location=device, weights_only=False)
    readout.load_state_dict(readout_state)
    readout.to(device)
    readout.eval()

    return model, readout


@torch.no_grad()
def predict_batch(model, readout, actions_batch: np.ndarray, device: torch.device):
    """Run model + readout on a batch of action sequences.

    Returns per-timestep predictions: (pred_phase, pred_material, pred_room) each (N, T).
    """
    N, T = actions_batch.shape
    x = F.one_hot(torch.tensor(actions_batch, dtype=torch.long), num_classes=8).float()  # (N, T, 8)
    x = x.permute(0, 2, 1).to(device)  # (N, 8, T) channels first

    z, context, pred = model(x)  # context: (N, T, 256)

    # Dense readouts expect (B*L, C)
    ctx_flat = context.reshape(-1, context.shape[-1])  # (N*T, 256)

    pred_phase = readout[1](context).argmax(dim=-1).reshape(N, T)      # (N*T, 5) -> (N, T)
    pred_material = readout[2](context).argmax(dim=-1).reshape(N, T)   # (N*T, 4) -> (N, T)
    pred_room = readout[3](context).argmax(dim=-1).reshape(N, T)       # (N*T, 4) -> (N, T)

    return pred_phase.cpu().numpy(), pred_material.cpu().numpy(), pred_room.cpu().numpy()


def annotate_frame(
    frame: np.ndarray,
    step: int,
    action: int,
    gt_phase: int,
    gt_material: int,
    gt_room: int,
    pred_phase: int,
    pred_material: int,
    pred_room: int,
    font,
    panel_width: int = 400,
) -> np.ndarray:
    """Create a frame with game screen on left and annotation panel on right."""
    h, w, c = frame.shape

    # Create wider canvas with panel on the right
    canvas = np.zeros((h, w + panel_width, c), dtype=np.uint8)
    canvas[:, :w, :] = frame
    canvas[:, w:, :] = 30  # dark gray panel background

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    gt_phase_name = PHASE_NAMES[gt_phase]
    gt_mat_name = MATERIAL_NAMES[gt_material]
    pred_phase_name = PHASE_NAMES[pred_phase]
    pred_mat_name = MATERIAL_NAMES[pred_material]
    action_name = ACTION_NAMES[action]

    x0 = w + 12
    y0 = 12
    line_h = 22

    # Header
    draw.text((x0, y0), f"step {step:>3d}   action: {action_name}", fill=(255, 255, 255), font=font)
    y0 += line_h + 8

    # Separator
    draw.line([(x0, y0), (x0 + panel_width - 24, y0)], fill=(100, 100, 100), width=1)
    y0 += 8

    # Ground truth section
    draw.text((x0, y0), "GROUND TRUTH", fill=(180, 180, 180), font=font)
    y0 += line_h + 2

    gt_phase_col = PHASE_COLORS.get(gt_phase_name, (200, 200, 200))
    gt_mat_col = MATERIAL_COLORS.get(gt_mat_name, (128, 128, 128))

    draw.text((x0, y0), f"  phase:    {gt_phase_name}", fill=gt_phase_col, font=font)
    y0 += line_h
    draw.text((x0, y0), f"  material: {gt_mat_name}", fill=gt_mat_col, font=font)
    y0 += line_h
    draw.text((x0, y0), f"  room:     {gt_room}", fill=(255, 255, 255), font=font)
    y0 += line_h + 8

    # Separator
    draw.line([(x0, y0), (x0 + panel_width - 24, y0)], fill=(100, 100, 100), width=1)
    y0 += 8

    # Prediction section
    draw.text((x0, y0), "PREDICTED", fill=(180, 180, 180), font=font)
    y0 += line_h + 2

    pred_phase_col = PHASE_COLORS.get(pred_phase_name, (200, 200, 200))
    pred_mat_col = MATERIAL_COLORS.get(pred_mat_name, (128, 128, 128))

    # Mark correct/incorrect with checkmark/cross
    phase_match = "+" if gt_phase == pred_phase else "X"
    mat_match = "+" if gt_material == pred_material else "X"
    room_match = "+" if gt_room == pred_room else "X"

    match_col = lambda m: (100, 255, 100) if m == "+" else (255, 80, 80)

    draw.text((x0, y0), f"  phase:    {pred_phase_name}", fill=pred_phase_col, font=font)
    draw.text((x0 + panel_width - 50, y0), f"[{phase_match}]", fill=match_col(phase_match), font=font)
    y0 += line_h
    draw.text((x0, y0), f"  material: {pred_mat_name}", fill=pred_mat_col, font=font)
    draw.text((x0 + panel_width - 50, y0), f"[{mat_match}]", fill=match_col(mat_match), font=font)
    y0 += line_h
    draw.text((x0, y0), f"  room:     {pred_room}", fill=(255, 255, 255), font=font)
    draw.text((x0 + panel_width - 50, y0), f"[{room_match}]", fill=match_col(room_match), font=font)

    return np.array(img)


def render_trajectory(
    env: ModularMazeEnv,
    traj_seed: int,
    start_pos: tuple,
    actions: np.ndarray,
    gt_phases: np.ndarray,
    gt_materials: np.ndarray,
    gt_rooms: np.ndarray,
    pred_phases: np.ndarray,
    pred_materials: np.ndarray,
    pred_rooms: np.ndarray,
    font,
) -> list[np.ndarray]:
    """Replay a trajectory and return annotated frames."""
    env.reset(seed=traj_seed)
    env._agent_pos = tuple(start_pos)
    env._last_valid_room = env._cell_to_room.get(tuple(start_pos), 0)

    frames = []
    for t in range(len(actions)):
        raw = env.render()
        frame = annotate_frame(
            raw, t, int(actions[t]),
            int(gt_phases[t]), int(gt_materials[t]), int(gt_rooms[t]),
            int(pred_phases[t]), int(pred_materials[t]), int(pred_rooms[t]),
            font,
        )
        frames.append(frame)
        env.step(int(actions[t]))

    return frames


def main():
    p = argparse.ArgumentParser(description="Render trajectories with GT + predicted labels")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Path to RPL checkpoint dir (e.g. checkpoints/maze_directional_pred)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    ds_path = Path(args.dataset)
    ckpt_path = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ds_path / "videos_with_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata + labels
    with open(ds_path / "metadata.json") as f:
        meta = json.load(f)
    npz = np.load(ds_path / "labels.npz")
    actions = npz["actions"]
    phases = npz["phase_labels"]
    materials = npz["material_labels"]
    rooms = npz["room_labels"]
    start_positions = npz["start_positions"]
    completion = npz["completion_steps"]

    # Load model + readout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {ckpt_path} ...")
    model, readout = load_model_and_readout(ckpt_path, device)

    # Run inference on selected trajectories
    indices = list(range(args.offset, args.offset + args.n))
    actions_batch = actions[indices]  # (n, T)
    pred_phases, pred_materials, pred_rooms = predict_batch(model, readout, actions_batch, device)

    # Compute accuracies for these trajectories
    T = actions_batch.shape[1]
    for j, i in enumerate(indices):
        ph_acc = (pred_phases[j] == phases[i, :T]).mean()
        mt_acc = (pred_materials[j] == materials[i, :T]).mean()
        rm_acc = (pred_rooms[j] == rooms[i, :T]).mean()
        print(f"  traj {i}: phase={ph_acc:.1%}  material={mt_acc:.1%}  room={rm_acc:.1%}")

    # Create render env
    layout = generate_world_grid(
        n_rooms=meta["n_rooms"], room_h=meta["room_h"], room_w=meta["room_w"],
        seed=meta["layout_seed"],
    )
    env = ModularMazeEnv(
        layout, obs_mode="symbolic", render_mode="rgb_array",
        max_steps=meta["seq_len"] + 10,
        terminate_on_all_safes_opened=False,
    )

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for j, i in enumerate(indices):
        traj_seed = meta["layout_seed"] + 1000 + i
        comp_str = f"done@{completion[i]}" if completion[i] > 0 else "incomplete"

        print(f"Rendering traj {i} ({comp_str})...")
        frames = render_trajectory(
            env, traj_seed, start_positions[i],
            actions[i], phases[i], materials[i], rooms[i],
            pred_phases[j], pred_materials[j], pred_rooms[j],
            font,
        )

        out_path = out_dir / f"traj_{i:05d}.mp4"
        iio.imwrite(str(out_path), frames, fps=args.fps, codec="libx264")
        print(f"  -> {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
