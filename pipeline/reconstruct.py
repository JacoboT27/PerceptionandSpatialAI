"""
pipeline/reconstruct.py
-----------------------
Wraps AMB3R inference.
Takes a directory of images, returns pointmaps, colours, confidence, and camera poses.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


def _check_checkpoint(checkpoint_path: Path):
    """Verify the checkpoint file exists, with a helpful message if not."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"AMB3R checkpoint not found at: {checkpoint_path}\n"
            "\n"
            "  Download it by running:\n"
            "    bash checkpoints/download_weights.sh\n"
            "\n"
            "  Or download manually from:\n"
            "    https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view\n"
            "  and place it at: checkpoints/amb3r.pt"
        )


def run_amb3r(
    frames_dir: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    conf_thresh: float = 0.5,
    max_images: int = 150,
) -> Dict[str, np.ndarray]:
    """
    Run AMB3R feed-forward 3D reconstruction on a folder of images.

    Args:
        frames_dir      : Directory containing extracted .jpg frames.
        checkpoint_path : Path to the AMB3R .pt checkpoint.
        device          : 'cuda' or 'cpu'.
        conf_thresh     : Confidence threshold (used for display only here;
                          filtering happens at export time).
        max_images      : Maximum number of frames to feed into AMB3R.

    Returns:
        Dictionary with keys:
            pts         : (N, 3) float32 — world-space 3D points
            colors      : (N, 3) float32 — RGB colours in [0, 1]
            conf_sig    : (N,)   float32 — confidence scores in [0, 1]
            poses       : (T, 4, 4) float32 — camera-to-world poses per frame
            image_paths : list of Path — frame files used (for transforms.json)
    """
    checkpoint_path = Path(checkpoint_path)
    _check_checkpoint(checkpoint_path)

    print(f"  Loading AMB3R weights from: {checkpoint_path}")

    # --- Import AMB3R (must be inside Docker / conda env with amb3r installed) ---
    try:
        from amb3r.model import AMB3R
        from amb3r.datasets import Demo
    except ImportError:
        raise ImportError(
            "AMB3R package not found. Make sure you are running inside the "
            "Docker container or the correct conda environment.\n"
            "See README.md for setup instructions."
        )

    # --- Load model ---
    model = AMB3R()
    model.load_weights(str(checkpoint_path))
    model = model.to(device)
    model.eval()
    print("  Model loaded.")

    # --- Load frames as AMB3R dataset ---
    # AMB3R's Demo dataset loads images from a directory.
    # resolution=(518, 392) is the default used in AMB3R's own demo.py.
    dataset = Demo(
        ROOT=str(frames_dir),
        resolution=(518, 392),
        num_seq=1,
        full_video=True,
        kf_every=1,
        disable_crop=False,
        max_images=max_images,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    _, views_all = batch

    # Move tensor values to device (skip non-tensors)
    for key in views_all:
        if isinstance(views_all[key], torch.Tensor):
            views_all[key] = views_all[key].to(device)

    print(f"  Running inference on {views_all['images'].shape[1]} frames...")

    # --- Forward pass ---
    with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                        dtype=torch.bfloat16):
        with torch.no_grad():
            res = model(views_all)

    # --- Extract outputs ---
    # Debug: inspect model output structure
    print(f"  Model output type: {type(res)}")
    if isinstance(res, (list, tuple)):
        print(f"  Output is list of length {len(res)}")
        out = res[-1]
        if isinstance(out, dict):
            print(f"  Last element keys: {list(out.keys())}")
    elif isinstance(res, dict):
        print(f"  Output keys: {list(res.keys())}")
        out = res

    out = res[-1] if isinstance(res, (list, tuple)) else res

    pts    = out['world_points'].cpu().numpy().reshape(-1, 3)
    conf   = out['world_points_conf'].cpu().numpy().reshape(-1)
    colors = out['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3).cpu().numpy()
    poses  = out['pose'].cpu().numpy()[0]

    # Sigmoid-like confidence normalisation matching AMB3R's own demo
    conf_sig = (conf - 1) / conf

    # Clip colors to [0, 1] (model output may be in [-1, 1] or [0, 1])
    colors = np.clip(colors, 0.0, 1.0)

    # Collect image paths in the order they were loaded
    image_paths = sorted(frames_dir.glob("*.jpg"))[:max_images]

    return {
        "pts":         pts.astype(np.float32),
        "colors":      colors.astype(np.float32),
        "conf_sig":    conf_sig.astype(np.float32),
        "poses":       poses.astype(np.float32),
        "image_paths": image_paths,
    }