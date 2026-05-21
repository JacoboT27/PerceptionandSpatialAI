"""
pipeline/reconstruct.py
-----------------------
Wraps AMB3R inference.
Takes a directory of images, returns pointmaps, colours, confidence, and camera poses.

NOTE on dependency patching:
  - torch_scatter is the REAL package, built from source in the Dockerfile
    against torch 2.7.1 for sm_120. It is NOT patched. (The old pure-Python
    patch was incomplete — it lacked segment_csr, which PTV3's backend needs.)
  - pytorch3d genuinely cannot be compiled for this stack, so its two ops
    used by AMB3R (knn_points, knn_gather) are still monkey-patched below.
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


def _patch_pytorch3d():
    """
    pytorch3d fails to compile for this PyTorch + Blackwell stack.
    AMB3R only uses knn_points and knn_gather from pytorch3d.ops —
    both are straightforward to reimplement with native PyTorch.
    """
    import sys
    import types
    import torch

    def knn_points(p1, p2, lengths1=None, lengths2=None, K=1,
                   return_nn=False, return_sorted=True):
        # p1: (N, P1, D), p2: (N, P2, D)
        dists = torch.cdist(p1, p2)                          # (N, P1, P2)
        knn_dists, knn_idx = dists.topk(K, dim=-1, largest=False, sorted=return_sorted)

        class KNNOutput:
            def __init__(self, dists, idx):
                self.dists = dists   # (N, P1, K)
                self.idx   = idx     # (N, P1, K)
        return KNNOutput(knn_dists ** 2, knn_idx)  # cdist returns L2; pytorch3d returns squared

    def knn_gather(p, idx, lengths=None):
        # p:   (N, P2, D)
        # idx: (N, P1, K)
        # returns (N, P1, K, D)
        N, P2, D   = p.shape
        _, P1, K   = idx.shape
        idx_flat   = idx.reshape(N, P1 * K)                  # (N, P1*K)
        gathered   = torch.gather(
            p, 1, idx_flat.unsqueeze(-1).expand(N, P1 * K, D)
        )                                                      # (N, P1*K, D)
        return gathered.reshape(N, P1, K, D)

    # Build a minimal pytorch3d.ops sub-module
    ops_mod = types.ModuleType('pytorch3d.ops')
    ops_mod.knn_points = knn_points
    ops_mod.knn_gather  = knn_gather

    root_mod = types.ModuleType('pytorch3d')
    root_mod.ops = ops_mod

    sys.modules['pytorch3d']      = root_mod
    sys.modules['pytorch3d.ops']  = ops_mod
    print("  [patch] pytorch3d.ops replaced with native PyTorch implementation.")


def run_amb3r(
    frames_dir: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    conf_thresh: float = 0.5,
    max_images: int = 150,
    iters: int = 0,
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

    # pytorch3d cannot be compiled for this stack, so patch it before any
    # AMB3R imports. torch_scatter is the real, source-built package now —
    # it must NOT be patched (the old patch lacked segment_csr).
    _patch_pytorch3d()

    print(f"  Loading AMB3R weights from: {checkpoint_path}")

    # --- Import AMB3R (must be inside Docker / conda env with amb3r installed) ---
    try:
        from amb3r.model import AMB3R
        from amb3r.datasets import Demo
    except ModuleNotFoundError as e:
        if 'amb3r' in str(e):
            raise ImportError(
                "AMB3R package not found. Make sure you are running inside the "
                "Docker container or the correct conda environment.\n"
                "See README.md for setup instructions."
            ) from e
        raise  # re-raise any other ModuleNotFoundError with original traceback

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

    print(f"  Running inference on {views_all['images'].shape[1]} frames "
          f"(iters={iters}: {'front-end only' if iters == 0 else 'with back-end'})...")

    # --- Forward pass ---
    # AMB3R.forward(frames, iters) runs the VGGT front-end once, then `iters`
    # sparse-voxel back-end refinement passes. iters=0 -> front-end only: it
    # skips the entire memory-hungry back-end (the stage that OOMs on GPUs with
    # limited VRAM) and still returns full pointmaps, confidence, and poses.
    # This is exactly the demo's "0 / 1 : disable / enable backend" toggle.
    with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                        dtype=torch.bfloat16):
        with torch.no_grad():
            res = model(views_all, iters=iters)

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

    # Store original shape (T, H, W) for pixel-to-point mapping in semantic stage
    world_points_raw = out['world_points']
    pts_shape = tuple(world_points_raw.shape[1:4])  # (T, H, W)

    # Extract camera intrinsics from dataset (T, 3, 3)
    if 'camera_intrinsics' in views_all and isinstance(views_all['camera_intrinsics'], torch.Tensor):
        intrinsics = views_all['camera_intrinsics'].cpu().numpy()[0]  # (T, 3, 3)
    else:
        # Fallback: approximate intrinsics based on resolution
        T = poses.shape[0]
        H, W = pts_shape[1], pts_shape[2]
        fx = fy = max(H, W)
        intrinsics = np.tile(
            np.array([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=np.float32),
            (T, 1, 1)
        )

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
        "intrinsics":  intrinsics.astype(np.float32),  # (T, 3, 3) for semantic stage
        "pts_shape":   pts_shape,                       # (T, H, W) for semantic stage
        "image_paths": image_paths,
    }