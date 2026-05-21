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


def _patch_torch_scatter():
    """
    torch_scatter 2.1.2 uses a C++ symbol (torch::jit::parseSchemaOrName)
    that was removed in PyTorch 2.6, causing an OSError on import.

    This patches sys.modules with a pure-PyTorch implementation of the
    scatter operations AMB3R needs, so torch_scatter never gets loaded.
    """
    import sys
    import types
    import torch

    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        size = list(src.size())
        size[dim] = dim_size
        result = torch.zeros(size, dtype=src.dtype, device=src.device)
        count  = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        idx_exp = index.view(
            *([1] * dim), -1, *([1] * (src.dim() - dim - 1))
        ).expand_as(src)
        result.scatter_add_(dim, idx_exp, src)
        count.scatter_add_(0, index.reshape(-1),
                           torch.ones(index.numel(), dtype=src.dtype, device=src.device))
        count = count.clamp(min=1)
        shape = [dim_size if i == dim else 1 for i in range(result.dim())]
        return result / count.view(shape)

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        size = list(src.size())
        size[dim] = dim_size
        if out is None:
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
        idx_exp = index.view(
            *([1] * dim), -1, *([1] * (src.dim() - dim - 1))
        ).expand_as(src)
        return out.scatter_add_(dim, idx_exp, src)

    def scatter_sum(src, index, dim=0, out=None, dim_size=None):
        return scatter_add(src, index, dim=dim, out=out, dim_size=dim_size)

    def scatter_max(src, index, dim=0, out=None, dim_size=None, fill_value=0):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        size = list(src.size())
        size[dim] = dim_size
        result = src.new_full(size, fill_value)
        idx_exp = index.view(
            *([1] * dim), -1, *([1] * (src.dim() - dim - 1))
        ).expand_as(src)
        result.scatter_reduce_(dim, idx_exp, src, reduce='amax', include_self=True)
        return result, result.new_zeros(size, dtype=torch.long)

    def scatter_min(src, index, dim=0, out=None, dim_size=None, fill_value=0):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        size = list(src.size())
        size[dim] = dim_size
        result = src.new_full(size, fill_value)
        idx_exp = index.view(
            *([1] * dim), -1, *([1] * (src.dim() - dim - 1))
        ).expand_as(src)
        result.scatter_reduce_(dim, idx_exp, src, reduce='amin', include_self=True)
        return result, result.new_zeros(size, dtype=torch.long)

    mod = types.ModuleType('torch_scatter')
    mod.scatter_mean = scatter_mean
    mod.scatter_add  = scatter_add
    mod.scatter_sum  = scatter_sum
    mod.scatter_max  = scatter_max
    mod.scatter_min  = scatter_min
    sys.modules['torch_scatter'] = mod
    print("  [patch] torch_scatter replaced with native PyTorch implementation.")


def _patch_pytorch3d():
    """
    pytorch3d fails to compile for PyTorch 2.6 + Blackwell GPUs.
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

    # Patch torch_scatter and pytorch3d before any AMB3R imports
    _patch_torch_scatter()
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

    print(f"  Running inference on {views_all['images'].shape[1]} frames...")

    # --- Forward pass ---
    with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                        dtype=torch.bfloat16):
        with torch.no_grad():
            res = model(views_all)

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

    # Clip colors to [0, 1]
    colors = np.clip(colors, 0.0, 1.0)

    # Collect image paths in the order they were loaded
    image_paths = sorted(frames_dir.glob("*.jpg"))[:max_images]

    return {
        "pts":         pts.astype(np.float32),
        "colors":      colors.astype(np.float32),
        "conf_sig":    conf_sig.astype(np.float32),
        "poses":       poses.astype(np.float32),
        "intrinsics":  intrinsics.astype(np.float32),  # (T, 3, 3)
        "pts_shape":   pts_shape,                       # (T, H, W)
        "image_paths": image_paths,
    }