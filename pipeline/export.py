"""
pipeline/export.py
------------------
Exports the AMB3R reconstruction to:
    1. scene.ply         — coloured point cloud (open in MeshLab / CloudCompare)
    2. transforms.json   — Nerfstudio-compatible camera poses (for future 3DGS)
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# --------------------------------------------------------------------------- #
# PLY export (no open3d dependency — pure numpy for portability)
# --------------------------------------------------------------------------- #

def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """
    Write a coloured point cloud to a binary-little-endian PLY file.

    Args:
        path   : Output .ply file path.
        points : (N, 3) float32 XYZ.
        colors : (N, 3) float32 RGB in [0, 1].
    """
    n = points.shape[0]
    colors_uint8 = (colors * 255).clip(0, 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # Interleave xyz + rgb into a structured array for efficient binary write
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
    ])
    data = np.empty(n, dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["r"] = colors_uint8[:, 0]
    data["g"] = colors_uint8[:, 1]
    data["b"] = colors_uint8[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# --------------------------------------------------------------------------- #
# Nerfstudio transforms.json export
# --------------------------------------------------------------------------- #

def _write_transforms_json(
    path: Path,
    poses: np.ndarray,
    image_paths: List[Path],
    frames_dir: Path,
    output_dir: Path,
):
    """
    Write a Nerfstudio-compatible transforms.json file.

    This lets you pipe the AMB3R camera poses directly into:
        ns-train splatfacto --data <output_dir>

    Args:
        path        : Output transforms.json path.
        poses       : (T, 4, 4) camera-to-world matrices.
        image_paths : List of frame file paths (same order as poses).
        frames_dir  : Directory containing frames (for relative path computation).
        output_dir  : Root output directory (transforms.json will reference frames relative to here).
    """
    frames = []
    for i, (pose, img_path) in enumerate(zip(poses, image_paths)):
        # Nerfstudio expects the image path relative to the transforms.json location
        try:
            rel_path = img_path.relative_to(output_dir)
        except ValueError:
            # If frames are outside output_dir, use absolute path as fallback
            rel_path = img_path

        frames.append({
            "file_path": str(rel_path),
            "transform_matrix": pose.tolist(),
        })

    data = {
        # Placeholder intrinsics — AMB3R doesn't expose calibrated focal length
        # in its base demo API. Replace with actual values if known.
        # For splatfacto, these will be optimised during training anyway.
        "camera_model": "OPENCV",
        "fl_x": 500.0,
        "fl_y": 500.0,
        "cx": 259.0,
        "cy": 196.0,
        "w": 518,
        "h": 392,
        "frames": frames,
        "_note": (
            "Intrinsics (fl_x, fl_y, cx, cy) are approximate placeholders. "
            "Nerfstudio's splatfacto will refine them during optimisation. "
            "For best results, provide calibrated values if available."
        ),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# Main export function
# --------------------------------------------------------------------------- #

def export_results(
    reconstruction: Dict,
    output_dir: Path,
    conf_thresh: float = 0.5,
    frames_dir: Path = None,
) -> Dict[str, Path]:
    """
    Export AMB3R reconstruction results.

    Args:
        reconstruction : Output dict from pipeline.reconstruct.run_amb3r().
        output_dir     : Directory to write outputs into.
        conf_thresh    : Points with conf_sig < conf_thresh are excluded from the .ply.
        frames_dir     : Directory containing extracted frames (for transforms.json paths).

    Returns:
        Dict mapping output label → file path.
    """
    pts         = reconstruction["pts"]
    colors      = reconstruction["colors"]
    conf_sig    = reconstruction["conf_sig"]
    poses       = reconstruction["poses"]
    image_paths = reconstruction["image_paths"]

    output_dir = Path(output_dir)

    exported = {}

    # --- 1. Filtered point cloud → scene.ply ---
    mask = conf_sig > conf_thresh
    pts_filtered    = pts[mask]
    colors_filtered = colors[mask]

    ply_path = output_dir / "scene.ply"
    print(f"  Writing point cloud → {ply_path}  ({pts_filtered.shape[0]:,} points)")
    _write_ply(ply_path, pts_filtered, colors_filtered)
    exported["point_cloud (.ply)"] = ply_path

    # Unfiltered version — useful for re-thresholding without re-running the model
    ply_path_full = output_dir / "scene_unfiltered.ply"
    print(f"  Writing unfiltered point cloud → {ply_path_full}  ({pts.shape[0]:,} points)")
    _write_ply(ply_path_full, pts, colors)
    exported["point_cloud unfiltered (.ply)"] = ply_path_full

    # --- 2. Camera poses → transforms.json ---
    transforms_path = output_dir / "transforms.json"
    print(f"  Writing camera poses → {transforms_path}  ({len(poses)} frames)")
    _write_transforms_json(
        path=transforms_path,
        poses=poses,
        image_paths=image_paths,
        frames_dir=frames_dir or output_dir / "frames",
        output_dir=output_dir,
    )
    exported["camera poses (.json)"] = transforms_path

    return exported