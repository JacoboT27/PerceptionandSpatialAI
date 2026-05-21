"""
pipeline/semantic.py
--------------------
Semantic labelling of the 3D point cloud.

One pretrained semantic-segmentation model labels every pixel of every
AMB3R frame with a class from a fixed taxonomy. Because AMB3R's world_points
is a dense per-pixel pointmap — point i has a known source frame and pixel —
the 2D label maps transfer to 3D with a single reshape. No projection, no
masks, no occlusion test, no voting.

Pipeline:
    frame   -> Mask2Former (COCO-panoptic) -> per-pixel class id
    stack   -> (T, H, W) label maps
    reshape -> per-point class id (already aligned with pts)

Outputs:
    - scene_semantic.ply : point cloud coloured by semantic class
    - labels.json        : class name -> colour + point-count summary

Requires run_amb3r() to return in `reconstruction`:
    pts, colors, conf_sig, pts_shape
with pts and colors the FULL (T*H*W, 3) arrays and pts_shape = (T, H, W).
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


# Segmentation model. COCO-panoptic (133 classes) covers common objects and
# stuff — e.g. "sports ball", "bed", "blanket", "floor", "wall". For purely
# indoor/architectural scenes an ADE20K model (150 classes), e.g.
# "facebook/mask2former-swin-base-ade-semantic", is an alternative.
# NOTE: if you change this, change it in the Dockerfile pre-download too.
MODEL_ID = "facebook/mask2former-swin-base-coco-panoptic"

# One colour per class actually present in the scene.
PALETTE = [
    [231, 76,  60 ], [52,  152, 219], [46,  204, 113], [241, 196, 15 ],
    [155, 89,  182], [230, 126, 34 ], [26,  188, 156], [127, 140, 141],
    [44,  62,  80 ], [243, 156, 18 ], [211, 84,  0  ], [39,  174, 96 ],
    [142, 68,  173], [22,  160, 133], [192, 57,  43 ], [41,  128, 185],
]


def _frames_from_reconstruction(reconstruction: Dict):
    """Recover AMB3R's processed frames as a list of T RGB uint8 (H,W,3) arrays."""
    T, H, W = reconstruction["pts_shape"]
    colors = np.asarray(reconstruction["colors"]).reshape(T, H, W, 3)
    return [(np.clip(colors[f], 0.0, 1.0) * 255.0).astype(np.uint8) for f in range(T)]


def _segment_frames(frames, model_id: str, device: str) -> Tuple[np.ndarray, Dict]:
    """
    Run the semantic segmenter on every frame.

    Returns:
        label_maps : (T, H, W) int32 array of class ids
        id2label   : dict {class id -> class name}
    """
    import torch
    import gc
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    print(f"  Loading segmentation model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    model = model.to(device).eval()

    H, W = frames[0].shape[:2]
    label_maps = np.zeros((len(frames), H, W), dtype=np.int32)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    for i, frame in enumerate(frames):
        inputs = processor(images=Image.fromarray(frame), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # post_process_semantic_segmentation -> a per-pixel class-id map at (H,W)
        seg = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(H, W)]
        )[0]
        label_maps[i] = seg.cpu().numpy().astype(np.int32)
        present = [id2label.get(int(c), str(c)) for c in np.unique(label_maps[i])]
        print(f"    frame {i}: {', '.join(present)}")

    del model
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return label_maps, id2label


def _write_semantic_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write a binary coloured point cloud."""
    n = points.shape[0]
    colors_u8 = (colors * 255).clip(0, 255).astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8), ("g", np.uint8), ("b", np.uint8),
    ])
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["r"], data["g"], data["b"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def run_semantic(
    reconstruction: Dict,
    output_dir: Path,
    conf_thresh: float,
    device: str = "cuda",
    model_id: str = MODEL_ID,
) -> Dict:
    """
    Semantically label the reconstruction.

    Returns a dict with: ply_path, labels_json, label_info, labels.
    """
    pts      = reconstruction["pts"]
    conf_sig = reconstruction["conf_sig"]
    T, H, W  = reconstruction["pts_shape"]

    # --- Stage 1: segment every frame ---
    frames = _frames_from_reconstruction(reconstruction)
    print(f"  Semantic segmentation over {len(frames)} frames "
          f"({W}x{H}, vocabulary from {model_id})...")
    label_maps, id2label = _segment_frames(frames, model_id, device)

    # --- Stage 2: transfer to 3D ---
    # AMB3R's pts is ordered frame-major / row-major / col-major, and so is
    # label_maps. Flattening label_maps therefore gives one class id per
    # point, already aligned with pts — a direct reshape, no projection.
    labels_id_all = label_maps.reshape(-1)
    if labels_id_all.shape[0] != pts.shape[0]:
        raise ValueError(
            f"Point/label count mismatch: {pts.shape[0]} points vs "
            f"{labels_id_all.shape[0]} labels. pts must be the full "
            f"(T*H*W, 3) array and pts_shape must be (T, H, W)."
        )

    conf_mask = conf_sig > conf_thresh
    labels_id = labels_id_all[conf_mask]
    labels    = np.array([id2label.get(int(c), str(c)) for c in labels_id], dtype=object)

    # --- Stage 3: colour ---
    unique_ids = sorted(int(c) for c in np.unique(labels_id))
    id_color   = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(unique_ids)}
    colors_semantic = np.array(
        [[c / 255.0 for c in id_color[int(c)]] for c in labels_id], dtype=np.float32
    )

    # Blend semantic colour with the original RGB — keeps texture readable.
    alpha = 0.55
    original_colors = np.asarray(reconstruction["colors"])[conf_mask]
    colors_blended = np.clip(
        alpha * colors_semantic + (1.0 - alpha) * original_colors, 0.0, 1.0
    )

    # --- Stage 4: write outputs ---
    output_dir = Path(output_dir)
    ply_path = output_dir / "scene_semantic.ply"
    print(f"  Writing semantic point cloud -> {ply_path}")
    _write_semantic_ply(ply_path, pts[conf_mask], colors_blended)

    label_info = {}
    for cid in unique_ids:
        name  = id2label.get(cid, str(cid))
        count = int((labels_id == cid).sum())
        label_info[name] = {
            "color_rgb": id_color[cid],
            "point_count": count,
            "percentage": round(100 * count / max(len(labels_id), 1), 1),
        }

    json_path = output_dir / "labels.json"
    with open(json_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"  Writing label summary -> {json_path}")

    return {
        "ply_path":    ply_path,
        "labels_json": json_path,
        "label_info":  label_info,
        "labels":      labels,
    }