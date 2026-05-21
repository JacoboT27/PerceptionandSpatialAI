"""
pipeline/semantic.py
--------------------
Automatic semantic labelling of the 3D point cloud.

Pipeline:
    1. RAM (Recognize Anything Model) — scans each frame, returns a list of
       tags (e.g. "basketball", "floor", "person") with no prompts needed.
    2. SAM (Segment Anything Model, automatic mode) — generates instance masks
       for every region in each frame, also with no prompts.
    3. CLIP — assigns the best RAM tag to each SAM mask by comparing the
       cropped region to the tag list.
    4. Projection — each 3D point is projected into the camera frames using
       the AMB3R poses + intrinsics; the label of the covering mask is assigned.

Outputs:
    - scene_semantic.ply   : point cloud coloured by semantic label
    - labels.json          : label name → colour + point count summary
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Colour palette — one colour per unique label (BGR for OpenCV, RGB for PLY)
# --------------------------------------------------------------------------- #

PALETTE = [
    [231, 76,  60 ],  # red
    [52,  152, 219],  # blue
    [46,  204, 113],  # green
    [241, 196, 15 ],  # yellow
    [155, 89,  182],  # purple
    [230, 126, 34 ],  # orange
    [26,  188, 156],  # teal
    [236, 240, 241],  # white
    [127, 140, 141],  # grey
    [44,  62,  80 ],  # dark blue
    [243, 156, 18 ],  # amber
    [211, 84,  0  ],  # dark orange
    [39,  174, 96 ],  # dark green
    [142, 68,  173],  # dark purple
    [22,  160, 133],  # dark teal
]


def _label_to_color(label: str, label_list: List[str]) -> List[int]:
    idx = label_list.index(label) % len(PALETTE)
    return PALETTE[idx]


# --------------------------------------------------------------------------- #
# Stage 1: RAM — automatic image tagging
# --------------------------------------------------------------------------- #

def _run_ram(image_paths: List[Path], checkpoint: Path, device: str) -> List[List[str]]:
    """
    Run RAM++ on each frame and return a list of tag lists.
    Each element is the list of object tags found in that frame.
    """
    import torch
    from PIL import Image
    from ram.models import ram_plus
    from ram import inference_ram as inference
    from ram import get_transform

    print(f"  Loading RAM++ from: {checkpoint}")
    transform = get_transform(image_size=384)
    model = ram_plus(pretrained=str(checkpoint), image_size=384, vit='swin_l')
    model.eval()
    model = model.to(device)

    all_tags = []
    for img_path in image_paths:
        image = transform(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            result = inference(image, model)
        # result is a tuple; first element is the tag string e.g. "basketball | floor | person"
        tags = [t.strip() for t in result[0].split('|') if t.strip()]
        all_tags.append(tags)
        print(f"    {img_path.name}: {', '.join(tags)}")

    return all_tags


# --------------------------------------------------------------------------- #
# Stage 2: SAM — automatic mask generation
# --------------------------------------------------------------------------- #

def _run_sam(image_paths: List[Path], checkpoint: Path, device: str):
    """
    Run SAM automatic mask generation on each frame.
    Returns a list of mask lists (one per frame).
    Each mask is a dict with 'segmentation' (H, W bool array).
    """
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    print(f"  Loading SAM from: {checkpoint}")
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam = sam.to(device)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,        # lower = faster, fewer masks
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,  # ignore tiny masks
    )

    all_masks = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        masks = generator.generate(img_rgb)
        all_masks.append(masks)
        print(f"    {img_path.name}: {len(masks)} masks")

    return all_masks


# --------------------------------------------------------------------------- #
# Stage 3: CLIP — assign RAM tags to SAM masks
# --------------------------------------------------------------------------- #

def _assign_labels_clip(
    image_paths: List[Path],
    all_masks,
    all_tags: List[List[str]],
    device: str,
) -> List[List[Tuple]]:
    """
    For each SAM mask, use CLIP to pick the best RAM tag.
    Returns a list (per frame) of (mask_array, label) tuples.
    """
    import torch
    import clip
    from PIL import Image

    print("  Loading CLIP for label assignment...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    labeled_masks = []
    for img_path, masks, tags in zip(image_paths, all_masks, all_tags):
        if not tags or not masks:
            labeled_masks.append([])
            continue

        img_pil = Image.open(img_path).convert("RGB")
        img_np  = np.array(img_pil)

        # Encode all tags as text
        text_tokens = clip.tokenize(tags).to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        frame_labeled = []
        for mask_info in masks:
            seg = mask_info['segmentation']   # (H, W) bool
            bbox = mask_info['bbox']          # x, y, w, h

            # Crop the masked region
            x, y, w, h = [int(v) for v in bbox]
            x2, y2 = min(x + w, img_np.shape[1]), min(y + h, img_np.shape[0])
            if x2 <= x or y2 <= y:
                continue

            crop = img_pil.crop((x, y, x2, y2))
            img_input = clip_preprocess(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feats = clip_model.encode_image(img_input)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sims = (img_feats @ text_feats.T).squeeze(0)
                best_idx = sims.argmax().item()

            label = tags[best_idx]
            frame_labeled.append((seg, label))

        labeled_masks.append(frame_labeled)

    return labeled_masks


# --------------------------------------------------------------------------- #
# Stage 4: Project 3D points → 2D pixels → labels
# --------------------------------------------------------------------------- #

def _project_labels(
    pts: np.ndarray,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    pts_shape: Tuple,
    labeled_masks: List[List[Tuple]],
    conf_mask: np.ndarray,
) -> np.ndarray:
    """
    For each 3D point, project into each camera frame and assign the label
    of the SAM mask that covers the projected pixel.

    Args:
        pts          : (N, 3) filtered world-space points
        poses        : (T, 4, 4) camera-to-world transforms
        intrinsics   : (T, 3, 3) camera intrinsic matrices
        pts_shape    : (T, H, W) original point layout from AMB3R
        labeled_masks: per-frame list of (mask, label) pairs
        conf_mask    : boolean array of shape (T*H*W,) used for filtering

    Returns:
        labels : (N,) array of label strings, one per filtered point
    """
    T, H, W = pts_shape
    N = pts.shape[0]
    labels = np.full(N, "unknown", dtype=object)

    # Build a label map per frame: (H, W) → label string
    label_maps = []
    for frame_labeled in labeled_masks:
        lmap = np.full((H, W), "unknown", dtype=object)
        for seg, label in frame_labeled:
            # seg might be a different resolution — resize if needed
            if seg.shape != (H, W):
                seg_resized = cv2.resize(
                    seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                seg_resized = seg
            lmap[seg_resized] = label
        label_maps.append(lmap)

    # For each point, find its source frame and pixel using the pts_shape layout
    # Points are laid out as: point i → frame i//(H*W), row (i%(H*W))//W, col i%(H*W)%W
    # We need to map filtered indices back to original indices
    original_indices = np.where(conf_mask)[0]

    for out_idx, orig_idx in enumerate(original_indices):
        frame_idx = orig_idx // (H * W)
        remainder = orig_idx % (H * W)
        row = remainder // W
        col = remainder % W

        if frame_idx < len(label_maps):
            labels[out_idx] = label_maps[frame_idx][row, col]

    return labels


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def _write_semantic_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write a coloured point cloud (same as export.py but semantic colours)."""
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


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_semantic(
    reconstruction: Dict,
    output_dir: Path,
    conf_thresh: float,
    ram_checkpoint: Path,
    sam_checkpoint: Path,
    device: str = "cuda",
) -> Dict:
    """
    Run the full semantic labelling pipeline.

    Args:
        reconstruction  : Output dict from run_amb3r().
        output_dir      : Directory to write outputs into.
        conf_thresh     : Same confidence threshold used for point filtering.
        ram_checkpoint  : Path to RAM++ weights.
        sam_checkpoint  : Path to SAM ViT-B weights.
        device          : 'cuda' or 'cpu'.

    Returns:
        Dict with keys: ply_path, labels_json_path, label_info
    """
    import torch

    pts         = reconstruction["pts"]
    conf_sig    = reconstruction["conf_sig"]
    poses       = reconstruction["poses"]
    intrinsics  = reconstruction["intrinsics"]
    pts_shape   = reconstruction["pts_shape"]
    image_paths = reconstruction["image_paths"]

    conf_mask = conf_sig > conf_thresh
    pts_filtered = pts[conf_mask]

    print(f"  Labelling {pts_filtered.shape[0]:,} points across {len(image_paths)} frames...")

    # --- Stage 1: RAM ---
    print("\n  [1/3] Running RAM++ (automatic image tagging)...")
    all_tags = _run_ram(image_paths, ram_checkpoint, device)

    # Collect all unique tags across all frames
    all_unique_tags = sorted(set(tag for tags in all_tags for tag in tags))
    print(f"  Tags found: {', '.join(all_unique_tags)}")

    # --- Stage 2: SAM ---
    print("\n  [2/3] Running SAM (automatic mask generation)...")
    all_masks = _run_sam(image_paths, sam_checkpoint, device)

    # --- Stage 3: CLIP label assignment ---
    print("\n  [3/3] Assigning labels via CLIP...")
    labeled_masks = _assign_labels_clip(image_paths, all_masks, all_tags, device)

    # --- Stage 4: Project into 3D ---
    print("\n  Projecting labels into 3D...")
    labels = _project_labels(
        pts_filtered, poses, intrinsics, pts_shape, labeled_masks, conf_mask
    )

    # --- Build colour array from labels ---
    unique_labels = sorted(set(labels.tolist()))
    label_colors  = {}
    for i, lbl in enumerate(unique_labels):
        label_colors[lbl] = [c / 255.0 for c in PALETTE[i % len(PALETTE)]]

    colors_semantic = np.array([label_colors[lbl] for lbl in labels], dtype=np.float32)

    # --- Write semantic .ply ---
    output_dir = Path(output_dir)
    ply_path = output_dir / "scene_semantic.ply"
    print(f"  Writing semantic point cloud → {ply_path}")
    _write_semantic_ply(ply_path, pts_filtered, colors_semantic)

    # --- Write labels.json summary ---
    label_info = {}
    for lbl in unique_labels:
        count = int((labels == lbl).sum())
        label_info[lbl] = {
            "color_rgb": [int(c * 255) for c in label_colors[lbl]],
            "point_count": count,
            "percentage": round(100 * count / len(labels), 1),
        }

    json_path = output_dir / "labels.json"
    with open(json_path, "w") as f:
        json.dump(label_info, f, indent=2)
    print(f"  Writing label summary → {json_path}")

    return {
        "ply_path":        ply_path,
        "labels_json":     json_path,
        "label_info":      label_info,
        "labels":          labels,
        "pts_filtered":    pts_filtered,
        "colors_semantic": colors_semantic,
        "label_colors":    label_colors,
    }