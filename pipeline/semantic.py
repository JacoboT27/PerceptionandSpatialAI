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

def _run_sam(image_paths: List[Path], checkpoint: Path, device: str,
             target_size: Tuple = (518, 392)):
    """
    Run SAM automatic mask generation on each frame.
    Images are resized to target_size (W, H) to match AMB3R's processing
    resolution — this ensures pixel-perfect alignment during 3D projection.

    Returns a list of mask lists (one per frame).
    Each mask has 'segmentation' (H, W bool) at target_size resolution.
    """
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    target_w, target_h = target_size   # (518, 392)

    print(f"  Loading SAM from: {checkpoint}")
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    sam = sam.to(device)

    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,
    )

    all_masks = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))

        # Resize to AMB3R's exact processing resolution so masks align with
        # the world_points pixel grid — same crop/resize AMB3R applies
        img_bgr = cv2.resize(img_bgr, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        masks = generator.generate(img_rgb)
        all_masks.append(masks)
        print(f"    {img_path.name}: {len(masks)} masks  (at {target_w}×{target_h})")

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
            area  = int(mask_info.get('area', seg.sum()))
            frame_labeled.append((seg, label, area))

        # Sort largest → smallest so small objects are applied last and not overwritten
        frame_labeled.sort(key=lambda x: x[2], reverse=True)
        labeled_masks.append([(seg, label) for seg, label, _ in frame_labeled])

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
) -> np.ndarray:
    """
    Project each 3D point into all camera frames, look up the SAM/CLIP label
    at each projected pixel, then assign the most common non-unknown label
    (majority vote). Works for any scene — no hardcoded label names.
    """
    T, H, W = pts_shape
    N = pts.shape[0]

    # Build label maps (H, W) per frame — already at AMB3R resolution
    label_maps = []
    for frame_labeled in labeled_masks:
        lmap = np.full((H, W), "unknown", dtype=object)
        for seg, label in frame_labeled:
            if seg.shape != (H, W):
                seg = cv2.resize(
                    seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            lmap[seg] = label
        label_maps.append(lmap)

    pts_h  = np.concatenate([pts, np.ones((N, 1), dtype=np.float32)], axis=1)
    votes  = [[] for _ in range(N)]

    for frame_idx in range(min(T, len(label_maps))):
        lmap     = label_maps[frame_idx]
        pose_c2w = poses[frame_idx]
        K        = intrinsics[frame_idx].copy()

        # AMB3R stores fx/fy normalised (<1.0); cx/cy are already in pixels
        if K[0, 0] < 2.0:
            K[0, 0] *= W
            K[1, 1] *= H

        pose_w2c = np.linalg.inv(pose_c2w)
        pts_cam  = (pose_w2c @ pts_h.T).T[:, :3]

        valid    = pts_cam[:, 2] > 0.01
        if not valid.any():
            continue

        u = (K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]).astype(np.int32)

        in_frame = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        for i in np.where(in_frame)[0]:
            votes[i].append(lmap[v[i], u[i]])

    # Majority vote: most common non-unknown label wins
    labels = np.full(N, "unknown", dtype=object)
    for i, vote_list in enumerate(votes):
        if not vote_list:
            continue
        non_unknown = [l for l in vote_list if l != "unknown"]
        if non_unknown:
            labels[i] = max(set(non_unknown), key=non_unknown.count)

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

def run_sam_debug(
    reconstruction: Dict,
    output_dir: Path,
    conf_thresh: float,
    sam_checkpoint: Path,
    device: str = "cuda",
) -> Path:
    """
    Debug mode: run SAM only, assign a random colour per mask, project into 3D.
    No RAM, no CLIP, no labels. Use this to verify SAM is segmenting correctly
    before attempting label assignment.
    """
    import torch

    pts         = reconstruction["pts"]
    conf_sig    = reconstruction["conf_sig"]
    poses       = reconstruction["poses"]
    intrinsics  = reconstruction["intrinsics"]
    pts_shape   = reconstruction["pts_shape"]
    image_paths = reconstruction["image_paths"]

    conf_mask    = conf_sig > conf_thresh
    pts_filtered = pts[conf_mask]
    T, H, W      = pts_shape

    print(f"  [SAM debug] Running on {len(image_paths)} frames...")
    all_masks = _run_sam(image_paths, sam_checkpoint, device, target_size=(W, H))

    # Assign a random colour per unique mask index across all frames
    rng = np.random.default_rng(42)
    colors_out = reconstruction["colors"][conf_mask].copy()  # start from original RGB

    original_indices = np.where(conf_mask)[0]

    for frame_idx, masks in enumerate(all_masks):
        # Sort largest first so small masks paint over large ones
        masks_sorted = sorted(masks, key=lambda m: m.get('area', 0), reverse=True)

        for mask_info in masks_sorted:
            seg = mask_info['segmentation']
            if seg.shape != (H, W):
                seg = cv2.resize(
                    seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Random colour for this mask
            color = rng.random(3).astype(np.float32)

            # Find all output points that belong to this frame + mask pixel
            frame_start = frame_idx * H * W
            frame_end   = frame_start + H * W
            frame_orig  = np.arange(frame_start, frame_end)

            # Which of those are in the filtered set?
            # Map original indices back to output positions
            for out_idx, orig_idx in enumerate(original_indices):
                if frame_start <= orig_idx < frame_end:
                    remainder = orig_idx - frame_start
                    row = remainder // W
                    col = remainder % W
                    if seg[row, col]:
                        # Blend 60% mask colour + 40% original
                        colors_out[out_idx] = 0.6 * color + 0.4 * colors_out[out_idx]

    output_dir = Path(output_dir)
    ply_path = output_dir / "scene_sam_debug.ply"
    print(f"  Writing SAM debug point cloud → {ply_path}")
    _write_semantic_ply(ply_path, pts_filtered, np.clip(colors_out, 0, 1))
    return ply_path


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

    conf_mask    = conf_sig > conf_thresh
    pts_filtered = pts[conf_mask]

    # For semantic projection use ALL points (no confidence filter) so that
    # low-confidence regions (shiny/reflective objects like basketballs) still
    # get labeled. We apply conf_mask only when writing the final .ply.
    pts_all = pts

    print(f"  Labelling {pts_filtered.shape[0]:,} points across {len(image_paths)} frames...")
    print(f"  (Projection uses all {pts_all.shape[0]:,} points to capture low-confidence objects)")

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

    # --- Stage 4: Project into 3D (all points, no confidence filter) ---
    print("\n  Projecting labels into 3D...")
    labels_all = _project_labels(
        pts_all, poses, intrinsics, pts_shape, labeled_masks
    )

    # Apply confidence mask for final output
    labels = labels_all[conf_mask]

    # --- Build colour array from labels ---
    unique_labels = sorted(set(labels.tolist()))
    label_colors  = {}
    for i, lbl in enumerate(unique_labels):
        label_colors[lbl] = [c / 255.0 for c in PALETTE[i % len(PALETTE)]]

    colors_semantic = np.array([label_colors[lbl] for lbl in labels], dtype=np.float32)

    # Blend semantic colours with original RGB — preserves texture while adding labels
    alpha = 0.55   # semantic weight (0 = original only, 1 = semantic only)
    original_colors = reconstruction["colors"][conf_mask]
    colors_blended  = alpha * colors_semantic + (1.0 - alpha) * original_colors
    colors_blended  = np.clip(colors_blended, 0.0, 1.0)

    # --- Write semantic .ply ---
    output_dir = Path(output_dir)
    ply_path = output_dir / "scene_semantic.ply"
    print(f"  Writing semantic point cloud → {ply_path}")
    _write_semantic_ply(ply_path, pts_filtered, colors_blended)

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
        "colors_semantic": colors_blended,
        "label_colors":    label_colors,
    }