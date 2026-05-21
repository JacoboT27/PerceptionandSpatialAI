"""
video-to-3d: run.py
-------------------
Single entry point for the video → 3D reconstruction pipeline.

Usage:
    python run.py --video inputs/myvideo.mp4
    python run.py --video inputs/myvideo.mp4 --no-semantic
    python run.py --video inputs/myvideo.mp4 --fps 3 --conf 0.6

Pipeline stages:
    1. Extract frames from video (ffmpeg)
    2. AMB3R feed-forward reconstruction → point cloud + camera poses
    3. Export .ply + transforms.json
    4. Semantic labelling (RAM + SAM + CLIP) — on by default, disable with --no-semantic
    5. Interactive viewer (Open3D)
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline.extract_frames import extract_frames
from pipeline.reconstruct import run_amb3r
from pipeline.export import export_results
from pipeline.visualize import view_pointcloud, view_semantic_pointcloud


def parse_args():
    parser = argparse.ArgumentParser(
        description="video-to-3d: reconstruct a 3D scene from a phone video using AMB3R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=str, required=True,
        help="Path to input video file (e.g. inputs/myvideo.mp4)")
    parser.add_argument("--output", type=str, default=None,
        help="Output directory. Defaults to outputs/<video_name>/")
    parser.add_argument("--fps", type=float, default=2.0,
        help="Frame extraction rate (fps). Recommended: 1–5.")
    parser.add_argument("--max_frames", type=int, default=150,
        help="Maximum number of frames to use.")
    parser.add_argument("--conf", type=float, default=0.5,
        help="Confidence threshold for point filtering (0.0–1.0).")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/amb3r.pt",
        help="Path to AMB3R checkpoint.")
    parser.add_argument("--ram_checkpoint", type=str, default="./checkpoints/ram_plus.pth",
        help="Path to RAM++ checkpoint.")
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam_vit_b.pth",
        help="Path to SAM ViT-B checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to run inference on.")
    parser.add_argument("--no-semantic", action="store_true", default=False,
        help="Skip semantic labelling stage.")
    parser.add_argument("--sam-only", action="store_true", default=False,
        help="Debug: run SAM segmentation only (no labels). Useful for checking mask quality.")
    parser.add_argument("--no-viewer", action="store_true", default=False,
        help="Skip the interactive 3D viewer (headless/server use).")
    return parser.parse_args()


def banner(text: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def free_gpu():
    """Release GPU memory between stages."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("outputs") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    semantic_enabled = not args.no_semantic and not args.sam_only
    total_stages = 5 if semantic_enabled else 4

    print(f"\n  video-to-3d pipeline")
    print(f"  Input     : {video_path}")
    print(f"  Output    : {output_dir}")
    print(f"  Device    : {args.device}")
    print(f"  Semantic  : {'enabled' if semantic_enabled else 'disabled (--no-semantic)'}")

    # ------------------------------------------------------------------ #
    # Stage 1: Frame Extraction
    # ------------------------------------------------------------------ #
    banner(f"Stage 1 / {total_stages} — Frame Extraction")
    t0 = time.time()

    n_frames = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"  Extracted {n_frames} frames  ({time.time() - t0:.1f}s)")

    if n_frames == 0:
        print("[ERROR] No frames extracted. Check your video path and ffmpeg installation.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Stage 2: AMB3R Reconstruction
    # ------------------------------------------------------------------ #
    banner(f"Stage 2 / {total_stages} — AMB3R Reconstruction")
    t1 = time.time()

    reconstruction = run_amb3r(
        frames_dir=frames_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.conf,
        max_images=args.max_frames,
    )

    print(f"  Reconstruction complete  ({time.time() - t1:.1f}s)")
    print(f"  Points (before filtering) : {reconstruction['pts'].shape[0]:,}")
    mask = reconstruction['conf_sig'] > args.conf
    print(f"  Points (after  filtering) : {mask.sum():,}  (conf > {args.conf})")

    # Free AMB3R from GPU before next stage
    free_gpu()

    # ------------------------------------------------------------------ #
    # Stage 3: Export
    # ------------------------------------------------------------------ #
    banner(f"Stage 3 / {total_stages} — Exporting Results")
    t2 = time.time()

    exported = export_results(
        reconstruction=reconstruction,
        output_dir=output_dir,
        conf_thresh=args.conf,
        frames_dir=frames_dir,
    )
    print(f"  Export complete  ({time.time() - t2:.1f}s)")

    # ------------------------------------------------------------------ #
    # Stage 4 (optional): SAM debug — masks only, no labels
    # ------------------------------------------------------------------ #
    if args.sam_only:
        banner("Stage 4 — SAM Debug (masks only)")
        from pipeline.semantic import run_sam_debug
        sam_ckpt = Path(args.sam_checkpoint)
        if not sam_ckpt.exists():
            print(f"  [WARNING] SAM checkpoint not found: {sam_ckpt}")
        else:
            t3 = time.time()
            debug_ply = run_sam_debug(
                reconstruction=reconstruction,
                output_dir=output_dir,
                conf_thresh=args.conf,
                sam_checkpoint=sam_ckpt,
                device=args.device,
            )
            free_gpu()
            print(f"  SAM debug complete  ({time.time() - t3:.1f}s)")
            if not args.no_viewer:
                banner("Viewer — SAM Debug")
                view_pointcloud(debug_ply)
        return

    # ------------------------------------------------------------------ #
    # Stage 4: Semantic Labelling
    # ------------------------------------------------------------------ #
    semantic_result = None
    if semantic_enabled:
        banner(f"Stage 4 / {total_stages} — Semantic Labelling")
        t3 = time.time()

        ram_ckpt = Path(args.ram_checkpoint)
        sam_ckpt = Path(args.sam_checkpoint)

        if not ram_ckpt.exists():
            print(f"  [WARNING] RAM checkpoint not found: {ram_ckpt}")
            print("  Run: bash checkpoints/download_weights.sh")
            print("  Skipping semantic stage.")
        elif not sam_ckpt.exists():
            print(f"  [WARNING] SAM checkpoint not found: {sam_ckpt}")
            print("  Run: bash checkpoints/download_weights.sh")
            print("  Skipping semantic stage.")
        else:
            from pipeline.semantic import run_semantic
            semantic_result = run_semantic(
                reconstruction=reconstruction,
                output_dir=output_dir,
                conf_thresh=args.conf,
                ram_checkpoint=ram_ckpt,
                sam_checkpoint=sam_ckpt,
                device=args.device,
            )
            print(f"  Semantic labelling complete  ({time.time() - t3:.1f}s)")
            free_gpu()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    banner("Done")
    total = time.time() - t0
    print(f"  Total time : {total:.1f}s\n")
    print(f"  Outputs saved to: {output_dir}/\n")
    for label, path in exported.items():
        print(f"    [{label}]  {path}")
    if semantic_result:
        print(f"    [semantic .ply]  {semantic_result['ply_path']}")
        print(f"    [labels   .json] {semantic_result['labels_json']}")

    # ------------------------------------------------------------------ #
    # Stage 5: Viewer
    # ------------------------------------------------------------------ #
    if not args.no_viewer:
        banner(f"Stage {total_stages} / {total_stages} — Launching Viewer")
        if semantic_result:
            view_semantic_pointcloud(
                ply_path=semantic_result['ply_path'],
                label_info=semantic_result['label_info'],
            )
        else:
            view_pointcloud(output_dir / "scene.ply")
    else:
        print("\n  Viewer skipped (--no-viewer).\n")


if __name__ == "__main__":
    main()