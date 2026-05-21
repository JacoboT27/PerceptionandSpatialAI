"""
pipeline/visualize.py
---------------------
Interactive 3D point cloud viewer using Open3D.
Supports both standard (RGB) and semantic (label-coloured) point clouds.

Controls:
    Left mouse    : rotate
    Right mouse   : pan
    Scroll        : zoom
    Q / Escape    : quit
"""

from pathlib import Path
from typing import Dict, Optional


def view_pointcloud(ply_path: Path):
    """Open an interactive Open3D viewer for a standard RGB .ply point cloud."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [viewer] open3d not found — skipping viewer.")
        print(f"  Open {ply_path} manually in MeshLab or CloudCompare.")
        return

    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"  [viewer] File not found: {ply_path}")
        return

    print(f"  Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n   = len(pcd.points)
    print(f"  Points: {n:,}")

    if n == 0:
        print("  [viewer] Point cloud is empty.")
        return

    _launch_viewer(pcd, title="video-to-3d — Point Cloud")


def view_semantic_pointcloud(
    ply_path: Path,
    label_info: Optional[Dict] = None,
):
    """
    Open an interactive Open3D viewer for a semantically labelled point cloud.
    Prints a colour-coded legend to the terminal alongside the viewer.

    Args:
        ply_path   : Path to scene_semantic.ply
        label_info : Dict from semantic.run_semantic() with label → color/count info
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  [viewer] open3d not found — skipping viewer.")
        return

    ply_path = Path(ply_path)
    if not ply_path.exists():
        print(f"  [viewer] File not found: {ply_path}")
        return

    print(f"  Loading semantic point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    n   = len(pcd.points)
    print(f"  Points: {n:,}")

    if n == 0:
        print("  [viewer] Point cloud is empty.")
        return

    # Print legend to terminal
    if label_info:
        _print_legend(label_info)

    _launch_viewer(pcd, title="video-to-3d — Semantic Labels")


def _launch_viewer(pcd, title: str = "video-to-3d"):
    """Shared viewer launcher."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name=title, width=1280, height=720)

    if not created:
        print()
        print("  [viewer] Could not open display window. To fix this, run on your host:")
        print("           xhost +local:docker")
        print("  Then rerun with:")
        print("           docker compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix app python run.py ...")
        print()
        print("  Your point cloud was saved and can be opened manually in MeshLab or CloudCompare.")
        return

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]
    opt.point_size = 2.0

    vis.reset_view_point(True)

    print()
    print("  ┌─────────────────────────────────┐")
    print("  │   3D Viewer Controls             │")
    print("  │   Left drag   : rotate           │")
    print("  │   Right drag  : pan              │")
    print("  │   Scroll      : zoom             │")
    print("  │   Q / Escape  : close            │")
    print("  └─────────────────────────────────┘")
    print()

    vis.run()
    vis.destroy_window()


def _print_legend(label_info: Dict, min_pct: float = 5.0):
    """Print dominant labels only (above min_pct threshold)."""
    dominant = {k: v for k, v in label_info.items() if v['percentage'] >= min_pct}
    if not dominant:
        dominant = label_info  # fallback: show all if none pass threshold

    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │  Semantic Legend  (labels ≥ 5% of points)       │")
    print("  ├─────────────────────────────────────────────────┤")
    for label, info in sorted(dominant.items(), key=lambda x: -x[1]['point_count']):
        r, g, b  = info['color_rgb']
        count    = info['point_count']
        pct      = info['percentage']
        color_block = f"\033[48;2;{r};{g};{b}m   \033[0m"
        print(f"  │  {color_block}  {label:<20} {count:>8,} pts  ({pct:>5.1f}%)  │")
    print("  └─────────────────────────────────────────────────┘")
    print()