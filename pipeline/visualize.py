"""
pipeline/visualize.py
---------------------
Interactive 3D point cloud viewer using Open3D.
Opens automatically after reconstruction unless --no-viewer is passed.

Controls:
    Left mouse    : rotate
    Right mouse   : pan
    Scroll        : zoom
    Q / Escape    : quit
"""

from pathlib import Path


def view_pointcloud(ply_path: Path):
    """
    Open an interactive Open3D viewer for a .ply point cloud.

    Args:
        ply_path : Path to the .ply file to visualise.
    """
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

    print(f"  Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))

    n_points = len(pcd.points)
    print(f"  Points: {n_points:,}")

    if n_points == 0:
        print("  [viewer] Point cloud is empty — nothing to show.")
        return

    # --- Viewer settings ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="video-to-3d — Point Cloud Viewer",
        width=1280,
        height=720,
    )

    vis.add_geometry(pcd)

    # Render options
    opt = vis.get_render_option()
    opt.background_color = [0.1, 0.1, 0.1]   # dark background
    opt.point_size = 2.0                       # slightly larger points

    # Centre the view on the point cloud
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