#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma_retargeting.src.viser_utils import create_motion_control_sliders  # noqa: E402


DEFAULT_NPZ = PACKAGE_ROOT / "demo_data" / "TML_data" / "walk_up_karen_stairs.npz"
DEFAULT_STAIRS_DIR = PACKAGE_ROOT / "demo_data" / "climb" / "up_continuous_v2_staircase_hierarchy"
DEFAULT_ROBOT_URDF = PACKAGE_ROOT / "models" / "g1" / "g1_29dof.urdf"


def load_motion(npz_path: Path) -> tuple[np.ndarray, int]:
    data = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=float)
    fps = int(float(data["fps"])) if "fps" in data else 30
    return qpos, fps


def load_default_stair_pose(stairs_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = stairs_dir / "staircase_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        position = np.asarray(metadata.get("default_root_position_m", [0.0, 0.0, 0.0]), dtype=float)
        quat = np.asarray(metadata.get("default_root_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=float)
        return position, quat
    return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def add_stair_markers(server: viser.ViserServer, stairs_dir: Path) -> None:
    metadata_path = stairs_dir / "staircase_metadata.json"
    if not metadata_path.exists():
        return

    metadata = json.loads(metadata_path.read_text())
    colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.4, 1.0)]
    for idx, stair in enumerate(metadata.get("stairs", []), start=1):
        marker = np.asarray(stair["top_front_left_marker_m"], dtype=float)
        server.scene.add_icosphere(
            f"/stairs/marker_{idx}",
            radius=0.025,
            color=colors[(idx - 1) % len(colors)],
            position=tuple(marker),
        )


def add_robot_root_trace(server: viser.ViserServer, qpos: np.ndarray) -> None:
    points = np.asarray(qpos[:, 0:3], dtype=float)
    if points.shape[0] < 2:
        return
    server.scene.add_line_segments(
        "/robot_root_trace",
        points=np.stack([points[:-1], points[1:]], axis=1),
        colors=np.tile(np.array([[0.95, 0.45, 0.1], [0.95, 0.45, 0.1]], dtype=float), (points.shape[0] - 1, 1, 1)),
        line_width=2.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a retargeted MuJoCo qpos motion with a static staircase in Viser."
    )
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="Retargeted motion npz with qpos/fps.")
    parser.add_argument(
        "--stairs-dir",
        type=Path,
        default=DEFAULT_STAIRS_DIR,
        help="Folder created by fbx_staircase_to_assets.py containing multi_boxes.urdf.",
    )
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF, help="Robot URDF for Viser.")
    parser.add_argument("--fps", type=int, default=None, help="Override playback FPS from the npz.")
    parser.add_argument("--stairs-x", type=float, default=None, help="Initial staircase X translation in meters.")
    parser.add_argument("--stairs-y", type=float, default=None, help="Initial staircase Y translation in meters.")
    parser.add_argument("--stairs-z", type=float, default=None, help="Initial staircase Z translation in meters.")
    parser.add_argument("--stairs-yaw-deg", type=float, default=0.0, help="Initial staircase yaw in degrees.")
    args = parser.parse_args()

    qpos, npz_fps = load_motion(args.npz.resolve())
    actual_fps = args.fps if args.fps is not None else npz_fps

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    robot_root = server.scene.add_frame("/robot", show_axes=False)
    stairs_root = server.scene.add_frame("/stairs", show_axes=True)

    robot_urdf = yourdfpy.URDF.load(str(args.robot_urdf.resolve()), load_meshes=True, build_scene_graph=True)
    stairs_urdf_path = (args.stairs_dir / "multi_boxes.urdf").resolve()
    stairs_urdf = yourdfpy.URDF.load(str(stairs_urdf_path), load_meshes=True, build_scene_graph=True)

    viser_robot = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
    viser_stairs = ViserUrdf(server, urdf_or_path=stairs_urdf, root_node_name="/stairs")
    viser_robot.show_visual = True
    viser_stairs.show_visual = True

    joint_limits = viser_robot.get_actuated_joint_limits()
    robot_dof = len(joint_limits)

    create_motion_control_sliders(
        server=server,
        viser_robot=viser_robot,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        viser_object=None,
        object_base_frame=None,
        contains_object_in_qpos=False,
        initial_fps=actual_fps,
        initial_interp_mult=2,
        loop=True,
    )

    stair_pos, stair_quat = load_default_stair_pose(args.stairs_dir.resolve())
    if args.stairs_x is not None:
        stair_pos[0] = args.stairs_x
    if args.stairs_y is not None:
        stair_pos[1] = args.stairs_y
    if args.stairs_z is not None:
        stair_pos[2] = args.stairs_z

    yaw_rad = np.deg2rad(args.stairs_yaw_deg)
    stair_quat = np.array([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=float)

    stairs_root.position = stair_pos
    stairs_root.wxyz = stair_quat

    add_stair_markers(server, args.stairs_dir.resolve())
    add_robot_root_trace(server, qpos)

    with server.gui.add_folder("Stairs Placement"):
        show_stairs = server.gui.add_checkbox("Show stairs", initial_value=True)
        stairs_x = server.gui.add_number("X", initial_value=float(stair_pos[0]), step=0.01)
        stairs_y = server.gui.add_number("Y", initial_value=float(stair_pos[1]), step=0.01)
        stairs_z = server.gui.add_number("Z", initial_value=float(stair_pos[2]), step=0.01)
        stairs_yaw = server.gui.add_number("Yaw (deg)", initial_value=float(args.stairs_yaw_deg), step=1.0)

    @show_stairs.on_update
    def _(_evt) -> None:
        viser_stairs.show_visual = bool(show_stairs.value)

    def _apply_stairs_pose() -> None:
        yaw = np.deg2rad(float(stairs_yaw.value))
        stairs_root.position = np.array([stairs_x.value, stairs_y.value, stairs_z.value], dtype=float)
        stairs_root.wxyz = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)

    @stairs_x.on_update
    def _(_evt) -> None:
        _apply_stairs_pose()

    @stairs_y.on_update
    def _(_evt) -> None:
        _apply_stairs_pose()

    @stairs_z.on_update
    def _(_evt) -> None:
        _apply_stairs_pose()

    @stairs_yaw.on_update
    def _(_evt) -> None:
        _apply_stairs_pose()

    print(f"Loaded motion: {args.npz.resolve()} | frames={qpos.shape[0]} | fps={actual_fps}")
    print(f"Loaded stairs: {stairs_urdf_path}")
    print("Open the Viser URL printed above. Use the 'Stairs Placement' controls to align the staircase.")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
