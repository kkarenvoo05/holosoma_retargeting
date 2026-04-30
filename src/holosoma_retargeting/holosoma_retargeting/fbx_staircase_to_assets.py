#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_FBX = (
    PACKAGE_ROOT / "demo_data" / "TML_data" / "up_continuous_v2_staircase_hierarchy.fbx"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "demo_data" / "climb" / DEFAULT_FBX.stem
SCENE_TEMPLATE = (
    PACKAGE_ROOT / "demo_data" / "climb" / "staircase" / "g1_29dof_spherehand_w_multi_boxes.xml"
)


@dataclass
class FbxGeometry:
    geometry_id: int
    name: str
    vertices: np.ndarray
    polygons: list[list[int]]


@dataclass
class FbxModel:
    model_id: int
    name: str
    translation: np.ndarray
    rotation_deg: np.ndarray
    scaling: np.ndarray


@dataclass
class StairMesh:
    name: str
    vertices: np.ndarray
    polygons: list[list[int]]
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def _extract_block(text: str, header: str) -> str:
    start = text.find(header)
    if start == -1:
        raise ValueError(f"Could not find FBX block starting with: {header}")

    brace_start = text.find("{", start)
    if brace_start == -1:
        raise ValueError(f"Could not find opening brace for block: {header}")

    depth = 0
    for idx in range(brace_start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError(f"Could not find closing brace for block: {header}")


def _parse_number_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.replace("\n", "").split(",") if token.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.replace("\n", "").split(",") if token.strip()]


def _polygon_indices_to_faces(indices: list[int]) -> list[list[int]]:
    faces: list[list[int]] = []
    current: list[int] = []
    for idx in indices:
        if idx < 0:
            current.append((-idx) - 1)
            faces.append(current)
            current = []
        else:
            current.append(idx)
    if current:
        faces.append(current)
    return faces


def _parse_ascii_fbx(fbx_path: Path) -> list[StairMesh]:
    text = fbx_path.read_text()
    if "Objects:" not in text or "Connections:" not in text:
        raise ValueError("Unsupported FBX layout. This script expects an ASCII FBX with Objects/Connections blocks.")

    objects_block = _extract_block(text, "Objects:")
    connections_block = _extract_block(text, "Connections:")

    geometries: dict[int, FbxGeometry] = {}
    geometry_headers = re.findall(r'Geometry:\s*(\d+),\s*"Geometry::([^"]+)",\s*"Mesh"\s*\{', objects_block)
    for geometry_id_raw, name in geometry_headers:
        block = _extract_block(objects_block, f'Geometry: {geometry_id_raw}, "Geometry::{name}", "Mesh"')
        vertices_match = re.search(r"Vertices:\s*\*\d+\s*\{\s*a:\s*([^}]*)\}", block, re.S)
        polygons_match = re.search(r"PolygonVertexIndex:\s*\*\d+\s*\{\s*a:\s*([^}]*)\}", block, re.S)
        if vertices_match is None or polygons_match is None:
            continue

        vertices_flat = _parse_number_list(vertices_match.group(1))
        vertices = np.asarray(vertices_flat, dtype=float).reshape(-1, 3)
        polygon_indices = _parse_int_list(polygons_match.group(1))

        geometries[int(geometry_id_raw)] = FbxGeometry(
            geometry_id=int(geometry_id_raw),
            name=name,
            vertices=vertices,
            polygons=_polygon_indices_to_faces(polygon_indices),
        )

    models: dict[int, FbxModel] = {}
    model_headers = re.findall(r'Model:\s*(\d+),\s*"Model::([^"]+)",\s*"Mesh"\s*\{', objects_block)
    for model_id_raw, name in model_headers:
        block = _extract_block(objects_block, f'Model: {model_id_raw}, "Model::{name}", "Mesh"')
        translation_match = re.search(
            r'P:\s*"Lcl Translation",\s*"Lcl Translation",\s*"",\s*"A",\s*([^\n]+)',
            block,
        )
        rotation_match = re.search(
            r'P:\s*"Lcl Rotation",\s*"Lcl Rotation",\s*"",\s*"A",\s*([^\n]+)',
            block,
        )
        scaling_match = re.search(
            r'P:\s*"Lcl Scaling",\s*"Lcl Scaling",\s*"",\s*"A",\s*([^\n]+)',
            block,
        )
        if translation_match is None:
            continue

        translation = np.asarray(_parse_number_list(translation_match.group(1)), dtype=float)
        rotation = (
            np.asarray(_parse_number_list(rotation_match.group(1)), dtype=float)
            if rotation_match is not None
            else np.zeros(3)
        )
        scaling = (
            np.asarray(_parse_number_list(scaling_match.group(1)), dtype=float)
            if scaling_match is not None
            else np.ones(3)
        )

        models[int(model_id_raw)] = FbxModel(
            model_id=int(model_id_raw),
            name=name,
            translation=translation,
            rotation_deg=rotation,
            scaling=scaling,
        )

    geometry_to_model: dict[int, int] = {}
    for child_raw, parent_raw in re.findall(r'C:\s*"OO",\s*(\d+),\s*(\d+)', connections_block):
        child = int(child_raw)
        parent = int(parent_raw)
        if child in geometries and parent in models:
            geometry_to_model[child] = parent

    stair_meshes: list[StairMesh] = []
    for geometry_id, geometry in geometries.items():
        if geometry_id not in geometry_to_model:
            continue

        model = models[geometry_to_model[geometry_id]]
        if not np.allclose(model.rotation_deg, 0.0):
            raise ValueError(f"Model '{model.name}' has non-zero rotation {model.rotation_deg}; this exporter only handles translation.")
        if not np.allclose(model.scaling, 1.0):
            raise ValueError(f"Model '{model.name}' has non-unit scaling {model.scaling}; this exporter only handles unit scale.")

        translated_vertices = geometry.vertices + model.translation.reshape(1, 3)
        stair_meshes.append(
            StairMesh(
                name=model.name,
                vertices=translated_vertices,
                polygons=geometry.polygons,
                bounds_min=translated_vertices.min(axis=0),
                bounds_max=translated_vertices.max(axis=0),
            )
        )

    if not stair_meshes:
        raise ValueError(f"No staircase meshes were extracted from {fbx_path}")

    stair_meshes.sort(key=lambda mesh: mesh.bounds_max[1])
    return stair_meshes


def _convert_y_up_to_z_up(vertices: np.ndarray, unit_scale: float, flip_y: bool) -> np.ndarray:
    scaled = vertices * unit_scale
    x = scaled[:, 0]
    y = scaled[:, 1]
    z = scaled[:, 2]
    converted_y = -z if flip_y else z
    return np.stack([x, converted_y, y], axis=1)


def _format_obj(vertices: np.ndarray, polygons: list[list[int]], name: str) -> str:
    lines = [f"# {name}"]
    for vertex in vertices:
        lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
    for polygon in polygons:
        face = " ".join(str(index + 1) for index in polygon)
        lines.append(f"f {face}")
    lines.append("")
    return "\n".join(lines)


def _merge_meshes(meshes: list[StairMesh]) -> tuple[np.ndarray, list[list[int]]]:
    merged_vertices: list[np.ndarray] = []
    merged_faces: list[list[int]] = []
    vertex_offset = 0
    for mesh in meshes:
        merged_vertices.append(mesh.vertices)
        for polygon in mesh.polygons:
            merged_faces.append([index + vertex_offset for index in polygon])
        vertex_offset += mesh.vertices.shape[0]
    return np.vstack(merged_vertices), merged_faces


def _with_updated_bounds(mesh: StairMesh, vertices: np.ndarray) -> StairMesh:
    return StairMesh(
        name=mesh.name,
        vertices=vertices,
        polygons=mesh.polygons,
        bounds_min=vertices.min(axis=0),
        bounds_max=vertices.max(axis=0),
    )


def _extend_step_height(mesh: StairMesh, extra_height: float) -> StairMesh:
    if extra_height == 0.0:
        return mesh
    vertices = mesh.vertices.copy()
    z_max = float(vertices[:, 2].max())
    top_mask = np.isclose(vertices[:, 2], z_max)
    vertices[top_mask, 2] += extra_height
    return _with_updated_bounds(mesh, vertices)


def _extend_step_length(mesh: StairMesh, length_scale: float) -> StairMesh:
    if np.isclose(length_scale, 1.0):
        return mesh
    vertices = mesh.vertices.copy()

    # Recover the bottom face as a skewed quadrilateral, then scale the shorter
    # adjacent edge so the top step grows as a proper parallelogram landing.
    z_min = float(vertices[:, 2].min())
    bottom_indices = np.where(np.isclose(vertices[:, 2], z_min))[0]
    if bottom_indices.size != 4:
        raise ValueError(f"Expected 4 bottom vertices for {mesh.name}, got {bottom_indices.size}")

    base = vertices[bottom_indices, :2]
    anchor_local = 0
    vectors = [(idx, base[idx] - base[anchor_local]) for idx in range(1, 4)]
    vectors.sort(key=lambda item: float(np.linalg.norm(item[1])))
    depth_local = vectors[0][0]
    width_local = vectors[1][0]
    depth_vec = base[depth_local] - base[anchor_local]

    diag_local = next(idx for idx in range(4) if idx not in {anchor_local, depth_local, width_local})

    coeff_matrix = np.column_stack([base[width_local] - base[anchor_local], depth_vec])
    coeff_inv = np.linalg.inv(coeff_matrix)

    for global_idx in bottom_indices:
        coeff = coeff_inv @ (vertices[global_idx, :2] - base[anchor_local])
        coeff[1] *= length_scale
        vertices[global_idx, :2] = base[anchor_local] + (coeff_matrix @ coeff)

    top_indices = np.where(~np.isclose(vertices[:, 2], z_min))[0]
    for global_idx in top_indices:
        coeff = coeff_inv @ (vertices[global_idx, :2] - base[anchor_local])
        coeff[1] *= length_scale
        vertices[global_idx, :2] = base[anchor_local] + (coeff_matrix @ coeff)

    return _with_updated_bounds(mesh, vertices)


def _apply_step_adjustments(
    meshes: list[StairMesh],
    first_step_height_offset: float,
    third_step_length_scale: float,
) -> list[StairMesh]:
    adjusted = list(meshes)
    if adjusted:
        adjusted[0] = _extend_step_height(adjusted[0], first_step_height_offset)
    if len(adjusted) >= 3:
        adjusted[2] = _extend_step_length(adjusted[2], third_step_length_scale)
    return adjusted


def _relative_mesh_path(output_dir: Path, mesh_file: Path) -> str:
    return mesh_file.relative_to(output_dir).as_posix()


def _write_urdf(output_dir: Path, box_mesh_files: list[Path]) -> None:
    colors = [
        "0.3 0.7 0.9 0.5",
        "0.7 0.3 0.9 0.5",
        "0.9 0.7 0.3 0.5",
    ]
    lines = ['<?xml version="1.0"?>', '<robot name="multi_boxes">', '  <link name="world"/>', ""]

    for idx, mesh_path in enumerate(box_mesh_files, start=1):
        mesh_rel = _relative_mesh_path(output_dir, mesh_path)
        lines.extend(
            [
                f'  <link name="multi_boxes_box{idx}_link">',
                "    <visual>",
                '      <origin rpy="0 0 0" xyz="0 0 0"/>',
                "      <geometry>",
                f'        <mesh filename="{mesh_rel}" scale="1 1 1"/>',
                "      </geometry>",
                f'      <material name="box{idx}_material">',
                f'        <color rgba="{colors[idx - 1]}"/>',
                "      </material>",
                "    </visual>",
                "    <collision>",
                '      <origin rpy="0 0 0" xyz="0 0 0"/>',
                "      <geometry>",
                f'        <mesh filename="{mesh_rel}" scale="1 1 1"/>',
                "      </geometry>",
                "    </collision>",
                "    <inertial>",
                '      <mass value="33.33"/>',
                '      <origin rpy="0 0 0" xyz="0 0 0"/>',
                '      <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="10.0"/>',
                "    </inertial>",
                "  </link>",
                "",
                f'  <joint name="world_to_box{idx}" type="fixed">',
                '    <parent link="world"/>',
                f'    <child link="multi_boxes_box{idx}_link"/>',
                '    <origin rpy="0 0 0" xyz="0 0 0"/>',
                "  </joint>",
                "",
            ]
        )

    lines.append("</robot>")
    (output_dir / "multi_boxes.urdf").write_text("\n".join(lines) + "\n")


def _write_box_assets(output_dir: Path, box_mesh_files: list[Path]) -> None:
    colors = [
        "0.3 0.7 0.9 0.5",
        "0.7 0.3 0.9 0.5",
        "0.9 0.7 0.3 0.5",
    ]
    lines = ["<mujocoinclude>"]
    for idx, mesh_path in enumerate(box_mesh_files, start=1):
        mesh_rel = _relative_mesh_path(output_dir, mesh_path)
        lines.append(f'    <mesh name="box{idx}" file="{mesh_rel}" scale="1 1 1"/>')
    for idx, color in enumerate(colors, start=1):
        lines.append(f'    <material name="box{idx}_material" rgba="{color}"/>')
    lines.append("</mujocoinclude>")
    (output_dir / "box_assets.xml").write_text("\n".join(lines) + "\n")


def _write_box_body(output_dir: Path, mesh_count: int) -> None:
    lines = ["<mujocoinclude>"]
    for idx in range(1, mesh_count + 1):
        lines.extend(
            [
                f'    <body name="multi_boxes_box{idx}_link" pos="0 0 0" quat="1 0 0 0">',
                f'        <geom name="multi_boxes_link_{idx}" type="mesh" mesh="box{idx}" pos="0 0 0" quat="1 0 0 0" material="box{idx}_material" contype="1" conaffinity="1"/>',
                "    </body>",
                "",
            ]
        )
    lines.append("</mujocoinclude>")
    (output_dir / "box_body.xml").write_text("\n".join(lines) + "\n")


def _write_box_environment(output_dir: Path, meshes: list[StairMesh], box_mesh_files: list[Path]) -> None:
    colors = [
        "0.3 0.7 0.9 0.8",
        "0.7 0.3 0.9 0.8",
        "0.9 0.7 0.3 0.8",
    ]
    lines = [
        "<mujoco model=\"stair_environment\">",
        "  <compiler angle=\"radian\"/>",
        "",
        "  <asset>",
        '    <material name="ground_material" rgba="0.5 0.5 0.5 1"/>',
        "",
    ]

    for idx, mesh_path in enumerate(box_mesh_files, start=1):
        mesh_rel = _relative_mesh_path(output_dir, mesh_path)
        lines.append(f'    <mesh name="stair_step{idx}" file="{mesh_rel}"/>')
    lines.append("")
    for idx, color in enumerate(colors, start=1):
        lines.append(f'    <material name="stair_step{idx}_material" rgba="{color}"/>')

    lines.extend(
        [
            "  </asset>",
            "",
            "  <worldbody>",
            '    <geom name="ground" type="plane" size="10 10 0.1" pos="0 0 0" material="ground_material"/>',
            "",
        ]
    )

    for idx, mesh in enumerate(meshes, start=1):
        marker = np.array([mesh.bounds_min[0], mesh.bounds_max[1], mesh.bounds_max[2]])
        lines.extend(
            [
                f'    <body name="stair_step{idx}_body" pos="0 0 0" quat="1 0 0 0">',
                f'      <geom name="stair_step{idx}_geom" type="mesh" mesh="stair_step{idx}" pos="0 0 0" quat="1 0 0 0" material="stair_step{idx}_material" contype="1" conaffinity="1"/>',
                "    </body>",
                f'    <site name="stair_marker_{idx}" type="sphere" size="0.02" pos="{marker[0]:.6f} {marker[1]:.6f} {marker[2]:.6f}" rgba="{1 if idx == 1 else 0} {1 if idx == 2 else 0} {1 if idx == 3 else 0} 1"/>',
                "",
            ]
        )

    lines.extend(["  </worldbody>", "", "</mujoco>"])
    (output_dir / "box_environment.xml").write_text("\n".join(lines) + "\n")


def _write_metadata(
    output_dir: Path,
    fbx_path: Path,
    meshes: list[StairMesh],
    unit_scale: float,
    flip_y: bool,
    first_step_height_offset: float,
    third_step_length_scale: float,
) -> None:
    metadata = {
        "source_fbx": str(fbx_path),
        "unit_scale_to_meters": unit_scale,
        "axis_conversion": "fbx_y_up_to_z_up_with_negated_y" if flip_y else "fbx_y_up_to_z_up",
        "adjustments": {
            "first_step_height_offset_m": first_step_height_offset,
            "third_step_length_scale": third_step_length_scale,
            "third_step_length_axis": "x",
            "third_step_length_anchor": "front_edge_x_max",
        },
        "stairs": [],
    }
    for mesh in meshes:
        bounds_size = mesh.bounds_max - mesh.bounds_min
        metadata["stairs"].append(
            {
                "name": mesh.name,
                "bounds_min_m": mesh.bounds_min.tolist(),
                "bounds_max_m": mesh.bounds_max.tolist(),
                "size_m": bounds_size.tolist(),
                "top_front_left_marker_m": [mesh.bounds_min[0], mesh.bounds_max[1], mesh.bounds_max[2]],
            }
        )

    all_vertices = np.vstack([mesh.vertices for mesh in meshes])
    metadata["combined_bounds_min_m"] = all_vertices.min(axis=0).tolist()
    metadata["combined_bounds_max_m"] = all_vertices.max(axis=0).tolist()
    (output_dir / "staircase_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def export_staircase_assets(
    fbx_path: Path,
    output_dir: Path,
    unit_scale: float,
    flip_y: bool,
    first_step_height_offset: float,
    third_step_length_scale: float,
) -> None:
    raw_meshes = _parse_ascii_fbx(fbx_path)
    converted_meshes: list[StairMesh] = []
    for raw_mesh in raw_meshes:
        converted_vertices = _convert_y_up_to_z_up(raw_mesh.vertices, unit_scale=unit_scale, flip_y=flip_y)
        converted_meshes.append(
            StairMesh(
                name=raw_mesh.name,
                vertices=converted_vertices,
                polygons=raw_mesh.polygons,
                bounds_min=converted_vertices.min(axis=0),
                bounds_max=converted_vertices.max(axis=0),
            )
        )
    converted_meshes = _apply_step_adjustments(
        converted_meshes,
        first_step_height_offset=first_step_height_offset,
        third_step_length_scale=third_step_length_scale,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    box_models_dir = output_dir / "box_models"
    box_models_dir.mkdir(parents=True, exist_ok=True)

    box_mesh_files: list[Path] = []
    for idx, mesh in enumerate(converted_meshes, start=1):
        box_path = box_models_dir / f"box{idx}.obj"
        box_path.write_text(_format_obj(mesh.vertices, mesh.polygons, mesh.name))
        box_mesh_files.append(box_path)

    merged_vertices, merged_faces = _merge_meshes(converted_meshes)
    (output_dir / "multi_boxes.obj").write_text(_format_obj(merged_vertices, merged_faces, "multi_boxes"))

    _write_urdf(output_dir, box_mesh_files)
    _write_box_assets(output_dir, box_mesh_files)
    _write_box_body(output_dir, len(box_mesh_files))
    _write_box_environment(output_dir, converted_meshes, box_mesh_files)
    _write_metadata(
        output_dir,
        fbx_path,
        converted_meshes,
        unit_scale,
        flip_y,
        first_step_height_offset,
        third_step_length_scale,
    )

    if SCENE_TEMPLATE.exists():
        shutil.copyfile(SCENE_TEMPLATE, output_dir / SCENE_TEMPLATE.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an ASCII FBX staircase into a climb-style asset folder with OBJ/URDF/XML files."
    )
    parser.add_argument("--fbx", type=Path, default=DEFAULT_FBX, help="Path to the input ASCII FBX staircase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder to create, similar to demo_data/climb/staircase.",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=0.01,
        help="Scale applied to FBX coordinates before export. Use 0.01 for centimeter-to-meter conversion.",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Keep the converted Y axis positive instead of negating it after Y-up to Z-up conversion.",
    )
    parser.add_argument(
        "--first-step-height-offset",
        type=float,
        default=0.0,
        help="Extra height in meters to add only to the first step.",
    )
    parser.add_argument(
        "--third-step-length-scale",
        type=float,
        default=1.0,
        help="Scale factor for the third step length along the walking-direction x-axis.",
    )
    args = parser.parse_args()

    export_staircase_assets(
        fbx_path=args.fbx.resolve(),
        output_dir=args.output_dir.resolve(),
        unit_scale=args.unit_scale,
        flip_y=not args.no_flip_y,
        first_step_height_offset=args.first_step_height_offset,
        third_step_length_scale=args.third_step_length_scale,
    )

    print(f"Wrote staircase assets to: {args.output_dir.resolve()}")
    if SCENE_TEMPLATE.exists():
        print(f"Scene XML template copied to: {(args.output_dir / SCENE_TEMPLATE.name).resolve()}")


if __name__ == "__main__":
    main()
