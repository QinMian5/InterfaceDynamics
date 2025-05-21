# Author: Mian Qin
# Date Created: 2025/5/6
from pathlib import Path

import numpy as np
import gmsh
import pyvista
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI
from torch.distributions.constraints import symmetric

parameters_flat_surface = {
    "45": [70, 40],
    "60": [65, 50],
    "70": [55, 50],
    "80": [55, 55],
    "90": [55, 60],
    "120": [45, 70],
    "150": [45, 80]
}


def main_generate_mesh_flat(theta: str, visualization=False):
    save_dir = Path("./mesh/flat_surface")
    save_dir.mkdir(exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    name = "Flat Surface"
    r_box, z_box = parameters_flat_surface[theta]
    z_plane = 1

    # Create model
    model = gmsh.model()
    model.add(name)
    model.setCurrent(name)

    box_tags = model.occ.addCylinder(0, 0, 0, 0, 0, z_box, r_box)
    plane_tags = model.occ.addCylinder(0, 0, 0, 0, 0, z_plane, r_box)

    surface_region_dim_tags = [(3, plane_tags)]
    water_region_dim_tags, _ = model.occ.cut([(3, box_tags)], [(3, surface_region_dim_tags[0][1])], removeTool=False)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    model.addPhysicalGroup(3, [water_region_dim_tags[0][1]], tag=1, name="water_region")
    model.addPhysicalGroup(3, [surface_region_dim_tags[0][1]], tag=2, name="surface_region")

    water_boundary = model.getBoundary(water_region_dim_tags, oriented=False)
    water_boundary_ids = set(b[1] for b in water_boundary)
    surface_boundary = model.getBoundary(surface_region_dim_tags, oriented=False)
    surface_boundary_ids = set(b[1] for b in surface_boundary)

    water_surface_boundary_ids = list(water_boundary_ids.intersection(surface_boundary_ids))
    water_box_boundary_ids = list(water_boundary_ids - surface_boundary_ids)
    surface_box_boundary_ids = list(surface_boundary_ids - water_boundary_ids)

    model.addPhysicalGroup(2, water_surface_boundary_ids, 1, "water_surface_boundary")
    model.addPhysicalGroup(2, water_box_boundary_ids, 2, "water_box_boundary")
    model.addPhysicalGroup(2, surface_box_boundary_ids, 3, "surface_box_boundary")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
    # gmsh.option.setNumber("Geometry.NumSubEdges", 20)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # gmsh.option.setNumber("Mesh.Optimize", 1)  # 启用网格优化
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # 使用Netgen优化
    # gmsh.option.setNumber("Mesh.QualityType", 2)  # 基于最小角度的质量评估
    gmsh.model.mesh.generate(3)

    mesh_save_path = save_dir / f"{theta}.msh"
    gmsh.write(str(mesh_save_path))

    if visualization:
        gmsh.fltk.run()  # 打开 GMSH 图形界面查看

    gmsh.finalize()


parameters_pillar = {
    "45": [70+10, 40],
    "60": [65+10, 50],
    "70": [55+10, 50],
    "80": [55+10, 55],
    "90": [55+10, 60],
    "120": [45+10, 70],
    "150": [45+10, 80]
}


def main_generate_mesh_pillar(theta: str, r_pillar: float, symmetric=True, visualization=False):
    root_dir = Path("./mesh/pillar_system")
    root_dir.mkdir(exist_ok=True)
    save_dir = root_dir / f"{r_pillar}"
    save_dir.mkdir(exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    name = "Pillar System"
    r_box, z_box = parameters_pillar[theta]
    z_plane = 1
    z_pillar = z_box - 1
    x_cylinder = 0 if symmetric else r_box - r_pillar - 10

    # Create model
    model = gmsh.model()
    model.add(name)
    model.setCurrent(name)

    box_tags = model.occ.addCylinder(0, 0, 0, 0, 0, z_box, r_box)
    cylinder_tags = model.occ.addCylinder(x_cylinder, 0, 0, 0, 0, z_pillar, r_pillar)
    plane_tags = model.occ.addCylinder(0, 0, 0, 0, 0, z_plane, r_box)

    surface_region_dim_tags, _ = model.occ.fuse([(3, cylinder_tags)], [(3, plane_tags)])
    water_region_dim_tags, _ = model.occ.cut([(3, box_tags)], [(3, surface_region_dim_tags[0][1])], removeTool=False)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    model.addPhysicalGroup(3, [water_region_dim_tags[0][1]], tag=1, name="water_region")
    model.addPhysicalGroup(3, [surface_region_dim_tags[0][1]], tag=2, name="surface_region")

    water_boundary = model.getBoundary(water_region_dim_tags, oriented=False)
    water_boundary_ids = set(b[1] for b in water_boundary)
    surface_boundary = model.getBoundary(surface_region_dim_tags, oriented=False)
    surface_boundary_ids = set(b[1] for b in surface_boundary)

    water_surface_boundary_ids = list(water_boundary_ids.intersection(surface_boundary_ids))
    water_box_boundary_ids = list(water_boundary_ids - surface_boundary_ids)
    surface_box_boundary_ids = list(surface_boundary_ids - water_boundary_ids)

    model.addPhysicalGroup(2, water_surface_boundary_ids, 1, "water_surface_boundary")
    model.addPhysicalGroup(2, water_box_boundary_ids, 2, "water_box_boundary")
    model.addPhysicalGroup(2, surface_box_boundary_ids, 3, "surface_box_boundary")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
    # gmsh.option.setNumber("Geometry.NumSubEdges", 20)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # gmsh.option.setNumber("Mesh.Optimize", 1)  # 启用网格优化
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # 使用Netgen优化
    # gmsh.option.setNumber("Mesh.QualityType", 2)  # 基于最小角度的质量评估
    gmsh.model.mesh.generate(3)

    # mesh_save_path = save_dir / f"{theta}.msh"
    if symmetric:
        mesh_save_path = save_dir / f"{theta}.msh"
    else:
        mesh_save_path = save_dir / f"{theta}_asym.msh"
    gmsh.write(str(mesh_save_path))

    if visualization:
        gmsh.fltk.run()  # 打开 GMSH 图形界面查看

    gmsh.finalize()


def load_mesh():
    ...


def main():
    # for theta in parameters_flat_surface.keys():
    #     main_generate_mesh_flat(theta)

    r_pillar = 10
    for theta in parameters_pillar.keys():
    # for theta in ["45", "60", "90"]:
        main_generate_mesh_pillar(theta, r_pillar, symmetric=True, visualization=False)
        main_generate_mesh_pillar(theta, r_pillar, symmetric=False, visualization=False)
    # main_generate_mesh_pillar("90", r_pillar)


if __name__ == "__main__":
    main()
