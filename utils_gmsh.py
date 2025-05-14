# Author: Mian Qin
# Date Created: 2025/5/6
import numpy as np
import gmsh
import pyvista
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI


def gmsh_flat_surface(model: gmsh.model, x_box, y_box, z_box, z_plane, name="Flat Surface") -> gmsh.model:
    model.add(name)
    model.setCurrent(name)

    box_tags = model.occ.addBox(0, 0, 0, x_box, y_box, z_box)
    plane_tags = model.occ.addBox(0, 0, 0, x_box, y_box, z_plane)

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
    return model


def gmsh_pillar_system(model: gmsh.model, x_box, y_box, z_box, z_plane, r_pillar, z_pillar, name="Pillar System") -> gmsh.model:
    model.add(name)
    model.setCurrent(name)

    x_pillar = x_box / 2
    y_pillar = y_box / 2

    box_tags = model.occ.addBox(0, 0, 0, x_box, y_box, z_box)
    cylinder_tags = model.occ.addCylinder(x_pillar, y_pillar, 0, 0, 0, z_pillar, r_pillar)
    plane_tags = model.occ.addBox(0, 0, 0, x_box, y_box, z_plane)

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
    return model


def main_generate_mesh_flat():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Create model
    model = gmsh.model()
    model = gmsh_flat_surface(model, 100, 100, 80, 5)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
    # gmsh.option.setNumber("Geometry.NumSubEdges", 20)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # gmsh.option.setNumber("Mesh.Optimize", 1)  # 启用网格优化
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # 使用Netgen优化
    # gmsh.option.setNumber("Mesh.QualityType", 2)  # 基于最小角度的质量评估
    gmsh.model.mesh.generate(3)

    # 保存为 .msh 文件
    gmsh.write("mesh/flat_surface.msh")

    # 可视化（可选）
    gmsh.fltk.run()  # 打开 GMSH 图形界面查看

    # 结束 GMSH
    gmsh.finalize()


def main_generate_mesh_pillar():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    # Create model
    model = gmsh.model()
    model = gmsh_pillar_system(model, 100, 100, 60, 5, 10, 60)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
    # gmsh.option.setNumber("Geometry.NumSubEdges", 20)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # gmsh.option.setNumber("Mesh.Optimize", 1)  # 启用网格优化
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # 使用Netgen优化
    # gmsh.option.setNumber("Mesh.QualityType", 2)  # 基于最小角度的质量评估
    gmsh.model.mesh.generate(3)

    # 保存为 .msh 文件
    gmsh.write("mesh/pillar_system.msh")

    # 可视化（可选）
    gmsh.fltk.run()  # 打开 GMSH 图形界面查看

    # 结束 GMSH
    gmsh.finalize()


def load_mesh():
    ...


def main():
    # main_generate_mesh_flat()
    main_generate_mesh_pillar()


if __name__ == "__main__":
    main()
