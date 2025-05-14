# Author: Mian Qin
# Date Created: 2025/5/2
import pickle
from pathlib import Path
from itertools import accumulate

from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.nls.petsc import NewtonSolver
import ufl
import basix.ufl
import pyvista
import gmsh
import trimesh
import numpy as np
import scipy.constants as const
import pandas as pd

from scifem import create_real_functionspace

from utils import create_lambda_star_func, setup_logger

ksi = 4.0


def clip_petsc_vec(vec, min_val=0.0, max_val=1.0):
    """限制 PETSc 向量的值范围"""
    arr = vec.getArray()
    arr[arr > max_val] = max_val
    arr[arr < min_val] = min_val
    vec.setArray(arr)
    vec.ghostUpdate()


def compute_form(domain, form: ufl.Form):
    integrate_form = fem.assemble_scalar(fem.form(form))
    integrate_form = domain.comm.allreduce(integrate_form, op=MPI.SUM)
    return integrate_form


def compute_volume(domain, phi):
    volume = compute_form(domain, phi * ufl.dx)
    return volume


def compute_A_iw_form(phi):
    A_iw_form = 1 / ksi * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
    return A_iw_form


def compute_surface_energy_iw_form(phi, gamma_iw):
    A_iw_form = compute_A_iw_form(phi)
    surface_energy_iw_form = gamma_iw * A_iw_form
    return surface_energy_iw_form


def compute_A_is_form(phi):
    A_is_form = phi * ufl.ds
    return A_is_form


def compute_surface_energy_is_form(phi, gamma_is_ws):
    A_is_form = compute_A_is_form(phi)
    surface_energy_is_form = gamma_is_ws / (4 * ksi) * A_is_form
    return surface_energy_is_form


def compute_double_well_energy_form(phi, gamma_iw):
    double_well_energy_form = gamma_iw / (3 * ksi) * phi ** 2 * (1 - phi) ** 2 * ufl.dx
    return double_well_energy_form


def generate_interface(V, phi):
    # 将 FEniCSx 函数 phi 附加到 PyVista 网格
    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["phi"] = phi.x.array.real  # 确保为实数
    grid.set_active_scalars("phi")

    # 提取等值面 (phi=0.5)
    contour = grid.contour(isosurfaces=[0.5])

    nodes = np.array(contour.points)  # 等价于 Marching Cubes 的 vertices

    # 获取等值面的面连接性 (shape: [n_faces, 3] 或 [n_faces, 4])
    faces = np.array(contour.faces.reshape(-1, 4)[:, 1:])
    interface_type = np.zeros(len(faces), dtype=int)
    return nodes, faces, interface_type


def export_surface(domain, facet_tags, water_surface_boundary_tag, save_dir):
    # 获取标记为 water_surface_boundary_tag 的面的拓扑信息
    water_surface_facets = facet_tags.find(water_surface_boundary_tag)

    # 获取这些面片的顶点连接性
    mesh_topology = domain.topology
    mesh_topology.create_connectivity(mesh_topology.dim - 1, mesh_topology.dim)  # 面-体单元连接性
    facet_to_vertex = mesh_topology.connectivity(mesh_topology.dim - 1, 0)  # 面-顶点连接性

    # 收集所有唯一顶点索引
    vertices = []
    for facet in water_surface_facets:
        vertices.extend(facet_to_vertex.links(facet))
    unique_vertices, inverse = np.unique(vertices, return_inverse=True)
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}

    # 获取顶点坐标
    vertex_coords = domain.geometry.x[unique_vertices]

    # 获取面片的顶点连接性（每个面片由3个顶点组成）
    triangles = []
    for facet in water_surface_facets:
        facet_vertices = facet_to_vertex.links(facet)
        triangles.append([vertex_map[v] for v in facet_vertices])
    triangles = np.array(triangles)

    mesh = trimesh.Trimesh(vertices=vertex_coords, faces=triangles)
    save_dir.mkdir(exist_ok=True)
    mesh.export(save_dir / "surface.ply", file_type="ply", encoding='ascii')


def main_Poisson():
    domain, cell_tags, facet_tags = gmshio.read_from_msh(
        "mesh/pillar_system.msh",
        MPI.COMM_WORLD,
        rank=0,
        gdim=3  # 几何维度 (3D)
    )
    water_region_tag = 1
    surface_region_tag = 2
    water_surface_boundary_tag = 1
    water_box_boundary_tag = 2
    surface_box_boundary_tag = 3

    tdim = domain.topology.dim
    fdim = tdim - 1

    water_cells = cell_tags.find(water_region_tag)
    domain_water, orig_cell_map, vertex_map = mesh.create_submesh(domain, tdim, water_cells)[:3]

    # 获取子网格的 `facet → cell` 连接关系
    domain_water.topology.create_connectivity(fdim, tdim)
    facet_to_cell = domain_water.topology.connectivity(fdim, tdim)

    # 获取原始网格的 `cell → facet` 连接关系
    domain.topology.create_connectivity(tdim, fdim)
    cell_to_facet = domain.topology.connectivity(tdim, fdim)

    sub_to_orig_facet_map = {}
    orig_to_sub_facet_map = {}
    submesh_facets = mesh.locate_entities_boundary(
        domain_water, fdim,
        lambda x: np.full(x.shape[1], True)  # 选择所有边界面
    )
    for sub_facet in submesh_facets:
        # 子网格中该面所属的单元
        sub_cell = facet_to_cell.links(sub_facet)[0]

        # 对应的原始网格单元
        orig_cell = orig_cell_map[sub_cell]

        # 原始网格中该单元的所有面
        orig_facets = cell_to_facet.links(orig_cell)

        # 比较面的顶点（或使用局部索引，如果顺序一致）
        sub_vertices = domain_water.topology.connectivity(fdim, 0).links(sub_facet)
        orig_sub_vertices = vertex_map[sub_vertices]
        for orig_facet in orig_facets:
            orig_vertices = domain.topology.connectivity(fdim, 0).links(orig_facet)
            if np.allclose(
                    np.sort(orig_sub_vertices),
                    np.sort(orig_vertices),
                    atol=1e-10
            ):
                sub_to_orig_facet_map[sub_facet] = orig_facet
                orig_to_sub_facet_map[orig_facet] = sub_facet
                break

    indices = []
    facet_values = []
    for facet_value in [1, 2, 3]:
        facets = facet_tags.find(facet_value)
        for orig_index in facets:
            if orig_index in orig_to_sub_facet_map:
                sub_index = orig_to_sub_facet_map[orig_index]
                indices.append(sub_index)
                facet_values.append(facet_value)

    # 创建子网格的 `facet_tags`
    facet_tags_water = mesh.meshtags(
        domain_water, fdim,
        np.array(indices, dtype=np.int32),
        np.array(facet_values, dtype=np.int32),
    )

    V = fem.functionspace(domain_water, ("Lagrange", 1))

    domain_water.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain_water.topology)

    water_surface_facets = facet_tags_water.find(water_surface_boundary_tag)
    dofs_water_surface = fem.locate_dofs_topological(V, fdim, water_surface_facets)
    bc_water_surface = fem.dirichletbc(default_scalar_type(1.0), dofs_water_surface, V)

    water_box_facets = facet_tags_water.find(water_box_boundary_tag)
    dofs_water_box = fem.locate_dofs_topological(V, fdim, water_box_facets)
    bc_water_box = fem.dirichletbc(default_scalar_type(0.0), dofs_water_box, V)

    # bcs = [bc_water_surface, bc_water_box]
    bcs = [bc_water_box, bc_water_surface]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain_water, default_scalar_type(0.0))

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # 创建并求解线性问题
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    with XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain_water)
        xdmf.write_function(uh)


def main_PFM(system, theta: str):
    root_dir = Path("./FEM")
    root_dir.mkdir(exist_ok=True)
    assert system in ["pillar", "flat"]
    if system == "pillar":
        path_mesh = Path("./mesh/pillar_system.msh")
        save_dir = root_dir / "pillar" / f"{theta}"
    elif system == "flat":
        path_mesh = Path("./mesh/flat_surface.msh")
        save_dir = root_dir / "flat" / f"{theta}"
    else:
        raise ValueError(f"System {system} not supported")
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = setup_logger(save_dir)
    # logger.info = print

    trajectory_save_dir = save_dir / "trajectory"
    trajectory_save_dir.mkdir(exist_ok=True)

    t_eql_init = 10000
    ramp_rate = 1.0
    t_eql_prd = 1000

    lambda_star = 300

    # 梯度下降参数
    learning_rate_phi = 0.05
    learning_rate_l = 1e-9
    tolerance = 1e-6

    cos_theta = np.cos(np.radians(float(theta)))
    rho = 5.09e4 * 1e-30 * const.N_A  # 1/A^3
    delta_mu = 0  # J/mol
    gamma_iw = 25.9e-3 * 1e-20  # J/A^2, real water
    gamma_is_ws = -cos_theta * gamma_iw

    domain, cell_tags, facet_tags = gmshio.read_from_msh(
        path_mesh,
        MPI.COMM_WORLD,
        rank=0,
        gdim=3  # 几何维度 (3D)
    )
    water_region_tag = 1
    surface_region_tag = 2
    water_surface_boundary_tag = 1
    water_box_boundary_tag = 2
    surface_box_boundary_tag = 3

    export_surface(domain, facet_tags, water_surface_boundary_tag, save_dir)

    tdim = domain.topology.dim
    fdim = tdim - 1

    water_cells = cell_tags.find(water_region_tag)
    domain_water, orig_cell_map, vertex_map = mesh.create_submesh(domain, tdim, water_cells)[:3]

    # 获取子网格的 `facet → cell` 连接关系
    domain_water.topology.create_connectivity(fdim, tdim)
    facet_to_cell = domain_water.topology.connectivity(fdim, tdim)

    # 获取原始网格的 `cell → facet` 连接关系
    domain.topology.create_connectivity(tdim, fdim)
    cell_to_facet = domain.topology.connectivity(tdim, fdim)

    sub_to_orig_facet_map = {}
    orig_to_sub_facet_map = {}
    submesh_facets = mesh.locate_entities_boundary(
        domain_water, fdim,
        lambda x: np.full(x.shape[1], True)  # 选择所有边界面
    )
    for sub_facet in submesh_facets:
        # 子网格中该面所属的单元
        sub_cell = facet_to_cell.links(sub_facet)[0]

        # 对应的原始网格单元
        orig_cell = orig_cell_map[sub_cell]

        # 原始网格中该单元的所有面
        orig_facets = cell_to_facet.links(orig_cell)

        # 比较面的顶点（或使用局部索引，如果顺序一致）
        sub_vertices = domain_water.topology.connectivity(fdim, 0).links(sub_facet)
        orig_sub_vertices = vertex_map[sub_vertices]
        for orig_facet in orig_facets:
            orig_vertices = domain.topology.connectivity(fdim, 0).links(orig_facet)
            if np.allclose(
                    np.sort(orig_sub_vertices),
                    np.sort(orig_vertices),
                    atol=1e-10
            ):
                sub_to_orig_facet_map[sub_facet] = orig_facet
                orig_to_sub_facet_map[orig_facet] = sub_facet
                break

    indices = []
    facet_values = []
    for facet_value in [1, 2, 3]:
        facets = facet_tags.find(facet_value)
        for orig_index in facets:
            if orig_index in orig_to_sub_facet_map:
                sub_index = orig_to_sub_facet_map[orig_index]
                indices.append(sub_index)
                facet_values.append(facet_value)

    # 创建子网格的 `facet_tags`
    facet_tags_water = mesh.meshtags(
        domain_water, fdim,
        np.array(indices, dtype=np.int32),
        np.array(facet_values, dtype=np.int32),
    )

    V = fem.functionspace(domain_water, ("Lagrange", 1))
    domain_water.topology.create_connectivity(fdim, tdim)
    R = create_real_functionspace(domain_water)

    # W = fem.functionspace(domain_water, mixed_element([V.ufl_element(), R.ufl_element()]))
    # print(V.dofmap.index_map.size_global, R.dofmap.index_map.size_global, W.dofmap.index_map.size_global)

    water_box_facets = facet_tags_water.find(water_box_boundary_tag)
    dofs_water_box = fem.locate_dofs_topological(V, fdim, water_box_facets)
    bc_water_box = fem.dirichletbc(default_scalar_type(0.0), dofs_water_box, V)

    bcs = [bc_water_box]

    phi = fem.Function(V)
    l = fem.Function(R)
    current_lambda_star = fem.Function(R)
    x = ufl.SpatialCoordinate(domain_water)
    r = 25
    cos_theta = np.cos(np.radians(float(theta)))
    r_squared = (x[0] - 50) ** 2 + (x[1] - 50) ** 2 + (x[2] - (5 - r * cos_theta)) ** 2
    condition = ufl.conditional(r_squared <= r ** 2, 1.0, 0.0)
    phi.interpolate(fem.Expression(condition, V.element.interpolation_points()))
    lambda_0 = compute_volume(domain_water, phi) * rho
    current_lambda_star.x.array[0] = lambda_0

    t_ramp = int(abs(lambda_0 - lambda_star) / ramp_rate)
    # t_list = list(accumulate([0, t_eql_init, t_ramp, t_eql_prd]))
    t_list = list(accumulate([0, t_eql_init, 0, 0]))
    t_total = t_list[-1]
    lambda_list = [lambda_0, lambda_0, lambda_star, lambda_star]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)
    logger.info(f"lambda: {lambda_star_func(0):.0f} -> {lambda_star_func(t_total-0.1):.0f}, t_total: {t_total:.0f}")
    input("Press Enter to continue.")

    dphi = ufl.TestFunction(V)
    dl = ufl.TestFunction(R)
    energy_scale = const.N_A / 1000  # to kJ/mol
    E = energy_scale * (compute_surface_energy_iw_form(phi, gamma_iw) + compute_surface_energy_is_form(phi, gamma_is_ws) + compute_double_well_energy_form(phi, gamma_iw))
    # E = energy_scale * (K * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx + gamma_is_ws * phi * ufl.ds + h0 * phi ** 2 * (1 - phi) ** 2 * ufl.dx)
    volume = compute_volume(domain_water, fem.Constant(domain_water, 1.0))
    lambda_full = volume * rho
    constraint = -l * (phi - current_lambda_star / lambda_full) * ufl.dx

    E_total = E + constraint
    # E_total = E

    dE_dphi = ufl.derivative(E_total, phi, dphi)
    dE_dl = ufl.derivative(E_total, l, dl)
    dphi_form = fem.form(dE_dphi)
    dl_form = fem.form(dE_dl)

    # 准备组装梯度的向量
    gradient_phi = fem.Function(V)
    gradient_phi_vec = gradient_phi.x.petsc_vec
    gradient_l = fem.Function(R)
    gradient_l_vec = gradient_l.x.petsc_vec
    grad_phi_magnitude = fem.Function(V)
    grad_phi_magnitude_expr = fem.Expression(ufl.sqrt(ufl.dot(ufl.grad(phi), ufl.grad(phi))),
                                             V.element.interpolation_points())
    # dt = fem.Constant(domain_water, 2.0)
    # num_smoothing_steps = 1
    #
    # u = ufl.TrialFunction(V)
    # v = ufl.TestFunction(V)
    # a_smooth = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # L_smooth = grad_phi_magnitude * v * ufl.dx

    # bilinear_form = fem.form(a_smooth)
    # linear_form = fem.form(L_smooth)
    # A = fem.petsc.assemble_matrix(bilinear_form)
    # A.assemble()
    # b = fem.petsc.create_vector(linear_form)

    # smoother = LinearProblem(a_smooth, L_smooth, petsc_options={"ksp_type": "cg", "pc_type": "jacobi"})

    instantaneous_interface_dict = {}
    data = []
    for t in range(t_total+1):
        lambda_star = lambda_star_func(t)
        current_lambda_star.x.array[0] = lambda_star
        # 组装梯度
        with gradient_phi_vec.localForm() as loc_g:
            loc_g.set(0.0)
        fem.petsc.assemble_vector(gradient_phi_vec, dphi_form)
        gradient_phi_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        grad_phi_magnitude.interpolate(grad_phi_magnitude_expr)
        # 对 grad_phi_magnitude 进行 Gaussian 平滑
        # for _ in range(num_smoothing_steps):
        #     grad_phi_magnitude_smoothed = smoother.solve()
        #     grad_phi_magnitude.interpolate(grad_phi_magnitude_smoothed)

        with gradient_phi_vec.localForm() as loc_g, phi.x.petsc_vec.localForm() as loc_phi, grad_phi_magnitude.x.petsc_vec.localForm() as loc_grad_mag:
            # 避免除以零，加一个小常数
            scaling_factor = loc_grad_mag.array / (loc_grad_mag.array.max() + 1e-5)
            scaling_factor[scaling_factor >= 0.2] = 1
            loc_g.array *= scaling_factor

        with gradient_l_vec.localForm() as loc_g:
            loc_g.set(0.0)
        fem.petsc.assemble_vector(gradient_l_vec, dl_form)
        gradient_l_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # 梯度下降更新
        phi.x.petsc_vec.axpy(-learning_rate_phi, gradient_phi_vec)
        l.x.petsc_vec.axpy(learning_rate_l, gradient_l_vec)
        clip_petsc_vec(phi.x.petsc_vec, 0, 1)
        for bc in bcs:
            bc.set(phi.x.array, alpha=1.0)

        # 计算梯度范数以检查收敛性
        grad_norm_phi = gradient_phi_vec.norm()
        grad_norm_l = gradient_l_vec.norm()
        if np.sqrt(grad_norm_phi**2 + grad_norm_l**2) < tolerance:
            logger.info(f"Converged after {t} iterations")
            break
        if t % 100 == 0:
            current_lambda = compute_volume(domain_water, phi) * rho
            logger.info(f"Iteration {t}, Gradient norm: {grad_norm_phi:.2f}, {grad_norm_l:.2f}, Lambda: {current_lambda:.2f}")
            surface_energy_iw = compute_form(domain_water, compute_surface_energy_iw_form(phi, gamma_iw))
            surface_energy_is_iw = compute_form(domain_water, compute_surface_energy_is_form(phi, gamma_is_ws))
            double_well_energy = compute_form(domain_water, compute_double_well_energy_form(phi, gamma_iw))
            total_energy = surface_energy_iw + surface_energy_is_iw + double_well_energy
            logger.info(f"Surface Energy iw: {energy_scale * surface_energy_iw:.2f}, Surface Energy is: {energy_scale * surface_energy_is_iw:.2f}, Double Well Energy: {energy_scale * double_well_energy:.2f}")
            info = {
                "t": t,
                "lambda": current_lambda,
                "total_energy": total_energy,
                "surface_energy_iw": surface_energy_iw,
                "surface_energy_is_iw": surface_energy_is_iw,
                "double_well_energy": double_well_energy,
                "bulk_energy": 0.0
            }
            # print(l.x.array, gradient_l.x.array)
            # print(phi.x.array.min(), phi.x.array.max())
            data.append(info)
            nodes, faces, interface_type = generate_interface(V, phi)
            instantaneous_interface_dict[f"{t}"] = [nodes, faces, interface_type]

            with XDMFFile(MPI.COMM_WORLD, trajectory_save_dir / f"{t:05d}.xdmf", "w") as xdmf:
                xdmf.write_mesh(domain_water)
                xdmf.write_function(phi)
            with XDMFFile(MPI.COMM_WORLD, trajectory_save_dir / f"{t:05d}_grad.xdmf", "w") as xdmf:
                xdmf.write_mesh(domain_water)
                xdmf.write_function(grad_phi_magnitude)

    df = pd.DataFrame(data)
    df.to_csv(save_dir / "intermediate_result.csv", index=False)
    with XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain_water)
        xdmf.write_function(phi)
    with open(save_dir / "instantaneous_interface.pickle", "wb") as file:
        pickle.dump(instantaneous_interface_dict, file)


def main():
    # main_Poisson()
    main_PFM("pillar", "80")


if __name__ == "__main__":
    main()
