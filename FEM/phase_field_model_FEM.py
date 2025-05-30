# Author: Mian Qin
# Date Created: 2025/5/2
import pickle
from pathlib import Path
from itertools import accumulate
import sys
import argparse

from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, default_scalar_type, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio, XDMFFile
import ufl
import basix.ufl
import pyvista
import trimesh
import numpy as np
import scipy.constants as const
import pandas as pd

from scifem import create_real_functionspace
import adios4dolfinx

from utils_FEM import create_lambda_star_func, setup_logger

ksi = 4.0


class GaussianSmoother:
    def __init__(self, phi, sigma, V, domain):
        self.phi = phi
        self.new_phi = fem.Function(V)
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        f = fem.Constant(domain, np.array(0.0))
        a = u * v * ufl.dx + sigma * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (phi + sigma * f) * v * ufl.dx
        self.bilinear_form = fem.form(a)
        self.linear_form = fem.form(L)
        A = fem.petsc.assemble_matrix(self.bilinear_form)
        A.assemble()
        self.b = fem.petsc.create_vector(self.linear_form)
        self.solver = PETSc.KSP().create(domain.comm)
        self.solver.setOperators(A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)

    def step(self):
        with self.b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b, self.linear_form)
        fem.petsc.apply_lifting(self.b, [self.bilinear_form], [[]])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver.solve(self.b, self.new_phi.x.petsc_vec)
        self.new_phi.x.scatter_forward()
        clip_petsc_vec(self.new_phi.x.petsc_vec)
        self.phi.x.array[:] = self.new_phi.x.array


def clip_petsc_vec(vec, min_val=0.0, max_val=1.0):
    """限制 PETSc 向量的值范围"""
    arr = vec.getArray()
    arr[arr > max_val] = max_val
    arr[arr < min_val] = min_val
    vec.setArray(arr)
    vec.ghostUpdate()


def eliminate_small_values(function: fem.Function, threshold_low=0.05, threshold_high=0.95):
    array = function.x.array
    array[array < threshold_low] = 0
    array[array > threshold_high] = 1


def compute_form(domain, form: ufl.Form):
    integrate_form = fem.assemble_scalar(fem.form(form))
    integrate_form = domain.comm.allreduce(integrate_form, op=MPI.SUM)
    return integrate_form


def compute_volume(domain, phi):
    volume = compute_form(domain, phi * ufl.dx)
    return volume


def compute_A_iw_form(phi):
    A_iw_form = ksi * ufl.dot(ufl.grad(phi), ufl.grad(phi)) * ufl.dx
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
    surface_energy_is_form = gamma_is_ws * A_is_form
    return surface_energy_is_form


def compute_double_well_energy_bulk_form(phi, gamma_iw, gamma_is_ws):
    double_well_energy_bulk_form = 4 * gamma_iw / ksi * phi ** 2 * (1 - phi) ** 2 * ufl.dx
    return double_well_energy_bulk_form


def compute_double_well_energy_surface_form(phi, gamma_iw, gamma_is_ws):
    double_well_energy_surface_form = 2 * gamma_iw * phi ** 2 * (1 - phi) ** 2 * ufl.ds
    return double_well_energy_surface_form


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


def compute_interface_area(nodes, faces):
    triangles = nodes[faces]

    ab = triangles[:, 1] - triangles[:, 0]
    ac = triangles[:, 2] - triangles[:, 0]
    cross_products = np.cross(ab, ac)

    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    total_area = np.sum(areas)
    return total_area


def compute_A_iw_postprocessing(nodes, faces, interface_type):
    A_iw = compute_interface_area(nodes, faces)
    return A_iw


def compute_A_is_postprocessing(domain, phi):
    H = ufl.conditional(ufl.gt(phi, 0.5), 1.0, 0.0)
    A_is_form = H * ufl.ds
    A_is = compute_form(domain, A_is_form)
    return A_is


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


def create_lambda_star_func_with_steps(lambda_0, lambda_star, ramp_rate, t_eql_init,
                                       lambda_step, t_eql_ramp, t_eql_prd):
    # Create segments for the ramp with intermediate equilibration periods
    t_segments = [0, t_eql_init]  # Start with initial equilibration
    lambda_segments = [lambda_0, lambda_0]

    current_lambda = lambda_0
    remaining_ramp = abs(lambda_star - lambda_0)
    direction = 1 if lambda_star > lambda_0 else -1

    while remaining_ramp > 0:
        # Determine the next step
        step = min(lambda_step, remaining_ramp)

        # Add ramp segment
        t_ramp_segment = step / ramp_rate
        t_segments.append(t_segments[-1] + t_ramp_segment)
        current_lambda += direction * step
        lambda_segments.append(current_lambda)

        # Add equilibration segment if not yet at target
        if not np.isclose(current_lambda, lambda_star):
            t_segments.append(t_segments[-1] + t_eql_ramp)
            lambda_segments.append(current_lambda)

        remaining_ramp -= step

    # Add final equilibration period
    t_segments.append(t_segments[-1] + t_eql_prd)
    lambda_segments.append(lambda_star)

    lambda_star_func = create_lambda_star_func(t_segments, lambda_segments)
    t_total = t_segments[-1]

    return lambda_star_func, t_total


def main_PFM(system, theta: str, path_mesh, save_dir, t_eql_init, ramp_rate, lambda_step, t_eql_ramp, t_eql_prd,
             r_initial, lambda_0, lambda_star, x_offset, if_continue=False):
    logger = setup_logger(save_dir)
    # logger.info = print

    trajectory_save_dir = save_dir / "trajectory"
    trajectory_save_dir.mkdir(exist_ok=True)
    phi_checkpoint_save_path = trajectory_save_dir / "phi_checkpoint.bp"
    phi_xdmf_save_path = trajectory_save_dir / "phi.xdmf"
    grad_phi_xdmf_save_path = trajectory_save_dir / "phi_grad.xdmf"
    intermediate_result_save_path = save_dir / "intermediate_result.csv"
    instantaneous_interface_save_path = save_dir / "instantaneous_interface.pickle"

    # 梯度下降参数
    learning_rate_phi = 0.01
    learning_rate_l = 1e-7
    tolerance = 1e-10

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
    # return
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
    gaussian_smoother = GaussianSmoother(phi, ksi / 4, V, domain_water)

    if if_continue:
        with open(instantaneous_interface_save_path, "rb") as file:
            instantaneous_interface_dict = pickle.load(file)
        df_intermediate_result = pd.read_csv(intermediate_result_save_path, index_col=0)
        l.x.array[:] = df_intermediate_result.iloc[-1]["l"]
        last_t = df_intermediate_result.iloc[-1]["t"]
        adios4dolfinx.read_function(phi_checkpoint_save_path, phi)
        intermediate_result_list = df_intermediate_result.to_dict("records")
    else:
        x = ufl.SpatialCoordinate(domain_water)
        cos_theta = np.cos(np.radians(float(theta)))
        if lambda_0 is None:  # Use r_initial
            r_squared = (ufl.sqrt((x[0] - x_offset) ** 2 + (x[1] - 0) ** 2) - 0) ** 2 + (
                    x[2] - (1 - r_initial * cos_theta)) ** 2
            condition = ufl.conditional(r_squared <= r_initial ** 2, 1.0, 0.0)
            phi.interpolate(fem.Expression(condition, V.element.interpolation_points()))
            gaussian_smoother.step()
        else:  # Find r_initial so that lambda is close to lambda_0
            current_r = r_initial
            r_step = 2
            r_squared = (ufl.sqrt((x[0] - x_offset) ** 2 + (x[1] - 0) ** 2) - 0) ** 2 + (
                    x[2] - (1 - current_r * cos_theta)) ** 2
            condition = ufl.conditional(r_squared <= current_r ** 2, 1.0, 0.0)
            phi.interpolate(fem.Expression(condition, V.element.interpolation_points()))
            current_lambda = compute_volume(domain_water, phi) * rho
            if abs(current_lambda - lambda_0) > 100:
                direction = 1 if current_lambda < lambda_0 else -1
                for i in range(100):  # Maximum 100 loops
                    current_r = current_r + direction * r_step
                    r_squared = (ufl.sqrt((x[0] - x_offset) ** 2 + (x[1] - 0) ** 2) - 0) ** 2 + (
                            x[2] - (1 - current_r * cos_theta)) ** 2
                    condition = ufl.conditional(r_squared <= current_r ** 2, 1.0, 0.0)
                    phi.interpolate(fem.Expression(condition, V.element.interpolation_points()))
                    current_lambda = compute_volume(domain_water, phi) * rho
                    if abs(current_lambda - lambda_0) < 100:
                        logger.info(f"Using r_initial = {current_r}")
                        break
                    new_direction = 1 if current_lambda < lambda_0 else -1
                    if new_direction != direction:
                        r_step = r_step / 2
                    direction = new_direction
                else:
                    logger.error("Cannot find r_initial to get lambda_0")
                    raise RuntimeError("Cannot find r_initial to get lambda_0")
        instantaneous_interface_dict = {}
        last_t = -1
        intermediate_result_list = []
        with (XDMFFile(MPI.COMM_WORLD, phi_xdmf_save_path, "w") as phi_xdmf,
              XDMFFile(MPI.COMM_WORLD, grad_phi_xdmf_save_path, "w") as grad_phi_xdmf):
            phi_xdmf.write_mesh(domain_water)
            grad_phi_xdmf.write_mesh(domain_water)

    if lambda_0 is None:
        lambda_0 = compute_volume(domain_water, phi) * rho
    current_lambda_star.x.array[0] = lambda_0

    if lambda_star is None:
        lambda_star = lambda_0
    lambda_star_func, t_total = create_lambda_star_func_with_steps(lambda_0, lambda_star, ramp_rate, t_eql_init,
                                                                   lambda_step, t_eql_ramp, t_eql_prd)
    logger.info(f"lambda: {lambda_star_func(0):.0f} -> {lambda_star_func(t_total - 0.1):.0f}, t_total: {t_total:.0f}")
    # input("Press Enter to continue.")

    dphi = ufl.TestFunction(V)
    dl = ufl.TestFunction(R)
    energy_scale = const.N_A / 1000  # to kJ/mol
    E = energy_scale * (compute_surface_energy_iw_form(phi, gamma_iw)
                        + compute_surface_energy_is_form(phi, gamma_is_ws)
                        + compute_double_well_energy_bulk_form(phi, gamma_iw, gamma_is_ws)
                        + compute_double_well_energy_surface_form(phi, gamma_iw, gamma_is_ws))
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

    t_start = last_t + 1
    with (XDMFFile(MPI.COMM_WORLD, phi_xdmf_save_path, "a") as phi_xdmf,
          XDMFFile(MPI.COMM_WORLD, grad_phi_xdmf_save_path, "a") as grad_phi_xdmf):
        for t in range(t_start, t_start + int(t_total) + 1):
            lambda_star = lambda_star_func(t)
            current_lambda_star.x.array[0] = lambda_star
            # 组装梯度
            with gradient_phi_vec.localForm() as loc_g:
                loc_g.set(0.0)
            fem.petsc.assemble_vector(gradient_phi_vec, dphi_form)
            gradient_phi_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            grad_phi_magnitude.interpolate(grad_phi_magnitude_expr)
            with gradient_phi_vec.localForm() as loc_g, grad_phi_magnitude.x.petsc_vec.localForm() as loc_grad_mag:
                # 避免除以零，加一个小常数
                scaling_factor = loc_grad_mag.array / (loc_grad_mag.array.max() + 1e-10)
                scaling_factor[scaling_factor >= 0.2] = 1
                loc_g.array *= scaling_factor

            with gradient_l_vec.localForm() as loc_g:
                loc_g.set(0.0)
            fem.petsc.assemble_vector(gradient_l_vec, dl_form)
            gradient_l_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            phi.x.petsc_vec.axpy(-learning_rate_phi, gradient_phi_vec)
            l.x.petsc_vec.axpy(learning_rate_l, gradient_l_vec)
            clip_petsc_vec(phi.x.petsc_vec, 0, 1)
            for bc in bcs:
                bc.set(phi.x.array, alpha=1.0)

            if t % 100 == 0:
                gaussian_smoother.step()
                eliminate_small_values(phi)

                current_lambda = compute_volume(domain_water, phi) * rho
                logger.info(
                    f"Iteration {t}/{t_start + t_total:.0f}, lambda: {current_lambda:.0f}, lambda_star: {lambda_star:.0f}")
                surface_energy_iw = compute_form(domain_water, compute_surface_energy_iw_form(phi, gamma_iw))
                surface_energy_is_ws = compute_form(domain_water, compute_surface_energy_is_form(phi, gamma_is_ws))
                double_well_energy_bulk = compute_form(domain_water,
                                                       compute_double_well_energy_bulk_form(phi, gamma_iw, gamma_is_ws))
                double_well_energy_surface = compute_form(domain_water,
                                                          compute_double_well_energy_surface_form(phi, gamma_iw,
                                                                                                  gamma_is_ws))
                total_energy = surface_energy_iw + surface_energy_is_ws + double_well_energy_bulk + double_well_energy_surface
                nodes, faces, interface_type = generate_interface(V, phi)
                A_iw = compute_A_iw_postprocessing(nodes, faces, interface_type)
                A_is = compute_A_is_postprocessing(domain_water, phi)
                instantaneous_interface_dict[f"{t}"] = [nodes, faces, interface_type]
                logger.info(
                    f"Surface Energy iw: {energy_scale * surface_energy_iw:.2f}, Surface Energy is: {energy_scale * surface_energy_is_ws:.2f}, Double Well Energy (Bulk, Surface): ({energy_scale * double_well_energy_bulk:.2f}, {energy_scale * double_well_energy_surface:.2f})")
                info = {
                    "t": t,
                    "lambda": current_lambda,
                    "lambda_star": lambda_star,
                    "A_iw": A_iw,
                    "A_is": A_is,
                    "total_energy": total_energy,
                    "surface_energy_iw": surface_energy_iw,
                    "surface_energy_is_ws": surface_energy_is_ws,
                    "double_well_energy_bulk": double_well_energy_bulk,
                    "double_well_energy_surface": double_well_energy_surface,
                    "bulk_energy": 0.0,
                    "l": l.x.array[0],
                }
                # print(l.x.array, gradient_l.x.array)
                # print(phi.x.array.min(), phi.x.array.max())
                intermediate_result_list.append(info)

                phi_xdmf.write_function(phi, t=t)
                grad_phi_xdmf.write_function(grad_phi_magnitude, t=t)

    adios4dolfinx.write_function(phi_checkpoint_save_path, phi)

    df = pd.DataFrame(intermediate_result_list)
    df.to_csv(intermediate_result_save_path)
    with open(instantaneous_interface_save_path, "wb") as file:
        pickle.dump(instantaneous_interface_dict, file)


def main():
    parser = argparse.ArgumentParser(description="Control the program functions.")
    parser.add_argument("--system", default="flat", choices=["flat", "pillar"])
    parser.add_argument("--theta", default="90")
    parser.add_argument("--job_name", default="test")
    parser.add_argument("--t_eql_init", default=10000, type=float)
    parser.add_argument("--ramp_rate", default=0.25, type=float)
    parser.add_argument("--lambda_step", default=100, type=float)
    parser.add_argument("--t_eql_ramp", default=0, type=float)
    parser.add_argument("--t_eql_prd", default=5000, type=float)
    parser.add_argument("--r_initial", default=40, type=float)
    parser.add_argument("--lambda_0", default=None, type=float)
    parser.add_argument("--lambda_star", default=None, type=float)
    parser.add_argument("--x_offset", default=0, type=float)
    parser.add_argument("--r_pillar", default=10,
                        help="Pillar radius (required when system is 'pillar')")
    parser.add_argument("--asymmetric", action="store_true")
    args = parser.parse_args()

    system = args.system
    theta = args.theta
    job_name = args.job_name
    t_eql_init = args.t_eql_init
    ramp_rate = args.ramp_rate
    lambda_step = args.lambda_step
    t_eql_ramp = args.t_eql_ramp
    t_eql_prd = args.t_eql_prd
    r_initial = args.r_initial
    lambda_0 = args.lambda_0
    lambda_star = args.lambda_star
    x_offset = args.x_offset

    root_dir = Path("../FEM")
    root_dir.mkdir(exist_ok=True)
    if system == "pillar":
        r_pillar = args.r_pillar
        mesh_filename = f"{theta}_asym.msh" if args.asymmetric else f"{theta}.msh"
        path_mesh = Path(f"../mesh/pillar_system/{r_pillar}/{mesh_filename}")
        save_dir = root_dir / "pillar" / f"{theta}" / job_name
    elif system == "flat":
        path_mesh = Path(f"../mesh/flat_surface/{theta}.msh")
        save_dir = root_dir / "flat" / f"{theta}" / job_name
    else:
        raise ValueError(f"System {system} not supported")
    save_dir.mkdir(exist_ok=True, parents=True)
    main_PFM(system, theta, path_mesh, save_dir, t_eql_init, ramp_rate, lambda_step, t_eql_ramp,
             t_eql_prd, r_initial, lambda_0, lambda_star, x_offset, if_continue=False)


if __name__ == "__main__":
    main()
