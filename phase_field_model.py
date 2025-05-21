# Author: Mian Qin
# Date Created: 2025/3/27
import os
from pathlib import Path

import numpy as np
import scipy.constants as const
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
from MDAnalysis.units import water

from utils_torch import *

run_env = os.environ.get("RUN_ENVIRONMENT")
if run_env == "mianqin_Mac":
    device = "mps"
elif run_env == "mianqin_PC":
    device = "cuda"
else:
    raise RuntimeError(f"Unknown environment: {run_env}")

rho = 5.09e4 * 1e-30  # mol/A^3
delta_mu = 0  # J/mol
gamma_iw = 25.9e-3 * 1e-20  # J/A^2, real water

ksi = 0.25  # unit: A
dx = 0.5  # unit: A
K = 1 * gamma_iw * ksi  # J/A
h0 = gamma_iw / (3 * ksi)
dV = dx ** 3  # unit A^3
dA = dx ** 2  # unit A^2


class PhaseFieldModel:
    def __init__(self, theta, phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask, optimizer=torch.optim.Adam, lr=1e-3):
        self.k = np.cos(np.radians(theta))
        self.gamma_is_ws = -gamma_iw * self.k
        self.phi = nn.Parameter(phi.to(device), requires_grad=True)
        self.phi_surface_surface = torch.zeros_like(self.phi, requires_grad=False).to(device)
        self.surface_mask = surface_mask.to(device)
        self.water_mask = water_mask.to(device)
        self.surface_surface_mask = surface_surface_mask.to(device)
        self.water_surface_mask = water_surface_mask.to(device)
        self.optimizer = optimizer([self.phi], lr=lr)

    def run(self, lambda_star_func, t_total, save_dir: Path, logger):
        n_digits = len(str(t_total))
        save_dir.mkdir(exist_ok=True)
        trajectory_dir = save_dir / "trajectory"
        trajectory_dir.mkdir(exist_ok=True)
        loss_list = []
        instantaneous_interface_dict = {}
        grad = None
        grad_max = None
        for t in range(t_total+1):
            if t % 10 == 0:
                grad = None
                grad_max = None
                self.phi_surface_surface = compute_phi_surface_surface(self.phi, self.surface_mask, self.water_mask, self.surface_surface_mask, self.water_surface_mask)

            lambda_star = float(lambda_star_func(t))
            loss, grad, grad_max = update_phi(self.phi, self.phi_surface_surface, self.optimizer, self.surface_mask, self.water_mask, self.surface_surface_mask, self.water_surface_mask, lambda_star, self.gamma_is_ws, grad, grad_max)
            loss_list.append(loss)

            if t % 100 == 0:
                with torch.no_grad():
                    lambda_ = compute_lambda(self.phi, self.water_mask)
                    phi_cpu = self.phi.detach().cpu().numpy()
                    # grad_term = compute_corrected_gradient_squared(self.phi, self.water_mask)
                    grad_term = grad
                    grad_term_cpu = grad_term.detach().cpu().numpy()
                info_message = f"t: {t:{n_digits}d}/{t_total}, loss: {loss:.2f}, lambda: {lambda_:.1f}, lambda_star: {lambda_star:.1f}"
                logger.info(info_message)

                if device == "cuda":
                    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
                    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2

                    debug_message = f"Memory allocated: {memory_allocated:.2f} MB, Memory reserved: {memory_reserved:.2f} MB, Max memory allocated: {max_memory_allocated:.2f} MB"
                    logger.debug(debug_message)

                hist, bin_edges = np.histogram(phi_cpu, bins=10, range=(0, 1))
                debug_message = "Histogram of phi:"
                for i in range(len(hist)):
                    debug_message += f"\n\t{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i] * dV * rho * const.N_A:.1f}"
                logger.debug(debug_message)

                hist, bin_edges = np.histogram(grad_term_cpu, bins=10)
                debug_message = "Histogram of grad phi:"
                N_total = hist.sum()
                for i in range(len(hist)):
                    debug_message += f"\n\t{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}: {hist[i] / N_total * 100:.2f}%"
                logger.debug(debug_message)

                export_phi(self.phi * self.water_mask, trajectory_dir, f"{t:06}.npy")
                nodes, faces, interface_type = generate_interface(self.phi * self.water_mask, dx)
                instantaneous_interface_dict[f"{t}"] = [nodes, faces, interface_type]
            # if t % 1000 == 0:
            #     export_phi(self.phi * self.water_mask, trajectory_dir, f"{t:06}.npy")
        # phi = phi * water_mask
        with open(save_dir / "instantaneous_interface.pickle", "wb") as file:
            pickle.dump(instantaneous_interface_dict, file)
        export_phi(self.phi * self.water_mask, save_dir)
        export_interface(self.phi * self.water_mask, dx, save_dir)
        export_surface(self.surface_mask, dx, save_dir)


def compute_bulk_energy(phi: torch.Tensor, water_mask: torch.Tensor):
    V_ice = torch.sum(phi * water_mask) * dV
    bulk_energy = V_ice * rho * delta_mu
    return bulk_energy


_kernel_dict = {}


def compute_gradient(phi: torch.Tensor, direction="xyz"):
    key = (phi.dtype, phi.device)

    if key not in _kernel_dict:
        # fourth order
        kernel_1d = torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=phi.dtype, device=phi.device)
        kernels = {"x": kernel_1d.view(1, 1, 5, 1, 1),
                  "y": kernel_1d.view(1, 1, 1, 5, 1),
                  "z": kernel_1d.view(1, 1, 1, 1, 5)}
        _kernel_dict[key] = kernels

    kernels = _kernel_dict[key]

    grad_component_4th_order = []
    if "x" in direction:
        kernel_x = kernels["x"]
        padded_phi_x = F.pad(phi, pad=(0, 0, 0, 0, 2, 2), mode='circular')
        grad_x = F.conv3d(padded_phi_x, kernel_x) / dx
        grad_component_4th_order.append(grad_x)
    if "y" in direction:
        kernel_y = kernels["y"]
        padded_phi_y = F.pad(phi, pad=(0, 0, 2, 2, 0, 0), mode='circular')
        grad_y = F.conv3d(padded_phi_y, kernel_y) / dx
        grad_component_4th_order.append(grad_y)
    if "z" in direction:
        kernel_z = kernels["z"]
        padded_phi_z = F.pad(phi, pad=(2, 2, 0, 0, 0, 0), mode='circular')
        grad_z = F.conv3d(padded_phi_z, kernel_z) / dx
        grad_component_4th_order.append(grad_z)

    if len(direction) == 1:
        return grad_component_4th_order[0]
    else:
        return grad_component_4th_order


def compute_gradient_squared(phi: torch.Tensor):
    grad_components = compute_gradient(phi)
    grad_components_tensor = torch.stack([g for g in grad_components]).view(3, *phi.shape)
    grad_squared = (grad_components_tensor ** 2).sum(dim=0)
    return grad_squared


def compute_corrected_gradient_squared(phi: torch.Tensor, water_mask: torch.Tensor):
    grad_squared = compute_gradient_squared(phi)
    bulk_mask = water_mask
    grad_squared_bulk = grad_squared * bulk_mask

    # padded_grad_squared_bulk = F.pad(grad_squared_bulk, pad=(1, 1, 1, 1, 1, 1), mode='circular')
    # expanded_grad_squared_bulk = F.max_pool3d(padded_grad_squared_bulk, 3, stride=1)
    # corrected_gradient_squared = expanded_grad_squared_bulk * water_surface_mask1 + grad_squared_bulk
    corrected_gradient_squared = grad_squared_bulk
    return corrected_gradient_squared


def compute_surface_energy_iw(phi: torch.Tensor, water_mask: torch.Tensor, water_surface_mask: torch.Tensor):
    grad_term = compute_corrected_gradient_squared(phi, water_mask)

    # bulk_mask = water_mask - water_surface_mask
    bulk_mask = water_mask
    surface_energy_iw = torch.sum((K * grad_term) * bulk_mask)
    return surface_energy_iw, grad_term.detach()


def compute_A_is(phi: torch.Tensor, water_surface_mask: torch.Tensor):
    A_is = torch.sum(phi * water_surface_mask) * dA / 2  # 2 layer so divided by 2
    return A_is


def compute_surface_energy_is(phi: torch.Tensor, water_surface_mask: torch.Tensor, gamma_is_ws):
    A_is = compute_A_is(phi, water_surface_mask)
    surface_energy_is = A_is * gamma_is_ws
    return surface_energy_is


def compute_double_well_energy(phi: torch.Tensor, water_mask: torch.Tensor):
    # phi = gaussian_smooth(phi, kernel_size=3)
    # f_phi = h0 * (phi ** 2 * (1 - phi) ** 2) * water_mask
    f_phi = 0.25 * h0 * phi * torch.abs(1 - phi)
    double_well_energy = torch.sum(f_phi)
    return double_well_energy


def compute_total_energy(phi: torch.Tensor, surface_mask: torch.Tensor, water_mask: torch.Tensor, water_surface_mask: torch.Tensor, gamma_is_ws):
    surface_energy_scale = 3.0
    bulk_energy = compute_bulk_energy(phi, water_mask)
    surface_energy_iw, grad_term = compute_surface_energy_iw(phi, water_mask, water_surface_mask)
    surface_energy_is = compute_surface_energy_is(phi, water_surface_mask, gamma_is_ws)
    double_well_energy = compute_double_well_energy(phi, water_mask)
    total_energy = bulk_energy + (surface_energy_iw + surface_energy_is) * surface_energy_scale + double_well_energy
    info = {
        "bulk_energy": bulk_energy.item(),
        "surface_energy_iw": surface_energy_iw.item(),
        "surface_energy_is": surface_energy_is.item(),
        "double_well_energy": double_well_energy.item(),
        "total_energy": total_energy.item()
    }
    return total_energy, grad_term, info


def compute_lambda(phi: torch.Tensor, water_mask: torch.Tensor):
    lambda_ = torch.sum(phi * water_mask) * dV * rho * const.N_A
    return lambda_


def compute_bias_potential(phi, water_mask, lambda_star):
    kappa = 200 / const.N_A
    # kappa = 0

    lambda_ = compute_lambda(phi, water_mask)
    # print(lambda_)
    bias_potential = kappa / 2 * (lambda_ - lambda_star) ** 2
    return bias_potential


def compute_mean_curvature(phi: torch.Tensor, phi_surface_surface: torch.Tensor, water_mask: torch.Tensor, bulk_mask: torch.Tensor, epsilon=1e-10):
    phi = gaussian_smooth(phi + phi_surface_surface) * water_mask
    grad_x, grad_y, grad_z = compute_gradient(phi + phi_surface_surface)
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    grad_xx, grad_xy, grad_xz = compute_gradient(grad_x)
    grad_yy, grad_yz = compute_gradient(grad_y, direction="yz")
    grad_zz = compute_gradient(grad_z, direction="z")

    numerator = (
        grad_x ** 2 * (grad_yy + grad_zz) +
        grad_y ** 2 * (grad_xx + grad_zz) +
        grad_z ** 2 * (grad_xx + grad_yy) -
        2 * grad_x * grad_y * grad_xy -
        2 * grad_x * grad_z * grad_xz -
        2 * grad_y * grad_z * grad_yz
    )
    denominator = 2 * (grad_norm ** 3 + epsilon)

    interface_mask = grad_norm > 0.12
    H = (numerator / denominator) * bulk_mask * interface_mask
    valid_H = H[interface_mask]
    valid_H_1percentile = torch.quantile(valid_H, 0.01)
    valid_H_99percentile = torch.quantile(valid_H, 0.99)
    filtered_H = valid_H[(valid_H > valid_H_1percentile) & (valid_H < valid_H_99percentile)]

    H_mean = torch.mean(filtered_H)
    H_std = torch.std(filtered_H)

    # phi_cpu = phi.cpu()
    # total_phi_cpu = (phi + phi_surface_surface).cpu()
    # grad_norm_cpu = grad_norm.cpu()
    # H_cpu = H.cpu()
    # filtered_H_cpu = filtered_H.cpu()
    return H, H_mean, H_std


@torch.no_grad()
def compute_phi_surface_surface(phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask):
    padded_phi = F.pad(phi, (2, 2, 2, 2, 2, 2), 'circular')
    padded_water_mask = F.pad(water_mask, (2, 2, 2, 2, 2, 2), 'circular')
    kernel = torch.ones((1, 1, 5, 5, 5), dtype=phi.dtype, device=phi.device)
    N_neighbor = torch.clamp(F.conv3d(padded_water_mask, kernel) * surface_surface_mask, min=1)
    phi_neighbor = F.conv3d(padded_phi, kernel) * surface_surface_mask
    phi_surface_surface = phi_neighbor / N_neighbor
    return phi_surface_surface


def update_phi(phi: torch.Tensor, phi_surface_surface: torch.Tensor, optimizer, surface_mask, water_mask, surface_surface_mask, water_surface_mask, lambda_star, gamma_is_ws, grad=None, grad_max=None):
    bias_potential = compute_bias_potential(phi + phi_surface_surface, water_mask, lambda_star)
    total_energy, grad_term, info = compute_total_energy(phi + phi_surface_surface, surface_mask, water_mask, water_surface_mask, gamma_is_ws)

    loss = (bias_potential + total_energy) * 1e20

    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        if grad is None:
        # grad = compute_corrected_gradient_squared(phi, water_mask)
            grad = gaussian_smooth(grad_term, sigma=5)
            grad_max = grad.max()
        grad_scale = grad / max(grad_max, 1e-5)
        # grad_scale = grad
        grad_scale[grad_scale > 0.2] = 1
        phi.grad.data.mul_(grad_scale * water_mask)
        # print(phi.grad.data.max())
    nn.utils.clip_grad_value_([phi], 0.25)
    optimizer.step()
    with torch.no_grad():
        phi.data.clamp_(0, 1)
    return loss.item(), grad, grad_max


def main():
    ...


if __name__ == "__main__":
    main()
