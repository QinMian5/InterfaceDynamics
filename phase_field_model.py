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

from utils import *

run_env = os.environ.get("RUN_ENVIRONMENT")
if run_env == "mianqin_Mac":
    device = "mps"
elif run_env == "mianqin_PC":
    device = "cuda"
else:
    raise RuntimeError(f"Unknown environment: {run_env}")

rho = 5e4 * 1e-30  # mol/A^3
delta_mu = 0  # J/mol
gamma_iw = 30.8e-3 * 1e-20  # J/A^2

ksi = 0.25  # unit: A
dx = 0.5  # unit: A
K = 3 * gamma_iw * ksi  # J/A
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
        for t in range(t_total+1):
            if t % 10 == 0:
                grad = None
                self.phi.data.copy_(update_surface(self.phi, self.surface_mask, self.water_mask, self.surface_surface_mask, self.water_surface_mask))
            lambda_star = float(lambda_star_func(t))
            loss, grad = update_phi(self.phi, self.optimizer, self.surface_mask, self.water_mask, self.surface_surface_mask, self.water_surface_mask, lambda_star, self.gamma_is_ws, grad)
            loss_list.append(loss)

            if t % 100 == 0:
                with torch.no_grad():
                    lambda_ = compute_lambda(self.phi, self.water_mask)
                    phi_cpu = self.phi.detach().cpu().numpy()
                    grad_term = compute_corrected_gradient_squared(self.phi, self.water_mask)
                    grad_term_cpu = grad_term.detach().cpu().numpy()
                info_message = f"t: {t:{n_digits}d}/{t_total}, loss: {loss:.2f}, lambda: {lambda_:.1f}, lambda_star: {lambda_star:.1f}"
                logger.info(info_message)

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


def compute_gradient(phi: torch.Tensor):
    # central difference
    # padded_phi = F.pad(phi, pad=(2, 2, 2, 2, 2, 2), mode='circular')
    # grad_padded_phi = torch.gradient(padded_phi, spacing=dx, edge_order=1, dim=(2, 3, 4))
    # grad_component_central = [g[..., 2:-2, 2:-2, 2:-2] for g in grad_padded_phi]
    #
    # # forward and backward difference
    # grad_component_forward = []
    # grad_component_backward = []
    # for dim in [2, 3, 4]:
    #     forward_phi = torch.roll(phi, shifts=-1, dims=dim)
    #     forward_grad = (forward_phi - phi) / dx
    #     grad_component_forward.append(forward_grad)
    #     backward_phi = torch.roll(phi, shifts=+1, dims=dim)
    #     backward_grad = (phi - backward_phi) / dx
    #     grad_component_backward.append(backward_grad)

    # fourth order
    kernel_1d = torch.tensor([1/12, -2/3, 0, 2/3, -1/12], dtype=phi.dtype, device=phi.device)

    kernel_x = kernel_1d.view(1, 1, 5, 1, 1)
    padded_phi_x = F.pad(phi, pad=(0, 0, 0, 0, 2, 2), mode='circular')
    grad_x = F.conv3d(padded_phi_x, kernel_x) / dx

    kernel_y = kernel_1d.view(1, 1, 1, 5, 1)
    padded_phi_y = F.pad(phi, pad=(0, 0, 2, 2, 0, 0), mode='circular')
    grad_y = F.conv3d(padded_phi_y, kernel_y) / dx

    kernel_z = kernel_1d.view(1, 1, 1, 1, 5)
    padded_phi_z = F.pad(phi, pad=(2, 2, 0, 0, 0, 0), mode='circular')
    grad_z = F.conv3d(padded_phi_z, kernel_z) / dx

    grad_component_4th_order = [grad_x, grad_y, grad_z]
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


def compute_surface_energy_iw(phi: torch.Tensor, water_mask: torch.Tensor):
    grad_term = compute_corrected_gradient_squared(phi, water_mask)

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
    surface_energy_iw, grad_term = compute_surface_energy_iw(phi, water_mask)
    surface_energy_is = compute_surface_energy_is(phi, water_surface_mask, gamma_is_ws)
    double_well_energy = compute_double_well_energy(phi, water_mask)
    total_energy = bulk_energy + (surface_energy_iw + surface_energy_is) * surface_energy_scale + double_well_energy
    return total_energy, grad_term


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


def compute_mean_curvature(phi: torch.Tensor, water_mask: torch.Tensor):
    ...


def update_surface(phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask):
    phi_water = phi * water_mask
    padded_phi = F.pad(phi_water, (2, 2, 2, 2, 2, 2), 'circular')
    padded_water_mask = F.pad(water_mask, (2, 2, 2, 2, 2, 2), 'circular')
    kernel = torch.ones((1, 1, 5, 5, 5), dtype=phi.dtype, device=phi.device)
    N_neighbor = torch.clamp(F.conv3d(padded_water_mask, kernel) * surface_surface_mask, min=1)
    phi_neighbor = F.conv3d(padded_phi, kernel) * surface_surface_mask
    phi_surface_surface = phi_neighbor / N_neighbor
    new_phi = phi_water + phi_surface_surface
    return new_phi


def update_phi(phi: torch.Tensor, optimizer, surface_mask, water_mask, surface_surface_mask, water_surface_mask, lambda_star, gamma_is_ws, grad=None):
    bias_potential = compute_bias_potential(phi, water_mask, lambda_star)
    total_energy, grad_term = compute_total_energy(phi, surface_mask, water_mask, water_surface_mask, gamma_is_ws)

    loss = (bias_potential + total_energy) * 1e20

    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        if grad is None:
        # grad = compute_corrected_gradient_squared(phi, water_mask)
            grad = gaussian_smooth(grad_term, sigma=5)
            # grad = maxpool_smooth(grad_term)
        # grad = grad_term
        grad_max = torch.quantile(grad, 0.995)
        grad_scale = grad / max(grad_max, 1e-5)
        grad_scale[grad_scale > 0.2] = 1
        phi.grad.data.mul_(grad_scale * water_mask)
        # print(phi.grad.data.max())
    nn.utils.clip_grad_value_([phi], 0.25)
    optimizer.step()
    with torch.no_grad():
        phi.data.clamp_(0, 1)
    return loss.item(), grad


def main():
    ...


if __name__ == "__main__":
    main()
