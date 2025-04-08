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

from utils import *

run_env = os.environ.get("RUN_ENVIRONMENT")
if run_env == "mianqin_Mac":
    device = "mps"
elif run_env == "mianqin_PC":
    device = "cuda"
else:
    raise RuntimeError(f"Unknown environment: {run_env}")

rho = 5e4 * 1e-30  # mol/A^3
delta_mu = -400  # J/mol
gamma_iw = 30.8e-3 * 1e-20  # J/A^2

ksi = 0.25  # unit: A
dx = 0.5  # unit: A
K = 3 * gamma_iw * ksi  # J/A
h0 = gamma_iw / (3 * ksi)
dV = dx ** 3  # unit A^3
dA = dx ** 2  # unit A^2


class PhaseFieldModel:
    def __init__(self, theta, phi, surface_mask, water_mask, water_surface_mask, optimizer=torch.optim.SGD, lr=0.1):
        self.k = np.cos(theta)
        self.gamma_is_ws = -gamma_iw * self.k
        self.phi = nn.Parameter(phi.to(device), requires_grad=True)
        self.surface_mask = surface_mask.to(device)
        self.water_mask = water_mask.to(device)
        self.water_surface_mask = water_surface_mask.to(device)
        self.optimizer = optimizer([self.phi], lr=lr)

    def run(self, lambda_star_func, t_total, save_dir: Path, logger):
        loss_list = []
        instantaneous_interface_dict = {}
        grad = None
        for t in range(t_total+1):
            if t % 10 == 0:
                grad = None
            lambda_star = float(lambda_star_func(t))
            loss, grad = update_phi(self.phi, self.optimizer, self.surface_mask, self.water_mask, self.water_surface_mask, lambda_star, self.gamma_is_ws, grad)
            loss_list.append(loss)

            if t % 100 == 0:
                with torch.no_grad():
                    lambda_ = compute_lambda(self.phi, self.water_mask)
                    phi_cpu = self.phi.detach().cpu().numpy()
                    grad_term = compute_gradient_squared(self.phi)
                    grad_term_cpu = grad_term.detach().cpu().numpy()
                info_message = f"t: {t}, loss: {loss:.2f}, lambda: {lambda_:.1f}"
                logger.info(info_message)

                hist, bin_edges = np.histogram(phi_cpu, bins=10, range=(0, 1))
                debug_message = "Histogram of phi:"
                for i in range(len(hist)):
                    debug_message += f"\n\t{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {hist[i] * dV * rho * const.N_A:.1f}"
                logger.debug(debug_message)

                hist, bin_edges = np.histogram(grad_term_cpu, bins=10)
                debug_message = "Histogram of grad phi:"
                for i in range(len(hist)):
                    debug_message += f"\n\t{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}: {hist[i] * dV * rho * const.N_A:.1f}"
                logger.debug(debug_message)

                nodes, faces, interface_type = generate_interface(self.phi, dx)
                instantaneous_interface_dict[f"{t}"] = [nodes, faces, interface_type]
        # phi = phi * water_mask
        with open(save_dir / "instantaneous_interface.pickle", "wb") as file:
            pickle.dump(instantaneous_interface_dict, file)
        export_phi(self.phi, save_dir)
        export_interface(self.phi, dx, save_dir)
        export_surface(self.surface_mask, dx, save_dir)


def compute_bulk_energy(phi: torch.Tensor, water_mask: torch.Tensor):
    V_ice = torch.sum(phi * water_mask) * dV
    bulk_energy = V_ice * rho * delta_mu
    return bulk_energy


def compute_gradient(phi: torch.Tensor):
    # # central difference
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
    # return grad_component_central, grad_component_forward, grad_component_backward
    return (grad_component_4th_order, )


def compute_gradient_squared(phi: torch.Tensor):
    grad_components = compute_gradient(phi)
    grad_components_tensor = torch.stack([g for grad_component in grad_components for g in grad_component]).view(len(grad_components), 3, *phi.shape)
    grad_squared = (grad_components_tensor ** 2).sum(dim=1).mean(dim=0)
    return grad_squared


def compute_surface_energy_iw(phi: torch.Tensor, water_mask: torch.Tensor, water_surface_mask: torch.Tensor):
    grad_term = compute_gradient_squared(phi)

    bulk_mask = water_mask - water_surface_mask
    surface_energy_iw = torch.sum((K * grad_term) * bulk_mask)
    return surface_energy_iw


def compute_surface_energy_is(phi: torch.Tensor, water_surface_mask: torch.Tensor, gamma_is_ws):
    A_is = torch.sum(phi * water_surface_mask) * dA  # 2 layer so divided by 2
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
    surface_energy_iw = surface_energy_scale * compute_surface_energy_iw(phi, water_mask, water_surface_mask)
    surface_energy_is = surface_energy_scale * compute_surface_energy_is(phi, water_surface_mask, gamma_is_ws)
    double_well_energy = compute_double_well_energy(phi, water_mask)
    total_energy = bulk_energy + surface_energy_iw + surface_energy_is + double_well_energy
    return total_energy


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


def update_phi(phi, optimizer, surface_mask, water_mask, water_surface_mask, lambda_star, gamma_is_ws, grad=None):
    bias_potential = compute_bias_potential(phi, water_mask, lambda_star)
    total_energy = compute_total_energy(phi, surface_mask, water_mask, water_surface_mask, gamma_is_ws)

    loss = (bias_potential + total_energy) * 1e20

    optimizer.zero_grad()
    loss.backward()
    with torch.no_grad():
        if grad is None:
            grad = compute_gradient_squared(phi)
            bulk_mask = water_mask - water_surface_mask
            grad = grad * bulk_mask
            grad = gaussian_smooth(grad)
        # grad_90percentile = torch.quantile(grad, 0.95)  # remove the top 5% for robustness
        grad_scale = grad / max(grad.max(), 1e-5)
        phi.grad.data.mul_(grad_scale)
    nn.utils.clip_grad_value_([phi], 0.25)
    optimizer.step()
    with torch.no_grad():
        phi.data.clamp_(0, 1)
    return loss.item(), grad


def main():
    ...


if __name__ == "__main__":
    main()
