# Author: Mian Qin
# Date Created: 2025/3/27
import numpy as np
import scipy.constants as const
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt

from utils import *

rho = 5e4 * 1e-30  # mol/A^3
delta_mu = 0  # J/mol
gamma_iw = 30.8e-3 * 1e-20  # J/A^2
k = 0.568
gamma_is_ws = -gamma_iw * k

device = "cuda"
L = np.array([50, 50, 50])  # unit: A
ksi = 0.25  # unit: A
dx = 0.5  # unit: A
K = 3 * gamma_iw * ksi  # J/A
h0 = gamma_iw / (3 * ksi)
dV = dx ** 3  # unit A^3
dA = dx ** 2  # unit A^2


N = np.floor(L / dx).astype(int)


def create_surface():
    surface_mask = torch.zeros((N[0], N[1], N[2]), device=device)
    # surface_mask[:, :, :20] = 1
    surface_mask = surface_mask.view(1, 1, N[0], N[1], N[2])
    water_mask = 1 - surface_mask

    kernel = torch.zeros((3, 3, 3), device=device)
    kernel[1, 1, :] = 1
    kernel[1, :, 1] = 1
    kernel[:, 1, 1] = 1
    kernel = kernel.view(1, 1, 3, 3, 3)

    expanded_surface_mask = torch.clamp(F.conv3d(surface_mask, kernel, stride=[1, 1, 1], padding="same"), 0, 1)
    water_surface_mask = water_mask * expanded_surface_mask
    return surface_mask, water_mask, water_surface_mask


def calculate_bulk_energy(phi: torch.tensor, water_mask: torch.tensor):
    V_ice = torch.sum(phi * water_mask) * dV
    bulk_energy = V_ice * rho * delta_mu
    return bulk_energy


def calculate_surface_energy_iw(phi: torch.tensor, water_mask: torch.tensor):
    padded_phi = F.pad(phi, pad=(2, 2, 2, 2, 2, 2), mode='circular')
    grad_padded_phi = torch.gradient(padded_phi, spacing=dx, dim=(2, 3, 4))
    grad_component = [g[..., 2:-2, 2:-2, 2:-2] for g in grad_padded_phi]
    grad_term = sum(g ** 2 for g in grad_component)
    grad = grad_term.detach().cpu().numpy()

    surface_energy_iw = torch.sum((K * grad_term) * water_mask)
    return surface_energy_iw


def calculate_surface_energy_is(phi: torch.tensor, water_surface_mask: torch.tensor):
    A_is = torch.sum(phi * water_surface_mask) * dA
    surface_energy_is = A_is * gamma_is_ws
    return surface_energy_is


def calculate_double_well_energy(phi: torch.tensor, water_mask: torch.tensor):
    f_phi = h0 * (phi ** 2 * (1 - phi) ** 2) * water_mask
    double_well_energy = torch.sum(f_phi)
    return double_well_energy


def calculate_total_energy(phi: torch.Tensor, surface_mask: torch.Tensor, water_mask: torch.Tensor, water_surface_mask: torch.Tensor):
    bulk_energy = calculate_bulk_energy(phi, water_mask)
    surface_energy_iw = calculate_surface_energy_iw(phi, water_mask)
    surface_energy_is = calculate_surface_energy_is(phi, water_surface_mask)
    double_well_energy = calculate_double_well_energy(phi, water_mask)
    total_energy = bulk_energy + surface_energy_iw + surface_energy_is + double_well_energy
    return total_energy


def calculate_lambda(phi: torch.Tensor, water_mask: torch.Tensor):
    lambda_ = torch.sum(phi * water_mask) * dV * rho * const.N_A
    return lambda_


def calculate_bias_potential(phi, water_mask, lambda_star):
    kappa = 200 / const.N_A

    lambda_ = calculate_lambda(phi, water_mask)
    # print(lambda_)
    bias_potential = kappa / 2 * (lambda_ - lambda_star) ** 2
    return bias_potential


def update_phi(phi, optimizer, surface_mask, water_mask, water_surface_mask, lambda_star):
    bias_potential = calculate_bias_potential(phi, water_mask, lambda_star)
    total_energy = calculate_total_energy(phi, surface_mask, water_mask, water_surface_mask)

    loss = (bias_potential + total_energy) * 1e20

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_([phi], 0.1)
    optimizer.step()
    with torch.no_grad():
        phi.data.clamp_(0, 1)
    return loss.item()


def main():
    lambda_star = 200
    t_ramp = 200

    surface_mask, water_mask, water_surface_mask = create_surface()
    radius = 10
    x = torch.arange(N[0], dtype=torch.float, device=device) * dx
    y = torch.arange(N[1], dtype=torch.float, device=device) * dx
    z = torch.arange(N[2], dtype=torch.float, device=device) * dx

    # 计算中心坐标
    center = [25, 25, 25]

    # 计算距离中心的距离（网格化计算）
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    # distance_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    distance_sq = torch.max(torch.stack((torch.abs(xx - center[0]), torch.abs(yy - center[1]), torch.abs(zz - center[2])), dim=0), dim=0)[0]

    # 将距离 ≤10埃的点设为1
    phi = (distance_sq <= radius).float().unsqueeze(0).unsqueeze(0)
    noise = torch.randn_like(phi) * 0.1
    phi = torch.clamp(phi + noise, 0, 1)
    phi = nn.Parameter(phi, requires_grad=True)
    optimizer = torch.optim.SGD([phi], lr=1)

    with torch.no_grad():
        lambda_0 = calculate_lambda(phi, water_mask)

    losses = []
    for i in range(500):
        if i < 200:
            lambda_ramp = (lambda_star - lambda_0) * i / 200 + lambda_0
        else:
            lambda_ramp = lambda_star
        loss = update_phi(phi, optimizer, surface_mask, water_mask, water_surface_mask, lambda_ramp)
        losses.append(loss)
        if i % 10 == 0:
            with torch.no_grad():
                lambda_ = calculate_lambda(phi, water_mask)
            print(f"Iteration: {i}, loss: {loss:.2f}, mean: {torch.mean(phi):.4f}, lambda: {lambda_:.1f}, phi_min: {torch.min(phi):.2f}, phi_max: {torch.max(phi):.2f}")
    export_interface(phi.view(N[0], N[1], N[2]), dx, "homogeneous")


if __name__ == "__main__":
    main()
