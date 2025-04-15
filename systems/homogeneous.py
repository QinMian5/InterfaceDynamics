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
from phase_field_model import PhaseFieldModel, dx, compute_lambda, compute_corrected_gradient_squared

L = np.array([70, 70, 70])  # unit: A
N = np.floor(L / dx).astype(int)


def create_surface():
    surface_mask = torch.zeros((N[0], N[1], N[2]))
    surface_mask = surface_mask.view(1, 1, N[0], N[1], N[2])
    water_mask = 1 - surface_mask

    kernel = torch.zeros((3, 3, 3))
    kernel[1, 1, :] = 1
    kernel[1, :, 1] = 1
    kernel[:, 1, 1] = 1
    kernel = kernel.view(1, 1, 3, 3, 3)

    expanded_surface = F.conv3d(surface_mask, kernel, stride=[1, 1, 1], padding="same")
    # expanded_surface_2 = F.conv3d(expanded_surface, kernel, stride=[1, 1, 1], padding="same")
    expanded_surface_mask = torch.clamp(expanded_surface, 0, 1)
    water_surface_mask = water_mask * expanded_surface_mask
    return surface_mask, water_mask, water_surface_mask


def create_initial_condition(water_mask):
    def f(x, y, z):
        return (x >= 15) & (x <= 55) & (y >= 15) & (y <= 55) & (z >= 15) & (z <= 55)

    ice_mask = generate_mask_from_func(N, dx, f)

    phi = torch.zeros((1, 1, N[0], N[1], N[2]))
    phi[0, 0] = ice_mask
    phi = phi * water_mask
    return phi


def main_run_simulation():
    save_dir = Path("./homogeneous")
    logger = setup_logger(save_dir)
    theta = 55
    k = np.cos(np.radians(theta))

    surface_mask, water_mask, water_surface_mask = create_surface()
    phi = create_initial_condition(water_mask)

    with torch.no_grad():
        lambda_0 = compute_lambda(phi, water_mask)

    t_list = [0, 1000, 10000, 10000]
    lambda_list = [lambda_0, lambda_0, lambda_0, lambda_0]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)

    phase_field_model = PhaseFieldModel(theta, phi, surface_mask, water_mask, water_surface_mask)
    phase_field_model.run(lambda_star_func, t_list[-1], save_dir, logger)


def main_check_result():
    save_dir = Path("./homogeneous")
    surface_mask, water_mask, water_surface_mask = create_surface()
    phi_cpu = np.load(save_dir / "phi.npy")
    phi = torch.from_numpy(phi_cpu).unsqueeze(0).unsqueeze(0)
    grad = compute_corrected_gradient_squared(phi, water_mask, water_surface_mask)
    print()


def main():
    main_run_simulation()
    # main_check_result()


if __name__ == "__main__":
    main()
