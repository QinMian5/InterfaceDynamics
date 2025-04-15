# Author: Mian Qin
# Date Created: 2025/3/27
import os
from pathlib import Path
from itertools import accumulate

import numpy as np
import scipy.constants as const
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt

from utils import *
from phase_field_model import PhaseFieldModel, dx, compute_lambda, compute_corrected_gradient_squared, update_surface

L = np.array([70, 70, 70])  # unit: A
N = np.floor(L / dx).astype(int)


def create_surface():
    surface_mask = torch.zeros((N[0], N[1], N[2]))
    surface_mask = surface_mask.view(1, 1, N[0], N[1], N[2])
    surface_mask[..., :, :, :20] = 1
    surface_mask[..., 50:90, 50:90, 0:80] = 1
    water_mask = 1 - surface_mask

    kernel = torch.zeros((5, 5, 5))
    kernel[2, 2, :] = 1
    kernel[2, :, 2] = 1
    kernel[:, 2, 2] = 1
    kernel = kernel.view(1, 1, 5, 5, 5)

    padded_surface_mask = F.pad(surface_mask, (2, 2, 2, 2, 2, 2), mode='circular')
    expanded_surface = F.conv3d(padded_surface_mask, kernel, stride=[1, 1, 1])
    expanded_surface_mask = torch.clamp(expanded_surface, 0, 1)
    water_surface_mask = water_mask * expanded_surface_mask

    padded_water_mask = F.pad(water_mask, (2, 2, 2, 2, 2, 2), mode='circular')
    expanded_water = F.conv3d(padded_water_mask, kernel, stride=[1, 1, 1])
    expanded_water_mask = torch.clamp(expanded_water, 0, 1)
    surface_surface_mask = surface_mask * expanded_water_mask
    return surface_mask, water_mask, surface_surface_mask, water_surface_mask


def create_initial_condition(water_mask):
    # def f(x, y, z):
    #     # Full of ice
    #     return z <= 50

    def f(x, y, z):
        return (x - 35) ** 2 + (y - 35) ** 2 + (z - 10) ** 2 <= 20 ** 2

    ice_mask = generate_mask_from_func(N, dx, f)

    phi = torch.zeros((1, 1, N[0], N[1], N[2]))
    phi[0, 0] = ice_mask
    phi = phi * water_mask
    return phi


def main_run_simulation():
    t_eql_init = 10000
    ramp_rate = 1 / 5
    t_eql_prd = 20000

    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface()
    phi = create_initial_condition(water_mask)
    # phi = torch.from_numpy(np.load(save_dir / "phi.npy")).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        lambda_0 = compute_lambda(phi, water_mask)

    theta = 55
    lambda_star = 5000
    save_dir = Path("./pillar") / f"{theta}/{int(lambda_0)}_{lambda_star}"
    logger = setup_logger(save_dir)

    t_ramp = int(abs(lambda_0 - lambda_star) / ramp_rate)
    t_list = list(accumulate([0, t_eql_init, t_ramp, t_eql_prd]))
    # t_list = list(accumulate([0, t_eql_init, 0, 0]))
    lambda_list = [lambda_0, lambda_0, lambda_star, lambda_star]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)

    phase_field_model = PhaseFieldModel(theta, phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask)
    phase_field_model.run(lambda_star_func, t_list[-1], save_dir, logger)


def main_check_result():
    save_dir = Path("./pillar")
    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface()
    phi_cpu = np.load(save_dir / "trajectory" / "040000.npy")
    phi = torch.from_numpy(phi_cpu).unsqueeze(0).unsqueeze(0)
    phi = update_surface(phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask)
    # grad_bulk = compute_gradient_squared(phi) * (water_mask - water_surface_mask)
    grad = compute_corrected_gradient_squared(phi, water_mask)
    print()


def main_post_processing():
    theta = 55
    job_name = "5540_500"



def main():
    main_run_simulation()
    # main_check_result()


if __name__ == "__main__":
    main()
