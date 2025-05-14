# Author: Mian Qin
# Date Created: 2025/3/27
import os
from pathlib import Path
from itertools import accumulate
from tqdm import tqdm
import cProfile

import numpy as np
import scipy.constants as const
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

from utils import *
from utils_plot import *
from phase_field_model import PhaseFieldModel, dx, device, compute_lambda, compute_corrected_gradient_squared, \
    compute_mean_curvature, compute_total_energy, compute_phi_surface_surface, gamma_iw

L = np.array([80, 80, 80])  # unit: A
N = np.floor(L / dx).astype(int)


def create_surface(device="cpu"):
    def f(x, y, z):
        return x < -100  # all false, no surface

    surface_mask = generate_mask_from_func(N, dx, f).float().unsqueeze(0).unsqueeze(0)
    # print_memory_mb(surface_mask)
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
    return surface_mask.to(device), water_mask.to(device), surface_surface_mask.to(device), water_surface_mask.to(device)


def create_initial_condition(water_mask):
    # def f(x, y, z):
    #     return (x >= 15) & (x <= 55) & (y >= 15) & (y <= 55) & (z >= 15) & (z <= 55)

    def f(x, y, z):
        r = 20
        return (x - 40) ** 2 + (y - 40) ** 2 + (z - 40) ** 2 <= r ** 2

    ice_mask = generate_mask_from_func(N, dx, f)

    phi = torch.zeros((1, 1, N[0], N[1], N[2]))
    phi[0, 0] = ice_mask
    phi = phi * water_mask
    return phi


def main_run_simulation():
    theta = 55
    t_eql_init = 5000
    ramp_rate = 0.1
    t_eql_prd = 5000

    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface()
    phi = create_initial_condition(water_mask)

    with torch.no_grad():
        lambda_0 = compute_lambda(phi, water_mask)

    lambda_star = 3000
    job_name = f"{int(lambda_0)}_{lambda_star}_{ramp_rate}"
    save_dir = Path("./homogeneous") / f"{job_name}"
    logger = setup_logger(save_dir)

    t_ramp = int(abs(lambda_0 - lambda_star) / ramp_rate)
    t_list = list(accumulate([0, t_eql_init, t_ramp, t_eql_prd]))
    # t_list = list(accumulate([0, t_eql_init, 0, 0]))
    lambda_list = [lambda_0, lambda_0, lambda_star, lambda_star]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)

    phase_field_model = PhaseFieldModel(theta, phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask)
    phase_field_model.run(lambda_star_func, t_list[-1], save_dir, logger)

    main_post_processing(theta, job_name)


def main_check_result():
    save_dir = Path("./homogeneous")
    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface()
    phi_cpu = np.load(save_dir / "phi.npy")
    phi = torch.from_numpy(phi_cpu).unsqueeze(0).unsqueeze(0)
    grad = compute_corrected_gradient_squared(phi, water_mask)
    print()


@torch.no_grad()
def main_post_processing(theta, job_name):
    k = np.cos(np.radians(float(theta)))
    save_dir = Path("./homogeneous") / job_name
    trajectory_dir = save_dir / "trajectory"

    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface(device=device)
    file_path_list = sorted(list(trajectory_dir.glob("*.npy")), key=lambda x: int(x.stem))

    data = []
    for file_path in tqdm(file_path_list):
        t = int(file_path.stem)
        phi_cpu = np.load(file_path)
        phi = torch.from_numpy(phi_cpu).to(device)
        phi_surface_surface = compute_phi_surface_surface(phi, surface_mask, water_mask, surface_surface_mask, water_surface_mask)

        lambda_ = compute_lambda(phi, water_mask)
        # H, H_mean, H_std = compute_mean_curvature(phi + phi_surface_surface, water_mask)
        _, _, info = compute_total_energy(phi + phi_surface_surface, surface_mask, water_mask, water_surface_mask, 0)
        total_energy = info["total_energy"]
        surface_energy_iw = info["surface_energy_iw"]
        surface_energy_is = info["surface_energy_is"]
        bulk_energy = info["bulk_energy"]

        # row_data = {"t": t, "lambda": lambda_.item(), "H_mean": H_mean.item(), "H_std": H_std.item(), "total_energy": total_energy}
        row_data = {"t": t, "lambda": lambda_.item(), "total_energy": total_energy, "surface_energy_iw": surface_energy_iw,
                    "surface_energy_is": surface_energy_is, "bulk_energy": bulk_energy}
        data.append(row_data)
    df = pd.DataFrame(data)
    df.to_csv(save_dir / "intermediate_result.csv", index=False)
    print()


def main():
    main_run_simulation()
    # main_check_result()


if __name__ == "__main__":
    main()
