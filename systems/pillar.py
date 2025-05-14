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

w_pillar = 20
theta2r = {
    "45": 50,
    "60": 35,
    # "60": 28,
    "70": 30,
    "80": 25,
    "90": 23,
    # "90": 20,
    "120": 20,
    "150": 20,
}
theta2r_star = {
    "45": 80,
    # "60": 60,
    "60": 50,
    "70": 50,
    "80": 43,
    "90": 41,
    "120": 30,
    "150": 30,
}
theta2L = {
    "45": np.array([130, 130, 50]),
    # "60": np.array([120, 120, 60]),
    "60": np.array([100, 100, 50]),
    "70": np.array([120, 120, 60]),
    "80": np.array([110, 110, 60]),
    # "90": np.array([100, 100, 70]),
    "90": np.array([80, 80, 50]),
    "120": np.array([90, 90, 70]),
    "150": np.array([90, 90, 80]),
}  # unit: A


def create_surface(L, N, device="cpu"):
    # def f(x, y, z):
    #     base = z <= 5
    #     pillar1 = (z <= L[2] - 10) & (x >= L[0] / 2 - w_pillar / 2) & (x <= L[0] / 2 + w_pillar / 2) & (
    #                 y >= L[1] / 2 - w_pillar / 2) & (y <= L[1] / 2 + w_pillar / 2)
    #     return base | pillar1
    #     # pillar2 = (z <= 55) & (x >= 110) & (x <= 130) & (y >= 50) & (y <= 70)
    #     # return base | pillar1 | pillar2

    def f(x, y, z):
        base = z <= 5
        pillar1 = (z <= L[2] - 10) & (
                (x - L[0] / 2 - dx / 2) ** 2 + (y - L[1] / 2 - dx / 2) ** 2 <= (w_pillar / 2 - 1e-5) ** 2)
        return base | pillar1

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
    return surface_mask.to(device), water_mask.to(device), surface_surface_mask.to(device), water_surface_mask.to(
        device)


def create_initial_condition(L, N, water_mask, r, theta):
    def f(x, y, z):
        return (x - L[0] / 2 - dx / 2) ** 2 + (y - L[1] / 2 - dx / 2) ** 2 + (
                    z - (5 - r * np.cos(np.radians(theta)))) ** 2 <= r ** 2

    # def f(x, y, z):
    #     return (x - L[0] / 2 - dx / 2 - 5) ** 2 + (y - L[1] / 2 - dx / 2) ** 2 + (
    #             z - (5 - r * np.cos(np.radians(theta)))) ** 2 <= r ** 2 + 1e-5

    ice_mask = generate_mask_from_func(N, dx, f)

    phi = torch.zeros((1, 1, N[0], N[1], N[2]))
    phi[0, 0] = ice_mask
    phi = phi * water_mask
    return phi


def main_compute_parameters(theta):
    # surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface()
    r_array = np.arange(10, 60, 5)  # A
    r_surface_array = r_array * np.sin(np.radians(theta))
    h_array = r_array * (1 - np.cos(np.radians(theta)))
    V_array = np.pi / 3 * r_array ** 3 * (2 + np.cos(np.radians(theta))) * (
            1 - np.cos(np.radians(theta))) ** 2 - h_array * w_pillar ** 2  # A^3
    lambda_array = V_array * 1e-30 * 5e4 * const.N_A
    for r, r_surface, h, lambda_ in zip(r_array, r_surface_array, h_array, lambda_array):
        print(f"r: {r}, r_surface: {r_surface}, h: {h}, lambda: {lambda_}")


def main_run_simulation(theta: str):
    t_eql_init = 50000
    ramp_rate = 0.1
    t_eql_prd = 5000

    L = theta2L[theta]
    N = np.floor(L / dx).astype(int)

    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface(L, N)
    phi = create_initial_condition(L, N, water_mask, r=theta2r[theta], theta=float(theta))
    phi_target = create_initial_condition(L, N, water_mask, r=theta2r_star[theta], theta=float(theta))

    with torch.no_grad():
        lambda_0 = int(compute_lambda(phi, water_mask))
        lambda_star = int(compute_lambda(phi_target, water_mask))
        lambda_star = lambda_0

    # print(lambda_0, lambda_star)
    # return

    job_name = f"{int(lambda_0)}_{lambda_star}_{ramp_rate}"
    save_dir = Path("./pillar") / f"{theta}/{job_name}"
    logger = setup_logger(save_dir)

    t_ramp = int(abs(lambda_0 - lambda_star) / ramp_rate)
    # t_list = list(accumulate([0, t_eql_init, t_ramp, t_eql_prd]))
    t_list = list(accumulate([0, t_eql_init, 0, 0]))
    lambda_list = [lambda_0, lambda_0, lambda_star, lambda_star]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)

    phase_field_model = PhaseFieldModel(float(theta), phi, surface_mask, water_mask, surface_surface_mask,
                                        water_surface_mask)
    phase_field_model.run(lambda_star_func, t_list[-1], save_dir, logger)

    main_post_processing(theta, job_name)


def main_check_result(theta, job_name):
    save_dir = Path("./pillar") / f"{theta}" / f"{job_name}"
    L = theta2L[theta]
    N = np.floor(L / dx).astype(int)
    trajectory_dir = save_dir / "trajectory"
    file_path_list = sorted(list(trajectory_dir.glob("*.npy")), key=lambda x: int(x.stem))
    print(len(file_path_list))
    for file_path in file_path_list[:]:
        t = int(file_path.stem)
        if t < 16000:
            continue
        phi_cpu = np.load(file_path)
        a = 1
    print()


@torch.no_grad()
def main_post_processing(theta: str, job_name):
    k = np.cos(np.radians(float(theta)))
    save_dir = Path("./pillar") / f"{theta}" / job_name
    trajectory_dir = save_dir / "trajectory"

    L = theta2L[theta]
    N = np.floor(L / dx).astype(int)

    surface_mask, water_mask, surface_surface_mask, water_surface_mask = create_surface(L, N, device=device)
    bulk_mask = water_mask - water_surface_mask
    file_path_list = sorted(list(trajectory_dir.glob("*.npy")), key=lambda x: int(x.stem))

    data = []
    for file_path in tqdm(file_path_list[:]):
        t = int(file_path.stem)
        phi_cpu = np.load(file_path)
        phi = torch.from_numpy(phi_cpu).to(device)
        phi_surface_surface = compute_phi_surface_surface(phi, surface_mask, water_mask, surface_surface_mask,
                                                          water_surface_mask)

        # phi_surface_surface_cpu = phi_surface_surface.cpu()
        lambda_ = compute_lambda(phi, water_mask)
        # H, H_mean, H_std = compute_mean_curvature(phi, phi_surface_surface, water_mask, bulk_mask)
        _, _, info = compute_total_energy(phi + phi_surface_surface, surface_mask, water_mask, water_surface_mask,
                                          -k * gamma_iw)
        total_energy = info["total_energy"]
        surface_energy_iw = info["surface_energy_iw"]
        surface_energy_is = info["surface_energy_is"]
        bulk_energy = info["bulk_energy"]

        # row_data = {"t": t, "lambda": lambda_.item(), "H_mean": H_mean.item(), "H_std": H_std.item(), "total_energy": total_energy}
        row_data = {"t": t, "lambda": lambda_.item(), "total_energy": total_energy,
                    "surface_energy_iw": surface_energy_iw,
                    "surface_energy_is": surface_energy_is, "bulk_energy": bulk_energy}
        data.append(row_data)
    df = pd.DataFrame(data)
    df.to_csv(save_dir / "intermediate_result.csv", index=False)


def main_draw_trajectory():
    theta = 55
    root_dir = Path("./pillar") / f"{theta}"
    job_name_list = ["323_10000_0.2"]
    skip_t_dict = {"323_10000_0.2": 0000}
    # job_name_list = ["508_5500_0.2", "508_5500_0.1", ("797_2000_0.02", "1995_5000_0.1"), "1392_500_0.1", "697_300_0.1",
    #                  "490_4000_0.1", "286_3000_0.1"]
    # skip_t_dict = {"5540_500": 1000, "487_5500": 0, "508_5500_0.1": 10000, "508_5500_0.2": 10000, "797_2000_0.02": 0,
    #                "1995_5000_0.1": 0, "1392_500_0.1": 0, "697_300_0.1": 0, "490_4000_0.1": 0, "286_3000_0.1": 0}

    fig_traj_G_lambda, ax_traj_G_lambda = create_fig_ax(r"trajectory, $G(\lambda)$", r"$\lambda$", r"$G\ (kT)$")

    for job_name_tuple in job_name_list:
        if isinstance(job_name_tuple, str):
            job_name_tuple = (job_name_tuple,)

        df_list = []
        for job_name in job_name_tuple:
            data_dir = root_dir / job_name
            df = pd.read_csv(data_dir / "intermediate_result.csv")

            skip_t = skip_t_dict[job_name]
            t = df["t"]
            valid_index = t > skip_t
            df_valid = df[valid_index]
            df_list.append(df_valid)

            t = df_valid["t"]
            lambda_ = df_valid["lambda"]
            # H_mean = df_valid["H_mean"]
            # H_std = df_valid["H_std"]
            G = df_valid["total_energy"] / (const.k * 300)
            # H = unp.uarray(H_mean, H_std)

            fig, ax = create_fig_ax(r"$\lambda(t)$", r"$t$", r"$\lambda$")
            plot_with_error_band(ax, t, lambda_)
            figure_save_path = root_dir / "figure" / f"lambda_t_{job_name}.png"
            save_figure(fig, figure_save_path)
            plt.close(fig)

            # fig, ax = create_fig_ax(r"Mean Curvature, $H(t)$", r"$t$", r"$H\ (Å^{-1})$")
            # plot_with_error_band(ax, t, H)
            # figure_save_path = root_dir / "figure" / f"H_t_{job_name}.png"
            # save_figure(fig, figure_save_path)
            # plt.close(fig)

            # fig, ax = create_fig_ax(r"Mean Curvature, $H(\lambda)$", r"$\lambda$", r"$H\ (Å^{-1})$")
            # plot_with_error_band(ax, lambda_, H)
            # figure_save_path = root_dir / "figure" / f"H_lambda_{job_name}.png"
            # save_figure(fig, figure_save_path)
            # plt.close(fig)

        df = pd.concat(df_list)

        t = df["t"]
        lambda_ = df["lambda"]
        # H_mean = df["H_mean"]
        # H_std = df["H_std"]
        G = df["total_energy"] / (const.k * 300)
        # H = unp.uarray(H_mean, H_std)

        ax_traj_G_lambda.plot(lambda_, G, "-", label="+".join(job_name_tuple), alpha=1.0)

    ax_traj_G_lambda.legend()
    figure_save_path = root_dir / "figure" / "traj_G_lambda.png"
    save_figure(fig_traj_G_lambda, figure_save_path)
    plt.close(fig_traj_G_lambda)


def main():
    # main_compute_parameters(150)
    main_run_simulation("90")
    # main_check_result("90", "517_3628_0.1")
    # main_post_processing("45", "760_3534_0.1")
    # main_draw_trajectory()

    # for theta in ["120", "150"]:
    #     main_run_simulation(theta)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.print_stats(sort="cumulative")
