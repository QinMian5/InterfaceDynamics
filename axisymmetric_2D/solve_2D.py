# Author: Mian Qin
# Date Created: 2025/5/27
from pathlib import Path
import json

import numpy as np
from scipy.optimize import root_scalar, minimize
from scipy.integrate import quad
import scipy.constants as const

from utils_plot import create_fig_ax, save_figure

rho = 5.09e4 * 1e-30 * const.N_A  # 1/A^3
delta_mu = 0  # J/mol
gamma_iw = 25.9e-3 * 1e-20 * const.N_A / 1000  # J/A^2, real water


def r_param_eq(t, a, r_pillar):
    return r_pillar + sum(a_i * np.cos((2 * i + 1) * t) for i, a_i in enumerate(a))


def dr_dt_param_eq(t, a):
    return sum(-(2 * i + 1) * a_i * np.sin((2 * i + 1) * t) for i, a_i in enumerate(a))


def z_param_eq(t, b):
    return sum(b_i * np.sin((2 * i + 1) * t) for i, b_i in enumerate(b))


def dz_dt_param_eq(t, b):
    return sum((2 * i + 1) * b_i * np.cos((2 * i + 1) * t) for i, b_i in enumerate(b))


def E_iw_func(a, b, r_pillar, gamma_iw):
    def integrand(t):
        return np.sqrt(dz_dt_param_eq(t, b) ** 2 + dr_dt_param_eq(t, a) ** 2) * 2 * np.pi * r_param_eq(t, a, r_pillar)

    E_iw = gamma_iw * quad(integrand, 0, np.pi / 2)[0]
    return E_iw


def E_is_func(a, b, r_pillar, gamma_iw, theta_rad: float):
    r_0 = r_param_eq(0, a, r_pillar)
    z_0 = z_param_eq(np.pi / 2, b)
    E_is = -gamma_iw * np.cos(theta_rad) * (np.pi * (r_0 ** 2 - r_pillar ** 2) + 2 * np.pi * r_pillar * z_0)
    return E_is


def V_func(a, b, r_pillar):
    def integrand(t):
        return -2 * np.pi * r_param_eq(t, a, r_pillar) * dr_dt_param_eq(t, a) * z_param_eq(t, b)

    V = quad(integrand, 0, np.pi / 2)[0]
    return V


def main_minimize(n, lambda_star, r_pillar, theta_rad, x0=None, show_plot=False, print_debug=False):
    V_0 = lambda_star / rho

    def objective_func(x):
        a, b = x[:n], x[n:]
        V = V_func(a, b, r_pillar)
        current_lambda = V * rho
        E_bias_potential = 0.05 * (current_lambda - lambda_star) ** 2
        E_iw = E_iw_func(a, b, r_pillar, gamma_iw)
        E_is = E_is_func(a, b, r_pillar, gamma_iw, theta_rad)
        E = E_iw + E_is + E_bias_potential
        return E

    def constraint_func(x):
        a, b = x[:n], x[n:]
        V = V_func(a, b, r_pillar)
        return V - V_0

    if x0 is None:
        x0 = np.zeros(2 * n)
        x0[0] = 10
        x0[n] = 10

    rough_result = minimize(objective_func, x0=x0, method="L-BFGS-B",
                            options={"maxiter": 100})
    result = minimize(objective_func, x0=rough_result.x, method="Powell", )
    if print_debug:
        print(f"n = {n}")
        print(result)
    if result.success:
        a = result.x[:n]
        b = result.x[n:]
        if show_plot:
            fig, ax = create_fig_ax("Axisymmetric Interface", "$r$", "$z$")
            t = np.linspace(0.0, np.pi / 2, 1000)
            r = r_param_eq(t, a, r_pillar)
            z = z_param_eq(t, b)
            ax.plot(r, z, "-")
            fig.show()
        return result.success, (a, b)
    else:
        return result.success, (None, None)


def main_increase_n(n_target, lambda_star, r_pillar, theta_deg):
    root_dir = Path("./axisymmetric_2D")
    root_dir.mkdir(exist_ok=True)
    intermediate_result_save_dir = root_dir / "intermediate_results"
    intermediate_result_save_dir.mkdir(exist_ok=True)
    result_save_dir = root_dir / "results"
    result_save_dir.mkdir(exist_ok=True)
    figure_save_dir = root_dir / "figures"
    figure_save_dir.mkdir(exist_ok=True)

    theta_rad = np.radians(theta_deg)
    a_dict = {}
    b_dict = {}
    for n in range(1, n_target + 1):
        x0 = np.zeros(2 * n)
        if n == 1:
            x0[0] = 1
            x0[n] = 1
        else:
            a0 = np.array(a_dict[n - 1])
            b0 = np.array(b_dict[n - 1])
            x0[:n - 1] = a0
            x0[n:2 * n - 1] = b0
        success, (a, b) = main_minimize(n, lambda_star, r_pillar, theta_rad, x0)
        if success:
            print(f"theta = {theta_deg}, n = {n}: success")
            a_dict[n] = a.tolist()
            b_dict[n] = b.tolist()
        else:
            print(f"theta = {theta_deg}, n = {n} failed to converge")
            break
    with open(intermediate_result_save_dir / f"{theta_deg}.json", "w") as f:
        json.dump([a_dict, b_dict], f, indent=4)

    results = {}
    for n in range(1, n_target + 1):
        if n in a_dict and n in b_dict:
            a = a_dict[n]
            b = b_dict[n]
            t = np.linspace(0.0, np.pi / 2, 1000)
            E_iw = E_iw_func(a, b, r_pillar, gamma_iw)
            E_is = E_is_func(a, b, r_pillar, gamma_iw, theta_rad)
            results[n] = {"E_iw": E_iw, "E_is": E_is}

    with open(result_save_dir / f"{theta_deg}.json", "w") as f:
        json.dump(results, f, indent=4)

    fig, ax = create_fig_ax("Axisymmetric Interface", "$r$", "$z$")
    for n in range(1, n_target + 1):
        if n in a_dict and n in b_dict:
            a = a_dict[n]
            b = b_dict[n]
            t = np.linspace(0.0, np.pi / 2, 1000)
            r = r_param_eq(t, a, r_pillar)
            z = z_param_eq(t, b)
            current_lambda = V_func(a, b, r_pillar) * rho
            ax.plot(r, z, "-", label=fr"$n = {n}, \lambda = {current_lambda:.0f}$")
    ax.legend()

    figure_save_path = figure_save_dir / f"{theta_deg}.png"
    save_figure(fig, figure_save_path)


def main():
    n = 7
    r_pillar = 10
    theta_deg = 70
    theta_rad = np.radians(theta_deg)
    # main_minimize(n, 2000, r_pillar, theta_rad, show_plot=True)
    for theta_deg in [45, 60, 70, 80, 90, 120, 150]:
        main_increase_n(n, 1000, r_pillar, theta_deg)
    # print(E_iw_func(a, b, r_pillar, gamma_iw))
    # print(E_is_func(a, b, r_pillar, gamma_iw, theta))
    # print(V_func(a, b, r_pillar))


if __name__ == "__main__":
    main()
