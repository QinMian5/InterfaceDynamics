# Author: Mian Qin
# Date Created: 2025/3/27
import pickle
import logging
import shutil
from pathlib import Path
from typing import Optional
from functools import lru_cache
from itertools import count

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh


def setup_logger(log_dir: Path, name: Optional[str] = None) -> logging.Logger:
    """
    设置一个同时输出到控制台和文件的 logger，并处理日志文件的轮转

    参数:
        save_dir: 日志文件保存的目录
        name: logger 的名称，如果为 None 则使用 root logger

    返回:
        配置好的 logger 对象
    """
    # 创建保存目录（如果不存在）
    log_dir.mkdir(exist_ok=True)

    # 设置日志文件路径
    log_file = log_dir / "log.log"

    if log_file.exists():
        for i in count(start=1):
            filename = log_file.name
            backup_name = f"#{filename}.{i}#"
            backup_path = log_file.parent / backup_name
            if not backup_path.exists():
                shutil.move(log_file, backup_path)
                break

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低日志级别

    # 清除已有的 handler，避免重复
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建 formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 创建文件 handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别的日志
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只记录 INFO 及以上级别的日志
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


@lru_cache(maxsize=10)
def create_gaussian_kernel(sigma, kernel_size):
    ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, steps=kernel_size)
    x, y, z = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
    return kernel


def gaussian_smooth(phi: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5):
    """
    三维高斯滤波
    Args:
        phi: 输入相场张量，形状 (1, 1, Nz, Ny, Nx)
        sigma: 高斯核标准差
        kernel_size: 核尺寸（奇数）
    """
    kernel = create_gaussian_kernel(sigma, kernel_size).to(phi.device)

    # 应用卷积（填充保持尺寸）
    pad = kernel_size // 2
    padded_phi = F.pad(phi, pad=(pad, pad, pad, pad, pad, pad), mode="replicate")
    return F.conv3d(padded_phi, kernel).squeeze()


def create_lambda_star_func(t_list, lambda_list):
    assert len(t_list) == len(lambda_list)
    lambda_star_func = interp1d(t_list, lambda_list, kind="linear", bounds_error=True)
    return lambda_star_func


def generate_interface(phi: torch.Tensor, dx: float, level=0.5, smooth=False):
    if smooth:
        phi = gaussian_smooth(phi)
    phi = phi.squeeze()
    phi_np = phi.detach().cpu().numpy()  # shape (Nx, Ny, Nz)
    if level < phi_np.min() or level > phi_np.max():
        nodes, faces = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    else:
        nodes, faces, _, _ = marching_cubes(
            phi_np,
            level=0.5,
            spacing=(dx, dx, dx),
            step_size=int(1/dx),
            allow_degenerate=False
        )
    interface_type = np.zeros(len(faces), dtype=int)  # 0: ice-water interface
    return nodes, faces, interface_type


def export_phi(phi: torch.Tensor, save_dir: Path):
    save_dir.mkdir(exist_ok=True)

    phi = phi.squeeze()
    phi_np = phi.detach().cpu().numpy()  # shape (Nx, Ny, Nz)
    np.save(save_dir / "phi.npy", phi_np)


def export_surface(surface_mask: torch.tensor, dx: float, save_dir: Path):
    save_dir.mkdir(exist_ok=True)

    nodes, faces, _ = generate_interface(surface_mask, dx, smooth=False)
    if nodes.shape[0] > 0:
        mesh = trimesh.Trimesh(vertices=nodes, faces=faces)
        mesh.export(save_dir / "surface.ply", file_type="ply", encoding='ascii')


def export_interface(phi: torch.Tensor, dx: float, save_dir: Path, output_prefix="interface"):
    """将相场数据导出为 PLY 和 Pickle 文件

    Args:
        phi (torch.Tensor): [Nx, Ny, Nz] 相场张量 (0=water, 1=ice)
        dx (float): 网格间距 (Å)
        save_dir: 保存目录
        output_prefix (str): 输出文件前缀（不含扩展名）
    """
    save_dir.mkdir(exist_ok=True)
    nodes, faces, interface_type = generate_interface(phi, dx, level=0.5, smooth=False)

    # 保存为 Pickle (兼容现有 Blender 代码)
    with open(save_dir / f"{output_prefix}.pickle", "wb") as f:
        pickle.dump((nodes, faces, interface_type), f)

    # 额外保存为 PLY 文件
    if nodes.shape[0] > 0:
        mesh = trimesh.Trimesh(vertices=nodes, faces=faces)
        mesh.export(save_dir / f"{output_prefix}.ply", file_type="ply", encoding='ascii')

    return nodes, faces, interface_type


def main():
    t_list = [0, 10, 20]
    lambda_list = [0, 50, 200]
    lambda_star_func = create_lambda_star_func(t_list, lambda_list)


if __name__ == "__main__":
    main()
