# Author: Mian Qin
# Date Created: 2025/3/27
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh


gaussian_kernel_dict = {}


def gaussian_smooth(phi: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5):
    """
    三维高斯滤波
    Args:
        phi: 输入相场张量，形状 (1, 1, Nz, Ny, Nx)
        sigma: 高斯核标准差
        kernel_size: 核尺寸（奇数）
    """
    if (sigma, kernel_size) in gaussian_kernel_dict:
        kernel = gaussian_kernel_dict[(sigma, kernel_size)]
    else:
        ax = torch.linspace(-(kernel_size//2), kernel_size//2, steps=kernel_size)
        x, y, z = torch.meshgrid(ax, ax, ax, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size).to(phi.device)
        gaussian_kernel_dict[(sigma, kernel_size)] = kernel

    # 应用卷积（填充保持尺寸）
    pad = kernel_size // 2
    padded_phi = F.pad(phi, pad=(pad, pad, pad, pad, pad, pad), mode="replicate")
    return F.conv3d(padded_phi, kernel).squeeze()


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


def export_interface(phi: torch.Tensor, dx: float, output_prefix: str):
    """将相场数据导出为 PLY 和 Pickle 文件

    Args:
        phi (torch.Tensor): [Nx, Ny, Nz] 相场张量 (0=water, 1=ice)
        dx (float): 网格间距 (Å)
        output_prefix (str): 输出文件前缀（不含扩展名）
    """
    nodes, faces, interface_type = generate_interface(phi, dx, level=0.5, smooth=False)

    # 保存为 Pickle (兼容现有 Blender 代码)
    with open(f"{output_prefix}.pickle", "wb") as f:
        pickle.dump((nodes, faces, interface_type), f)

    # 额外保存为 PLY 文件
    mesh = trimesh.Trimesh(vertices=nodes, faces=faces)
    mesh.export(f"{output_prefix}.ply", file_type="ply", encoding='ascii')

    return nodes, faces, interface_type


def main():
    ...


if __name__ == "__main__":
    main()
