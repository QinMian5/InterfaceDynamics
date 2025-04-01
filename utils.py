# Author: Mian Qin
# Date Created: 2025/3/27
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh


def gaussian_smooth(phi: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5):
    """
    三维高斯滤波
    Args:
        phi: 输入相场张量，形状 (1, 1, Nz, Ny, Nx)
        sigma: 高斯核标准差
        kernel_size: 核尺寸（奇数）
    """
    # 生成高斯核
    ax = torch.linspace(-(kernel_size//2), kernel_size//2, steps=kernel_size)
    x, y, z = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size).to(phi.device)

    # 应用卷积（填充保持尺寸）
    padding = kernel_size // 2
    return F.conv3d(phi.unsqueeze(0).unsqueeze(0), kernel, padding=padding).squeeze()


def export_interface(phi: torch.Tensor, dx: float, output_prefix: str):
    """将相场数据导出为 PLY 和 Pickle 文件

    Args:
        phi (torch.Tensor): [Nx, Ny, Nz] 相场张量 (0=water, 1=ice)
        dx (float): 网格间距 (Å)
        output_prefix (str): 输出文件前缀（不含扩展名）
    """
    # 转换到 CPU 和 numpy
    phi = gaussian_smooth(phi)
    phi_np = phi.detach().cpu().numpy()  # shape (Nx, Ny, Nz)

    # Marching Cubes 提取等值面
    vertices, faces, _, _ = marching_cubes(
        phi_np,
        level=0.5,
        spacing=(dx, dx, dx),  # 确保与物理空间坐标一致
        allow_degenerate=False
    )

    # 生成 interface_type 标签 (全部为 ice-water)
    interface_type = np.zeros(len(faces), dtype=int)  # 0 表示 ice-water interface

    # 保存为 Pickle (兼容现有 Blender 代码)
    with open(f"{output_prefix}.pickle", "wb") as f:
        pickle.dump((vertices, faces, interface_type), f)

    # 额外保存为 PLY 文件
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(f"{output_prefix}.ply", file_type="ply", encoding='ascii')

    return vertices, faces


def main():
    ...


if __name__ == "__main__":
    main()
