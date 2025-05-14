# Author: Mian Qin
# Date Created: 9/18/24
from pathlib import Path

import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
import matplotlib.container
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import uncertainties.unumpy as unp


def create_fig_ax(title, x_label, y_label, **kwargs) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots(**kwargs)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return fig, ax


def plot_with_error_band(ax: matplotlib.axes.Axes, x, y_u, fmt="", n_sigma=2, **kwargs) -> matplotlib.lines.Line2D:
    y = unp.nominal_values(y_u)
    y_err = n_sigma * unp.std_devs(y_u)
    line = ax.plot(x, y, fmt, **kwargs)[0]
    color = line.get_color()
    ax.fill_between(x, y - y_err, y + y_err, color=color, alpha=0.2)
    return line


def plot_with_error_bar(ax: matplotlib.axes.Axes, x, y_u, fmt="", n_sigma=2,
                        **kwargs) -> matplotlib.container.ErrorbarContainer:
    y = unp.nominal_values(y_u)
    y_err = n_sigma * unp.std_devs(y_u)
    container = ax.errorbar(x, y, yerr=y_err, fmt=fmt, capsize=5,
                            capthick=1, elinewidth=1, **kwargs)
    return container


def save_figure(fig: matplotlib.figure.Figure, path: Path):
    save_dir = path.parent
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved the figure to {path.resolve()}")


def combine_images(image_paths: list[Path], titles: list[str], m: int, n: int, fontsize=12):
    """
    将多张图片合并为一个 m*n 的布局，并在每张图片的左上角添加标题。

    参数:
    - image_paths: list, 图片路径的列表。
    - titles: list, 每张图片对应的标题列表。
    - m: int, 布局的行数。
    - n: int, 布局的列数。
    - figsize: tuple, 整个画布的大小，默认为 (10, 10)。
    - fontsize: int, 标题的字体大小，默认为 12。
    - save_path: str, 合并后图片的保存路径。如果为 None，则不保存图片。
    """
    # 检查输入是否有效
    if len(image_paths) != m * n:
        raise ValueError(f"图片数量 ({len(image_paths)}) 与布局 {m}x{n} 不匹配！")
    if len(titles) != m * n:
        raise ValueError(f"标题数量 ({len(titles)}) 与布局 {m}x{n} 不匹配！")

    plt.style.use("presentation.mplstyle")
    # 创建 m*n 的子图布局
    fig, axes = plt.subplots(m, n, figsize=(3 * n, 3 * m))
    axes = axes.flatten()  # 将二维的 axes 数组展平为一维，方便遍历

    # 遍历每张图片并添加到子图中
    for i, (img_path, title) in enumerate(zip(image_paths, titles)):
        img = mpimg.imread(img_path)  # 加载图片
        axes[i].imshow(img)  # 显示图片
        axes[i].set_title(title, loc='left', fontsize=fontsize, pad=-20)  # 添加标题
        axes[i].axis('off')  # 关闭坐标轴

        # 添加边框
        border = Rectangle((0, 0), 1, 1, transform=axes[i].transAxes,
                           edgecolor="black", linewidth=2, facecolor='none')
        axes[i].add_patch(border)

    # 调整子图之间的间距
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig


def main():
    ...


if __name__ == "__main__":
    main()
