# Author: Mian Qin
# Date Created: 2025/3/27
import pickle
import logging
import shutil
from pathlib import Path
from typing import Optional, Callable
from itertools import count

from scipy.interpolate import interp1d


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
    log_dir.mkdir(exist_ok=True, parents=True)

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


def create_lambda_star_func(t_list, lambda_list):
    assert len(t_list) == len(lambda_list)
    lambda_star_func = interp1d(t_list, lambda_list, kind="linear", bounds_error=True)
    return lambda_star_func


def main():
    ...


if __name__ == "__main__":
    main()
