"""
日志工具模块
功能概述: 提供统一的日志配置和管理接口
设计思路: 支持同时输出到控制台和文件，可配置日志级别和格式
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    配置并返回一个logger实例

    参数:
        name: logger名称，通常使用__name__
        log_file: 日志文件路径，如果为None则不记录到文件
        level: 日志级别，默认为INFO
        log_to_console: 是否输出到控制台，默认为True
        log_format: 自定义日志格式，如未提供则使用默认格式

    返回:
        配置好的logger实例

    示例:
        >>> logger = setup_logger(__name__, "app.log")
        >>> logger.info("程序启动")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    formatter = logging.Formatter(log_format)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        dir_path = os.path.dirname(log_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamped_log_filename(prefix: str = "ptm2cellnet") -> str:
    """
    生成带时间戳的日志文件名

    参数:
        prefix: 文件名前缀

    返回:
        带时间戳的日志文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/logs/{prefix}_{timestamp}.log"
