"""
日志工具模块
功能概述: 提供统一的日志配置和管理接口
设计思路: 支持同时输出到控制台和文件，可配置日志级别和格式
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional


# 支持的预设日志格式名称。``log_format`` 既可传入完整的 formatter 字符串，
# 也可传入这里的预设名称（用于与 ``configs/production.yaml`` 的
# ``logging.log_format: "json"`` 等配置对接）。
_PRESET_FORMATS = {
    "text": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "json": "__json__",  # 哨兵值，表示使用 JSON formatter
}


class _JsonFormatter(logging.Formatter):
    """将日志记录序列化为单行 JSON，便于日志聚合系统（ELK/Loki 等）消费。"""

    _RESERVED_KEYS = (
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 合并 logrecord 中通过 extra 附加的字段
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_KEYS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _resolve_formatter(log_format: Optional[str]) -> logging.Formatter:
    """根据 log_format 选择合适的 Formatter。

    优先识别预设名称（``text`` / ``json``）；否则视为标准 formatter 字符串。
    """
    if log_format is None:
        log_format = "text"

    preset = _PRESET_FORMATS.get(log_format)
    if preset == "__json__":
        return _JsonFormatter()
    if preset is not None:
        return logging.Formatter(preset)
    # 当作自定义 formatter 字符串
    return logging.Formatter(log_format)


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
        log_format: 自定义日志格式。可为标准 formatter 字符串，或预设名称
            ``"text"``（默认）/``"json"``（单行 JSON，对接 production.yaml）。

    返回:
        配置好的logger实例

    示例:
        >>> logger = setup_logger(__name__, "app.log")
        >>> logger.info("程序启动")
        >>> # 生产环境 JSON 日志
        >>> logger = setup_logger(__name__, log_format="json")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = _resolve_formatter(log_format)

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
