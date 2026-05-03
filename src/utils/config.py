"""
配置管理模块
功能概述: 提供统一的配置加载、验证和访问接口
设计思路: 使用YAML文件存储配置，支持多级嵌套配置，提供类型安全的访问方式
"""

import os
from typing import Any, Dict, Optional
import yaml  # type: ignore[import-untyped]


class Config:
    """
    配置管理类
    负责从YAML文件加载配置，并提供安全的配置访问接口
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置对象

        参数:
            config_dict: 可选的配置字典，如果提供则直接使用
        """
        self._config: Dict[str, Any] = config_dict or {}

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        """
        从YAML文件加载配置

        参数:
            file_path: YAML配置文件路径

        返回:
            Config对象

        异常:
            FileNotFoundError: 文件不存在时抛出
            yaml.YAMLError: YAML解析错误时抛出
        """
        resolved_path = file_path
        if not os.path.isabs(file_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidates = [
                os.path.abspath(file_path),
                os.path.join(project_root, file_path),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    resolved_path = candidate
                    break

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键访问

        参数:
            key: 配置键，支持点号分隔，如 "data.batch_size"
            default: 默认值，当键不存在时返回

        返回:
            配置值或默认值

        示例:
            >>> config.get("data.max_sequence_length")
            1000
            >>> config.get("nonexistent.key", "default_value")
            "default_value"
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持点号分隔的嵌套键

        参数:
            key: 配置键，支持点号分隔
            value: 要设置的值
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典

        返回:
            配置字典的深拷贝
        """
        import copy
        return copy.deepcopy(self._config)

    def save(self, file_path: str) -> None:
        """
        保存配置到YAML文件

        参数:
            file_path: 保存路径
        """
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return False
            value = value[k]
        return True
