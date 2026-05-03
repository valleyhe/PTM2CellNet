"""
文件IO工具模块
功能概述: 提供统一的文件读写接口，支持多种格式
设计思路: 封装常见文件操作，提供类型安全的接口，自动处理目录创建
"""

import os
import importlib
import pickle
import json
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch


def _ensure_parent_dir(file_path: str) -> None:
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


class _SafeUnpickler(pickle.Unpickler):
    _SAFE_GLOBALS = {
        "builtins": {
            "dict",
            "list",
            "set",
            "frozenset",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "bytes",
        },
        "numpy": {"dtype", "ndarray"},
        "numpy.core.multiarray": {"_reconstruct"},
    }

    def find_class(self, module: str, name: str):
        if module in self._SAFE_GLOBALS and name in self._SAFE_GLOBALS[module]:
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f"Disallowed global: {module}.{name}")


def save_pickle(obj: Any, file_path: str) -> None:
    """
    将对象保存为pickle文件

    参数:
        obj: 要保存的Python对象
        file_path: 保存路径

    异常:
        IOError: 文件写入失败时抛出
    """
    _ensure_parent_dir(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    """
    从pickle文件加载对象

    参数:
        file_path: pickle文件路径

    返回:
        加载的Python对象

    异常:
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "rb") as f:
        return _SafeUnpickler(f).load()


def save_json(obj: Union[Dict[str, Any], List[Any]], file_path: str, indent: int = 2) -> None:
    """
    将对象保存为JSON文件

    参数:
        obj: 要保存的对象（字典或列表）
        file_path: 保存路径
        indent: 缩进空格数
    """
    _ensure_parent_dir(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    从JSON文件加载对象

    参数:
        file_path: JSON文件路径

    返回:
        加载的对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return data
    raise TypeError("JSON内容必须是字典或列表")


def save_model(model: torch.nn.Module, file_path: str) -> None:
    """
    保存PyTorch模型

    参数:
        model: PyTorch模型
        file_path: 保存路径
    """
    _ensure_parent_dir(file_path)
    torch.save(model.state_dict(), file_path)


def load_model(model: torch.nn.Module, file_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    加载PyTorch模型权重

    参数:
        model: 模型实例
        file_path: 权重文件路径
        device: 加载到的设备

    返回:
        加载权重后的模型
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件不存在: {file_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    load_kwargs: Dict[str, Any] = {"map_location": device}
    if "weights_only" in torch.load.__code__.co_varnames:
        load_kwargs["weights_only"] = True
    model.load_state_dict(torch.load(file_path, **load_kwargs))
    return model


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    保存DataFrame到CSV文件

    参数:
        df: DataFrame对象
        file_path: 保存路径
    """
    _ensure_parent_dir(file_path)
    df.to_csv(file_path, index=False, encoding="utf-8")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    从CSV文件加载DataFrame

    参数:
        file_path: CSV文件路径

    返回:
        DataFrame对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return pd.read_csv(file_path)


def save_numpy(arr: NDArray[Any], file_path: str) -> None:
    """
    保存NumPy数组

    参数:
        arr: NumPy数组
        file_path: 保存路径
    """
    _ensure_parent_dir(file_path)
    np.save(file_path, arr)


def load_numpy(file_path: str) -> NDArray[Any]:
    """
    加载NumPy数组

    参数:
        file_path: 文件路径

    返回:
        NumPy数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return cast(NDArray[Any], np.load(file_path))


def ensure_dir(file_path: str) -> str:
    """
    确保文件所在目录存在

    参数:
        file_path: 文件路径

    返回:
        原文件路径
    """
    _ensure_parent_dir(file_path)
    return file_path


def file_exists(file_path: str) -> bool:
    """
    检查文件是否存在

    参数:
        file_path: 文件路径

    返回:
        是否存在
    """
    return os.path.exists(file_path)


def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）

    参数:
        file_path: 文件路径

    返回:
        文件大小
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return os.path.getsize(file_path)
