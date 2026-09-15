"""
文件IO工具模块
功能概述: 提供统一的文件读写接口，支持多种格式
设计思路: 封装常见文件操作，提供类型安全的接口，自动处理目录创建
"""

import json
import os
import pickle
from typing import Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from typing_extensions import TypedDict

from .logging import setup_logger
from .safe_io import _SafeUnpickler, _load_pickle_archive, safe_pickle_load, safe_torch_load  # noqa: F401

try:
    import h5py
except ImportError:  # pragma: no cover - exercised via monkeypatch
    h5py = None

logger = setup_logger(__name__)


def _ensure_parent_dir(file_path: str) -> None:
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _warn_hdf5_fallback() -> None:
    logger.warning("h5py is not available; falling back to a pickled archive for HDF5 IO")


# Typed structure for HDF5/JSON serialisable dictionaries.
# Keys are arbitrary strings; values are limited to the types supported
# by ``_write_hdf5_value`` and ``save_json``.
class SerializationConfig(TypedDict, total=False):
    """TypedDict for serialisable dictionary payloads.

    Only the value types that ``_write_hdf5_value`` and ``save_json``
    can handle are listed.  Keys are dynamic so this serves as
    documentation rather than an exhaustive contract.
    """

    pass  # Keys are dynamic; this marks the shape for readers.


# Union of value types accepted by HDF5 serialisation helpers.
Hdf5Value = Union[torch.Tensor, pd.DataFrame, np.ndarray, str, np.generic, int, float, bool]


def _is_string_array(array: NDArray[np.str_]) -> bool:
    if array.dtype.kind in {"U", "S"}:
        return True
    if array.dtype.kind != "O":
        return False
    flat = array.reshape(-1)
    return all(isinstance(item, str) for item in flat)


def _create_string_dataset(group: "h5py.Group", key: str, value: Union[str, NDArray[np.str_]]) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to create string datasets")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    data = value if isinstance(value, str) else np.asarray(value, dtype=object)
    group.create_dataset(key, data=data, dtype=string_dtype)


def _write_hdf5_value(group: "h5py.Group", key: str, value: Hdf5Value) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 serialization")

    if isinstance(value, torch.Tensor):
        dataset = group.create_dataset(key, data=value.detach().cpu().numpy())
        dataset.attrs["item_type"] = "torch_tensor"
        return

    if isinstance(value, pd.DataFrame):
        dataframe_group = group.create_group(key)
        dataframe_group.attrs["item_type"] = "dataframe"
        _create_string_dataset(dataframe_group, "__columns__", np.asarray(value.columns.astype(str).tolist()))
        _write_hdf5_value(dataframe_group, "__index__", value.index.to_numpy())
        columns_group = dataframe_group.create_group("columns")
        for column in value.columns:
            _write_hdf5_value(columns_group, str(column), value[column].to_numpy())
        return

    if isinstance(value, str):
        _create_string_dataset(group, key, value)
        group[key].attrs["item_type"] = "string"
        return

    if isinstance(value, np.ndarray):
        if _is_string_array(value):
            _create_string_dataset(group, key, value)
        else:
            group.create_dataset(key, data=value)
        group[key].attrs["item_type"] = "ndarray"
        return

    if isinstance(value, (np.generic, int, float, bool)):
        dataset = group.create_dataset(key, data=value)
        dataset.attrs["item_type"] = "scalar"
        return

    raise TypeError(f"Unsupported HDF5 value type for key '{key}': {type(value)!r}")


def _read_hdf5_dataset(dataset: "h5py.Dataset") -> Union[str, np.ndarray, torch.Tensor, np.generic, int, float, bool]:
    item_type = dataset.attrs.get("item_type", "ndarray")

    if item_type == "string":
        return cast(str, dataset.asstr()[()])

    if dataset.dtype.kind in {"O", "S"}:
        return np.asarray(dataset.asstr()[()])

    value = dataset[()]
    if item_type == "torch_tensor":
        return torch.from_numpy(np.asarray(value))
    if item_type == "scalar":
        return cast(
            Union[str, np.ndarray, torch.Tensor, np.generic, int, float, bool],
            value.item() if hasattr(value, "item") else value,
        )
    return np.asarray(value)


def _read_hdf5_value(
    node: Union["h5py.Dataset", "h5py.Group"],
) -> Union[str, np.ndarray, torch.Tensor, pd.DataFrame, np.generic, int, float, bool]:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 deserialization")

    if isinstance(node, h5py.Dataset):
        return _read_hdf5_dataset(node)

    item_type = node.attrs.get("item_type")
    if item_type != "dataframe":
        raise TypeError(f"Unsupported HDF5 group type: {item_type!r}")

    columns = [str(column) for column in node["__columns__"].asstr()[()]]
    index_values = cast(pd.Index, _read_hdf5_value(node["__index__"]))
    column_group = node["columns"]
    data = {column: _read_hdf5_value(column_group[column]) for column in columns}
    return pd.DataFrame(data, index=index_values)


def save_pickle(obj: object, file_path: str) -> None:
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


def load_pickle(file_path: str) -> object:
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


def save_hdf5(obj: Dict[str, Hdf5Value], file_path: str) -> None:
    """
    将字典对象保存为HDF5文件。

    支持NumPy数组、PyTorch张量、Pandas DataFrame和字符串；
    当 ``h5py`` 不可用时回退为pickle归档并记录warning。
    """
    _ensure_parent_dir(file_path)

    if h5py is None:
        _warn_hdf5_fallback()
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        return

    with h5py.File(file_path, "w") as handle:
        for key, value in obj.items():
            _write_hdf5_value(handle, key, value)


def load_hdf5(file_path: str) -> Dict[str, Hdf5Value]:
    """
    从HDF5文件加载字典对象。

    若 ``h5py`` 不可用，或文件是由pickle fallback生成，则回退到pickle读取并记录warning。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if h5py is None:
        _warn_hdf5_fallback()
        return cast(Dict[str, Hdf5Value], _load_pickle_archive(file_path))

    try:
        with h5py.File(file_path, "r") as handle:
            return {key: _read_hdf5_value(handle[key]) for key in handle.keys()}
    except OSError:
        logger.warning("File is not a valid HDF5 archive; falling back to pickle loading")
        return cast(Dict[str, Hdf5Value], _load_pickle_archive(file_path))


def save_json(obj: Union[Dict[str, object], List[object]], file_path: str, indent: int = 2) -> None:
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


def load_json(file_path: str) -> Union[Dict[str, object], List[object]]:
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


def load_model(
    model: torch.nn.Module,
    file_path: str,
    device: Optional[str] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """
    加载PyTorch模型权重

    统一 checkpoint 合约（审计报告 P0-3）：自动识别并解析裸 ``state_dict``、
    Lightning ``.ckpt``（含 ``state_dict`` + ``model.`` 前缀）以及旧格式
    ``model_state_dict``，使推理 CLI 可直接消费任意训练入口的产物。

    参数:
        model: 模型实例
        file_path: 权重文件路径（``.pt``/``.pth``/``.ckpt``）
        device: 加载到的设备
        strict: 是否严格加载（默认 True）。不匹配时抛出 ``RuntimeError``。

    返回:
        加载权重后的模型

    异常:
        FileNotFoundError: 权重文件不存在。
        RuntimeError: strict=True 且 missing/unexpected 键非空。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"模型文件不存在: {file_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 延迟导入避免循环依赖
    from .checkpoint_utils import extract_model_state_dict

    raw = safe_torch_load(file_path, map_location=device)
    state_dict = extract_model_state_dict(raw)

    if strict:
        # 严格模式：直接 load_state_dict，missing/unexpected 会触发 RuntimeError
        model.load_state_dict(state_dict)
    else:
        # 兼容模式：记录但不报错（仅用于显式声明的迁移路径）
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("load_model(strict=False): 缺失键 %s", missing[:5])
        if unexpected:
            logger.warning("load_model(strict=False): 多余键 %s", unexpected[:5])

    return model


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    保存DataFrame到文件

    支持自动识别 HDF5 文件（.h5/.hdf5 后缀），通过 ``df.to_hdf`` 写出，
    将 HDF5 接入主数据写出链路；其他后缀按 CSV 处理。

    参数:
        df: DataFrame对象
        file_path: 保存路径（.csv/.h5/.hdf5）
    """
    _ensure_parent_dir(file_path)

    # HDF5 分支：将 HDF5 接入主数据写出链路（路径后缀驱动）
    if file_path.lower().endswith((".h5", ".hdf5")):
        if h5py is None:
            # h5py 不可用时回退到项目 save_hdf5（其内部会 pickle fallback）
            save_hdf5({"dataframe": df}, file_path)
            return
        try:
            df.to_hdf(file_path, key="dataframe", mode="w", format="table")
        except (ImportError, ValueError, KeyError) as exc:
            # to_hdf 需要 pytables；若不可用，回退到项目 save_hdf5
            logger.warning(
                "df.to_hdf 失败 (%s)，尝试使用 save_hdf5 回退",
                exc,
            )
            save_hdf5({"dataframe": df}, file_path)
        return

    df.to_csv(file_path, index=False, encoding="utf-8")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    从文件加载DataFrame

    支持自动识别 HDF5 文件（.h5/.hdf5 后缀），通过 ``pd.read_hdf`` 读取，
    将 HDF5 接入主数据加载链路；其他后缀按 CSV 处理。

    参数:
        file_path: 数据文件路径（.csv/.h5/.hdf5）

    返回:
        DataFrame对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # HDF5 分支：将 HDF5 接入主数据加载链路（路径后缀驱动）
    if file_path.lower().endswith((".h5", ".hdf5")):
        try:
            return cast(pd.DataFrame, pd.read_hdf(file_path))
        except (ImportError, ValueError, KeyError) as exc:
            # read_hdf 需要 pytables；若不可用或 key 不匹配，回退到 load_hdf5
            logger.warning(
                "pd.read_hdf 失败 (%s)，尝试使用 load_hdf5 回退",
                exc,
            )
            data = load_hdf5(file_path)
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, dict) and len(data) == 1:
                value = next(iter(data.values()))
                if isinstance(value, pd.DataFrame):
                    return value
            raise TypeError(f"HDF5 文件内容无法解析为 DataFrame: {file_path}") from None

    return pd.read_csv(file_path)


def save_numpy(arr: NDArray[np.generic], file_path: str) -> None:
    """
    保存NumPy数组

    参数:
        arr: NumPy数组
        file_path: 保存路径
    """
    _ensure_parent_dir(file_path)
    np.save(file_path, arr)


def load_numpy(file_path: str) -> NDArray[np.generic]:
    """
    加载NumPy数组

    参数:
        file_path: 文件路径

    返回:
        NumPy数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    return cast(NDArray[np.generic], np.load(file_path))


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
