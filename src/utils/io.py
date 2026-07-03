"""
文件IO工具模块
功能概述: 提供统一的文件读写接口，支持多种格式
设计思路: 封装常见文件操作，提供类型安全的接口，自动处理目录创建
"""

import importlib
import json
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Set, Union, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from .logging import setup_logger

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


def _is_string_array(array: NDArray[Any]) -> bool:
    if array.dtype.kind in {"U", "S"}:
        return True
    if array.dtype.kind != "O":
        return False
    flat = array.reshape(-1)
    return all(isinstance(item, str) for item in flat)


def _create_string_dataset(group: Any, key: str, value: Any) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required to create string datasets")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    data = value if isinstance(value, str) else np.asarray(value, dtype=object)
    group.create_dataset(key, data=data, dtype=string_dtype)


def _write_hdf5_value(group: Any, key: str, value: Any) -> None:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 serialization")

    if isinstance(value, torch.Tensor):
        dataset = group.create_dataset(key, data=value.detach().cpu().numpy())
        dataset.attrs["item_type"] = "torch_tensor"
        return

    if isinstance(value, pd.DataFrame):
        dataframe_group = group.create_group(key)
        dataframe_group.attrs["item_type"] = "dataframe"
        _create_string_dataset(dataframe_group, "__columns__", value.columns.astype(str).tolist())
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


def _read_hdf5_dataset(dataset: Any) -> Any:
    item_type = dataset.attrs.get("item_type", "ndarray")

    if item_type == "string":
        return dataset.asstr()[()]

    if dataset.dtype.kind in {"O", "S"}:
        return np.asarray(dataset.asstr()[()])

    value = dataset[()]
    if item_type == "torch_tensor":
        return torch.from_numpy(np.asarray(value))
    if item_type == "scalar":
        return value.item() if hasattr(value, "item") else value
    return np.asarray(value)


def _read_hdf5_value(node: Any) -> Any:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 deserialization")

    if isinstance(node, h5py.Dataset):
        return _read_hdf5_dataset(node)

    item_type = node.attrs.get("item_type")
    if item_type != "dataframe":
        raise TypeError(f"Unsupported HDF5 group type: {item_type!r}")

    columns = [str(column) for column in node["__columns__"].asstr()[()]]
    index_values = _read_hdf5_value(node["__index__"])
    column_group = node["columns"]
    data = {column: _read_hdf5_value(column_group[column]) for column in columns}
    return pd.DataFrame(data, index=pd.Index(index_values))


def _load_pickle_archive(file_path: str) -> Dict[str, Any]:
    with open(file_path, "rb") as file_obj:
        data = _SafeUnpickler(file_obj).load()
    if not isinstance(data, dict):
        raise TypeError("Pickle fallback content must be a dictionary")
    return cast(Dict[str, Any], data)


_SAFE_MODULES: Set[str] = {'builtins', 'collections', 'typing', 'numpy', 'torch'}
_SAFE_CLASSES: Set[str] = {'dict', 'list', 'tuple', 'str', 'int', 'float', 'bool',
                           'OrderedDict', 'defaultdict', 'Counter', 'ndarray', 'DataFrame'}


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows known-safe modules and classes."""

    def find_class(self, module: str, name: str):
        mod = module.split('.')[0]
        if mod not in _SAFE_MODULES:
            raise pickle.UnpicklingError('Unsafe module: ' + module)
        if name not in _SAFE_CLASSES:
            raise pickle.UnpicklingError('Unsafe class: ' + module + '.' + name)
        return super().find_class(module, name)


def safe_pickle_load(file_obj: Any) -> Any:
    """Load pickle data using SafeUnpickler for restricted deserialization.

    Parameters
    ----------
    file_obj :
        Binary file-like object to load from.

    Returns
    -------
    The deserialized Python object.

    Raises
    ------
    ValueError
        If the pickle contains disallowed modules or classes.
    """
    try:
        return SafeUnpickler(file_obj).load()
    except pickle.UnpicklingError as e:
        raise ValueError('Unsafe pickle: ' + str(e))


class _SafeUnpickler(pickle.Unpickler):
    """受控反序列化器：仅允许已知安全或显式声明的全局对象。

    白名单覆盖：
        - Python 内置容器/标量；
        - NumPy 数组与 dtype（含 ``numpy.core.multiarray._reconstruct``、
          ``numpy.core.multiarray.scalar``）；
        - PyTorch 张量（``torch._utils._rebuild_tensor_v2``）；
        - Pandas DataFrame / Series / Index；
        - 项目自定义数据类（按需扩展 ``_PROJECT_SAFE_GLOBALS``）。
    """

    # 内置与常用库类型白名单
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
            "complex",
            "range",
            "slice",
            "object",
        },
        "numpy": {"dtype", "ndarray", "scalar"},
        # NumPy < 2.0 路径
        "numpy.core.multiarray": {"_reconstruct", "scalar"},
        "numpy.core.numeric": {"_frombuffer"},
        # NumPy >= 2.0 路径
        "numpy._core.multiarray": {"_reconstruct", "scalar"},
        "numpy._core.numeric": {"_frombuffer"},
        "torch._utils": {"_rebuild_tensor_v2", "_rebuild_parameter"},
        "torch.storage": {"_load_from_bytes", "_TypedStorage"},
        "pandas.core.frame": {"DataFrame"},
        "pandas.core.series": {"Series"},
        "pandas.core.indexes.base": {"Index", "_new_Index", "_UnfavorableIndex"},
        "pandas.core.indexes.numeric": {"Int64Index", "Float64Index", "UInt64Index"},
        "pandas.core.indexes.range": {"RangeIndex"},
        "pandas.core.internals.managers": {"BlockManager", "SingleBlockManager"},
        "pandas.core.internals.blocks": {"new_block"},
        "pandas.core.internals": {"new_block"},
        # Pandas C 扩展内部模块（用于 BlockManager 反序列化）
        "pandas._libs.internals": {"_unpickle_block"},
        "pandas": {"Timestamp"},
        "_codecs": {"encode"},
        "datetime": {"datetime", "date", "timedelta"},
        "collections": {"OrderedDict", "defaultdict", "Counter"},
    }

    # 项目自定义类白名单（按需扩展）。使用 ``module: {name}`` 形式。
    _PROJECT_SAFE_GLOBALS: Dict[str, set] = {
        # 示例：若未来需要 pickle 项目数据类，可在此显式声明
        # "src.data.schemas": {"PTMSite", "PTMRecord"},
    }

    @classmethod
    def _all_safe_globals(cls) -> Dict[str, set]:
        merged: Dict[str, set] = {}
        for source in (cls._SAFE_GLOBALS, cls._PROJECT_SAFE_GLOBALS):
            for module, names in source.items():
                merged.setdefault(module, set()).update(names)
        return merged

    def find_class(self, module: str, name: str):
        # Python 2 pickle 使用 "builtin"/"__builtin__" 作为 builtins 模块名，
        # 在 protocol 2 的历史 pickle 中仍可能出现（例如 slice 对象）。
        if module in {"__builtin__", "builtin"} and name in {"slice", "object"}:
            return getattr(importlib.import_module("builtins"), name)
        safe_globals = self._all_safe_globals()
        if module in safe_globals and name in safe_globals[module]:
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


def load_pickle(file_path: str, safe: bool = True) -> Any:
    """
    从pickle文件加载对象

    参数:
        file_path: pickle文件路径
        safe: 是否使用安全反序列化（白名单模式）。默认为 ``True``，
            仅允许已知安全类型（内置容器、NumPy/Torch/Pandas 等）。
            若需要加载自定义类实例等不在白名单内的对象，可显式传入
            ``safe=False`` 使用标准 ``pickle.load``——仅对可信文件使用。

    返回:
        加载的Python对象

    异常:
        FileNotFoundError: 文件不存在时抛出
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not safe:
        logger.warning(
            "load_pickle(safe=False) 使用标准 pickle.load 加载 %s，"
            "仅应用于可信文件以避免反序列化风险。", file_path,
        )
        with open(file_path, "rb") as f:
            return pickle.load(f)

    with open(file_path, "rb") as f:
        return _SafeUnpickler(f).load()


def save_hdf5(obj: Dict[str, Any], file_path: str) -> None:
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


def load_hdf5(file_path: str) -> Dict[str, Any]:
    """
    从HDF5文件加载字典对象。

    若 ``h5py`` 不可用，或文件是由pickle fallback生成，则回退到pickle读取并记录warning。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if h5py is None:
        _warn_hdf5_fallback()
        return _load_pickle_archive(file_path)

    try:
        with h5py.File(file_path, "r") as handle:
            return {key: _read_hdf5_value(handle[key]) for key in handle.keys()}
    except OSError:
        logger.warning("File is not a valid HDF5 archive; falling back to pickle loading")
        return _load_pickle_archive(file_path)


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


# Default allowlist of classes permitted when loading PyTorch checkpoints with
# weights_only=True + allowed_classes.  This is intentionally conservative:
# only tensor storage and fundamental numeric types.  If a checkpoint contains
# additional safe types (e.g. custom config dataclasses), callers should pass
# their own ``allowed_classes`` set.
_TORCH_LOAD_ALLOWED_CLASSES: set = {
    # Tensor internals — required for any state_dict / optimizer checkpoint
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter",
    "torch.storage._TypedStorage",
    "torch.storage._UntypedStorage",
    # NumPy scalars (common in optimizer state)
    "numpy.core.multiarray.scalar",
    "numpy._core.multiarray.scalar",
    # Basic Python types that PyTorch itself may pickle
    "builtins.dict",
    "builtins.list",
    "builtins.tuple",
    "builtins.set",
    "builtins.frozenset",
    "builtins.str",
    "builtins.int",
    "builtins.float",
    "builtins.bool",
    "builtins.bytes",
    "builtins.NoneType",
    "collections.OrderedDict",
}

# Classes that are allowed in torch.load(weights_only=True, allowed_classes=...)
# for DAVF-related legacy checkpoints.  These must be actual class objects
# (not strings), so the list is populated lazily below.
_DAVF_SAFE_CLASSES: Optional[list] = None


def safe_torch_load(
    path: Any,
    map_location: Any = None,
    *,
    allowed_classes: Optional[set] = None,
    enforce_safe_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a PyTorch artifact with a safe-by-default ``weights_only`` strategy.

    Security model
    --------------
    ``torch.load`` uses Python's ``pickle`` under the hood, which can execute
    arbitrary code during deserialization.  This helper enforces a layered
    defence:

    1. **Prefer ``weights_only=True``** — PyTorch restricts unpickling to only
       tensor-related types.  This is the default and should be used for all
       model weights / state dicts produced by this project.
    2. **``allowed_classes`` allowlist** — When the PyTorch version supports it
       (>= 2.0), an explicit set of permitted class names is forwarded so that
       even ``weights_only=True`` will only reconstruct known-safe types.
       Callers can extend this set for their own safe types.
    3. **Fallback with explicit opt-in** — If ``weights_only=True`` fails
       (e.g. the checkpoint contains optimiser state with NumPy scalars not
       covered by the default allowlist), the caller must pass
       ``weights_only=False`` explicitly.  An automatic fallback **is still
       provided** for backwards-compatibility, but it logs a warning at
       ``logger.warning`` level so that every unsafe load is auditable.
    4. **``enforce_safe_only`` (default ``True``)** — The automatic fallback to
       ``weights_only=False`` is **disabled by default**: if
       ``weights_only=True`` fails, the original exception is re-raised instead
       of silently falling back to unsafe pickle deserialization.  This makes
       the loader safe-by-default for untrusted inputs, where a failed safe
       load must never degrade into arbitrary-code-execution.  Callers loading
       fully-trusted legacy checkpoints that cannot be parsed with
       ``weights_only=True`` may opt out by passing ``enforce_safe_only=False``.

    **Do not** use ``weights_only=False`` for files from untrusted sources.

    Parameters
    ----------
    path :
        File path or file-like object to load.
    map_location :
        Passed through to ``torch.load``.
    allowed_classes :
        Optional set of fully-qualified class names (``"module.path.ClassName"``)
        to allow when ``weights_only=True``.  Merged with the built-in
        ``_TORCH_LOAD_ALLOWED_CLASSES`` allowlist.  Ignored if the installed
        PyTorch version does not support the ``allowed_classes`` argument.
    enforce_safe_only :
        If ``True`` (default), never fall back to ``weights_only=False``.  Any
        failure of the safe ``weights_only=True`` path propagates the exception
        to the caller instead of silently degrading into unsafe pickle
        deserialization.  This is the safe-by-default posture: untrusted inputs
        can never trigger arbitrary-code-execution.  Callers loading trusted,
        self-produced legacy checkpoints that genuinely cannot be parsed with
        ``weights_only=True`` (e.g. they embed custom config dataclasses) may
        opt out by passing ``enforce_safe_only=False`` — but only for files
        whose provenance is fully trusted.
    **kwargs :
        Additional keyword arguments forwarded to ``torch.load``.

    Returns
    -------
    The loaded PyTorch object.
    """
    explicit_weights_only = kwargs.pop("weights_only", None)

    # enforce_safe_only forbids any weights_only=False path — including an
    # explicit opt-in.  This guarantees the loader can never degrade into
    # arbitrary-code-execution pickle deserialization, which is the whole point
    # of using it for untrusted inputs.
    if enforce_safe_only and explicit_weights_only is False:
        raise ValueError(
            "safe_torch_load: weights_only=False is forbidden when "
            "enforce_safe_only=True (path: %s). Load the file with "
            "weights_only=True + allowed_classes= instead, or load a "
            "trusted file without enforce_safe_only." % (path,)
        )

    # If caller explicitly opts into unsafe loading, honour it but log loudly.
    if explicit_weights_only is False:
        logger.warning(
            "safe_torch_load: ⚠️ SECURITY RISK — weights_only=False requested for '%s'. "
            "This allows arbitrary code execution via pickle deserialization. "
            "Only use with files you trust completely (e.g. self-produced checkpoints). "
            "Prefer weights_only=True with allowed_classes= for non-tensor types.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)

    # Build the merged allowlist for the weights_only=True path.
    merged_classes = _TORCH_LOAD_ALLOWED_CLASSES
    if allowed_classes:
        merged_classes = _TORCH_LOAD_ALLOWED_CLASSES | allowed_classes

    # Attempt safe load with allowlist (PyTorch >= 2.0 supports allowed_classes).
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
            allowed_classes=merged_classes,
            **kwargs,
        )
    except TypeError as exc:
        # TypeError: either PyTorch doesn't support allowed_classes, or
        # weights_only keyword is unavailable (very old PyTorch).
        # Try again without allowed_classes.
        if "allowed_classes" in str(exc) or "unexpected keyword" in str(exc):
            logger.debug(
                "safe_torch_load: 'allowed_classes' not supported by this PyTorch "
                "version; retrying without it. Path: %s", path,
            )
            try:
                return torch.load(
                    path, map_location=map_location, weights_only=True, **kwargs,
                )
            except (TypeError, ValueError) as inner_exc:
                # weights_only=True itself may be unsupported or the checkpoint
                # contains types outside the default allowlist.
                if enforce_safe_only:
                    logger.error(
                        "safe_torch_load: weights_only=True failed for '%s' "
                        "(%s: %s) and enforce_safe_only=True forbids the "
                        "weights_only=False fallback; re-raising.",
                        path, type(inner_exc).__name__, inner_exc,
                    )
                    raise
                logger.warning(
                    "safe_torch_load: ⚠️ SECURITY RISK — weights_only=True failed for '%s' "
                    "(%s: %s). Falling back to weights_only=False. "
                    "This allows arbitrary code execution via pickle deserialization — "
                    "only load files you trust completely. "
                    "If this checkpoint contains safe types not in the default "
                    "allowlist, pass them via allowed_classes= or load with "
                    "weights_only=False explicitly.",
                    path, type(inner_exc).__name__, inner_exc,
                )
                return torch.load(
                    path, map_location=map_location, weights_only=False, **kwargs,
                )
        # Other TypeError (not about allowed_classes) — try plain weights_only=True.
        try:
            return torch.load(
                path, map_location=map_location, weights_only=True, **kwargs,
            )
        except (TypeError, ValueError) as inner_exc:
            if enforce_safe_only:
                logger.error(
                    "safe_torch_load: weights_only=True failed for '%s' "
                    "(%s: %s) and enforce_safe_only=True forbids the "
                    "weights_only=False fallback; re-raising.",
                    path, type(inner_exc).__name__, inner_exc,
                )
                raise
            logger.warning(
                "safe_torch_load: ⚠️ SECURITY RISK — weights_only=True failed for '%s' "
                "(%s: %s). Falling back to weights_only=False. "
                "This allows arbitrary code execution via pickle deserialization — "
                "only load files you trust completely.",
                path, type(inner_exc).__name__, inner_exc,
            )
            return torch.load(
                path, map_location=map_location, weights_only=False, **kwargs,
            )
    except ValueError as exc:
        # ValueError from weights_only=True: checkpoint contains types not on the
        # allowlist (e.g. NumPy scalars in optimizer state).  This is a signal
        # that the caller should either extend allowed_classes or explicitly
        # opt into unsafe loading.
        if enforce_safe_only:
            logger.error(
                "safe_torch_load: weights_only=True rejected types in '%s' "
                "(ValueError: %s) and enforce_safe_only=True forbids the "
                "weights_only=False fallback; re-raising.",
                path, exc,
            )
            raise
        logger.warning(
            "safe_torch_load: ⚠️ SECURITY RISK — weights_only=True rejected types in '%s' "
            "(ValueError: %s). Falling back to weights_only=False. "
            "This allows arbitrary code execution via pickle deserialization — "
            "only load files you trust completely. "
            "Consider passing allowed_classes= with the required types.",
            path, exc,
        )
        return torch.load(
            path, map_location=map_location, weights_only=False, **kwargs,
        )


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
                "df.to_hdf 失败 (%s)，尝试使用 save_hdf5 回退", exc,
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
            return pd.read_hdf(file_path)
        except (ImportError, ValueError, KeyError) as exc:
            # read_hdf 需要 pytables；若不可用或 key 不匹配，回退到 load_hdf5
            logger.warning(
                "pd.read_hdf 失败 (%s)，尝试使用 load_hdf5 回退", exc,
            )
            data = load_hdf5(file_path)
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, dict) and len(data) == 1:
                value = next(iter(data.values()))
                if isinstance(value, pd.DataFrame):
                    return value
            raise TypeError(
                f"HDF5 文件内容无法解析为 DataFrame: {file_path}"
            )

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
