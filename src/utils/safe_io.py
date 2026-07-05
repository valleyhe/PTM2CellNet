"""
Safe deserialization utilities for PTM2CellNet.

Provides restricted unpicklers and a safe ``torch.load`` wrapper to guard
against arbitrary code execution during pickle deserialization.  Extracted
from :mod:`src.utils.io` to keep both files under 500 lines.
"""

import importlib
import os
import pickle
from typing import Any, BinaryIO, Dict, List, Optional, Set, Union, cast

import torch

from .logging import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# _load_pickle_archive
# ---------------------------------------------------------------------------

def _load_pickle_archive(file_path: str) -> Dict[str, object]:
    with open(file_path, "rb") as file_obj:
        data = _SafeUnpickler(file_obj).load()
    if not isinstance(data, dict):
        raise TypeError("Pickle fallback content must be a dictionary")
    return cast(Dict[str, object], data)


# ---------------------------------------------------------------------------
# SafeUnpickler — simple module/class allowlist
# ---------------------------------------------------------------------------

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


def safe_pickle_load(file_obj: BinaryIO) -> object:
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
        raise ValueError('Unsafe pickle: ' + str(e)) from e


# ---------------------------------------------------------------------------
# _SafeUnpickler — comprehensive allowlist covering torch/pandas internals
# ---------------------------------------------------------------------------

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
    _PROJECT_SAFE_GLOBALS: Dict[str, Set[str]] = {
        # 示例：若未来需要 pickle 项目数据类，可在此显式声明
        # "src.data.schemas": {"PTMSite", "PTMRecord"},
    }

    @classmethod
    def _all_safe_globals(cls) -> Dict[str, Set[str]]:
        merged: Dict[str, Set[str]] = {}
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


# ---------------------------------------------------------------------------
# safe_torch_load — layered defence for PyTorch checkpoint loading
# ---------------------------------------------------------------------------

# Default allowlist of classes permitted when loading PyTorch checkpoints with
# weights_only=True + allowed_classes.  This is intentionally conservative:
# only tensor storage and fundamental numeric types.  If a checkpoint contains
# additional safe types (e.g. custom config dataclasses), callers should pass
# their own ``allowed_classes`` set.
_TORCH_LOAD_ALLOWED_CLASSES: Set[str] = {
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
_DAVF_SAFE_CLASSES: Optional[List[type]] = None


def safe_torch_load(
    path: Union[str, "os.PathLike[str]", BinaryIO],
    map_location: Optional[Union[str, torch.device]] = None,
    *,
    allowed_classes: Optional[Set[Union[str, type]]] = None,
    enforce_safe_only: bool = True,
    **kwargs: Any,
) -> object:
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

    Environment variables
    ---------------------
    ``PTM2CELLNET_SAFE_LOAD_ONLY`` :
        When set to ``"1"``, ``"true"``, or ``"yes"`` (case-insensitive),
        globally forces ``enforce_safe_only=True`` regardless of the per-call
        argument.  This is a defense-in-depth mechanism for production
        deployments where no unsafe loads should ever occur.

    Returns
    -------
    The loaded PyTorch object.
    """
    # Global hardening: environment variable can force safe-only mode regardless
    # of the per-call enforce_safe_only parameter. This is a defense-in-depth
    # measure for production deployments where no unsafe loads should ever occur.
    _global_safe_only = os.environ.get("PTM2CELLNET_SAFE_LOAD_ONLY", "").lower() in ("1", "true", "yes")
    if _global_safe_only:
        enforce_safe_only = True

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
            "safe_torch_load: \u26a0\ufe0f SECURITY RISK \u2014 weights_only=False requested for '%s'. "
            "This allows arbitrary code execution via pickle deserialization. "
            "Only use with files you trust completely (e.g. self-produced checkpoints). "
            "Prefer weights_only=True with allowed_classes= for non-tensor types.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)

    # Build the merged allowlist for the weights_only=True path.
    merged_classes: Set[Union[str, type]] = set(_TORCH_LOAD_ALLOWED_CLASSES)
    if allowed_classes:
        merged_classes = merged_classes | allowed_classes

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
                    "safe_torch_load: \u26a0\ufe0f SECURITY RISK \u2014 weights_only=True failed for '%s' "
                    "(%s: %s). Falling back to weights_only=False. "
                    "This allows arbitrary code execution via pickle deserialization \u2014 "
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
                "safe_torch_load: \u26a0\ufe0f SECURITY RISK \u2014 weights_only=True failed for '%s' "
                "(%s: %s). Falling back to weights_only=False. "
                "This allows arbitrary code execution via pickle deserialization \u2014 "
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
            "safe_torch_load: \u26a0\ufe0f SECURITY RISK \u2014 weights_only=True rejected types in '%s' "
            "(ValueError: %s). Falling back to weights_only=False. "
            "This allows arbitrary code execution via pickle deserialization \u2014 "
            "only load files you trust completely. "
            "Consider passing allowed_classes= with the required types.",
            path, exc,
        )
        return torch.load(
            path, map_location=map_location, weights_only=False, **kwargs,
        )
