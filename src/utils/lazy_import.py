from typing import Any, Generic, TypeVar, Optional
import importlib

T = TypeVar("T")


class LazyImport(Generic[T]):
    """Lazily import a module/attribute, providing type-safe access when available."""

    def __init__(self, module_path: str, attr_name: Optional[str] = None) -> None:
        self._module_path = module_path
        self._attr_name = attr_name
        self._module: Any = None
        self._import_error: Optional[ImportError] = None
        self._imported = False

    def _do_import(self) -> None:
        if self._imported:
            return
        self._imported = True
        try:
            self._module = importlib.import_module(self._module_path)
        except ImportError as e:
            self._import_error = e

    def get(self) -> Any:
        self._do_import()
        if self._import_error is not None:
            raise self._import_error
        if self._attr_name is not None and self._module is not None:
            return getattr(self._module, self._attr_name)
        return self._module

    def is_available(self) -> bool:
        self._do_import()
        return self._import_error is None and self._module is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.get()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        mod = self.get()
        return getattr(mod, name)
