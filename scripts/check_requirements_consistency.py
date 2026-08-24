#!/usr/bin/env python
"""Check requirements-*.txt interval constraints against requirements-lock.txt pins.

TD-N-32 / TD-N-34（2026-08-24）：多轨 requirements 使用 ``>=`` 区间、lock 使用
``==`` 钉扎，二者之间没有任何一致性校验，历史上已发生 lock 与声明漂移
（如 scgpt 与 scvi-tools>=1.2 互斥却同时留在 lock 中）。本脚本在 CI 与本地
提供同一入口，规则：

1. core 依赖必须在 lock 中存在，且钉扎版本满足声明区间（违反即失败）；
2. 可选轨（pretrained / mamba / analysis）依赖若出现在 lock 中，钉扎版本必须
   满足声明区间；不在 lock 中视为有意不装（可选能力），仅跳过；
3. lock 中允许存在声明轨之外的传递依赖（由 pip 解析产生），不校验。

用法::

    python scripts/check_requirements_consistency.py            # 默认路径
    python scripts/check_requirements_consistency.py --strict-optional
        # 可选轨依赖也必须在 lock 中存在（当前不启用）

退出码：0 = 一致；1 = 存在冲突（stdout 逐条列出）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent

CORE_FILES = [REPO_ROOT / "requirements-core.txt"]
OPTIONAL_FILES = [
    REPO_ROOT / "requirements-pretrained.txt",
    REPO_ROOT / "requirements-mamba.txt",
    REPO_ROOT / "requirements-analysis.txt",
]
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"


def parse_requirements(path: Path) -> list[Requirement]:
    """Parse one requirements file, recursing into ``-r`` includes."""
    requirements: list[Requirement] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            included = REPO_ROOT / line.split(maxsplit=1)[1].strip()
            requirements.extend(parse_requirements(included))
            continue
        if line.startswith("-"):
            raise ValueError(f"{path.name}: unsupported option line: {raw_line!r}")
        requirements.append(Requirement(line))
    return requirements


def parse_lock(path: Path) -> dict[str, Version]:
    """Parse ``pkg==version`` pins into a canonicalized-name map."""
    pins: dict[str, Version] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"{path.name}: expected 'pkg==version' pin, got: {raw_line!r}")
        name, _, version = line.partition("==")
        pins[canonicalize_name(name.strip())] = Version(version.strip())
    return pins


def check_track(
    track_name: str,
    requirements: list[Requirement],
    pins: dict[str, Version],
    *,
    require_presence: bool,
) -> list[str]:
    """Return a list of human-readable violation messages for one track."""
    errors: list[str] = []
    seen: set[str] = set()
    for req in requirements:
        key = canonicalize_name(req.name)
        if key in seen:
            continue
        seen.add(key)
        pin = pins.get(key)
        if pin is None:
            if require_presence:
                errors.append(f"[{track_name}] '{req.name}' missing from lock")
            continue
        if req.specifier and not req.specifier.contains(pin, prereleases=True):
            errors.append(
                f"[{track_name}] '{req.name}' pin {pin} violates '{req.specifier}'"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lock",
        type=Path,
        default=LOCK_FILE,
        help="Path to the pin file (default: requirements-lock.txt)",
    )
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="Also require optional-track dependencies to be present in the lock",
    )
    args = parser.parse_args()

    pins = parse_lock(args.lock)
    errors: list[str] = []
    for core_file in CORE_FILES:
        errors.extend(
            check_track(
                core_file.name,
                parse_requirements(core_file),
                pins,
                require_presence=True,
            )
        )
    for optional_file in OPTIONAL_FILES:
        errors.extend(
            check_track(
                optional_file.name,
                parse_requirements(optional_file),
                pins,
                require_presence=args.strict_optional,
            )
        )

    if errors:
        print(f"requirements/lock consistency FAILED ({len(errors)} issue(s)):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"requirements/lock consistency OK: {len(pins)} lock pins satisfy all "
        "core constraints (optional tracks checked where present)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
