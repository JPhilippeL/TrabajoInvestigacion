"""Shared helpers for the DEAttentionDTA GUI integration.

This module deliberately treats the original repository as an external source
of truth.  Upstream model files are loaded dynamically; they are not copied or
rewritten by the GUI layer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent
RUNNERS_ROOT = MODULE_ROOT / "core"

_MODULE_CACHE: dict[str, ModuleType] = {}


def resolve_project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a GUI path against the application root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def ensure_dir(value: str | Path) -> Path:
    """Create and return a directory."""
    path = resolve_project_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON file with stable formatting."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return output


def import_module_from_file(module_name: str, path: str | Path) -> ModuleType:
    """Load a Python module from a file and cache the loaded object."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Python module not found: {source}")
    cache_key = f"{module_name}:{source}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def load_prepare_runner() -> ModuleType:
    return import_module_from_file(
        "deattentiondta_prepare_urv",
        RUNNERS_ROOT / "Prepare_URV_Positions_From_V2_Dataset.py",
    )


def load_base_runner() -> ModuleType:
    return import_module_from_file(
        "deattentiondta_run_urv_5splits",
        RUNNERS_ROOT / "Run_URV_5Splits.py",
    )


def load_finetune_runner() -> ModuleType:
    return import_module_from_file(
        "deattentiondta_run_urv_finetune",
        RUNNERS_ROOT / "Run_URV_Finetune_Pretrained.py",
    )


def as_absolute_string(value: str | Path, *, must_exist: bool = False) -> str:
    return str(resolve_project_path(value, must_exist=must_exist))
