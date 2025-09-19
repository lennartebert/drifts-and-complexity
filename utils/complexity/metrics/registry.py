from __future__ import annotations
from typing import Dict, Type, Iterable
import importlib, pkgutil, sys

from utils.complexity.metrics.metric import Metric

_METRIC_REGISTRY: Dict[str, Type[Metric]] = {}
_DISCOVERED = False

def register_metric(name: str):
    def _wrap(cls: Type[Metric]) -> Type[Metric]:
        _METRIC_REGISTRY[name] = cls
        return cls
    return _wrap

def available_metric_names() -> Iterable[str]:
    discover_metrics()  # ensure populated
    return tuple(_METRIC_REGISTRY.keys())

def get_metric_class(name: str) -> Type[Metric]:
    discover_metrics()
    return _METRIC_REGISTRY[name]

def discover_metrics(package: str = "utils.complexity.metrics") -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        return  # not a package

    # Walk the whole metrics package tree and import submodules
    for finder, modname, ispkg in pkgutil.walk_packages(pkg.__path__, package + "."):
        # Skip private/dunder modules
        if any(part.startswith("_") for part in modname.split(".")):
            continue
        try:
            importlib.import_module(modname)
        except Exception as e:
            # Optional: log or store the error; don’t crash discovery
            # print(f"[metrics.discovery] failed to import {modname}: {e}")
            pass