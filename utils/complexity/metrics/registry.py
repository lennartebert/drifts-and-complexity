from __future__ import annotations
from typing import Dict, Type, Iterable, List, Callable
import importlib
import pkgutil

from utils.complexity.metrics.metric import Metric

# Public metric name -> list of implementing classes (variants)
_METRIC_REGISTRY: Dict[str, List[Type[Metric]]] = {}
_DISCOVERED = False


def register_metric(name: str) -> Callable[[Type[Metric]], Type[Metric]]:
    """
    Decorator to register a Metric class under a public *name*.

    Supports multiple implementations (variants) per name. This enables
    registering both a trace-based and a distribution-based class under
    the same metric name. The discovery mechanism will import modules so
    that decorated classes get registered.

    Parameters
    ----------
    name : str
        The public display/lookup name of the metric (e.g., "Number of Traces").

    Returns
    -------
    Callable[[Type[Metric]], Type[Metric]]
        A decorator that registers the given class and returns it unchanged.
    """
    def _wrap(cls: Type[Metric]) -> Type[Metric]:
        # Ensure a list exists for this name
        bucket = _METRIC_REGISTRY.setdefault(name, [])
        # Avoid duplicate registration of the same class
        if cls not in bucket:
            bucket.append(cls)
        # Ensure the class has a name attribute
        if not getattr(cls, "name", None):
            cls.name = name  # type: ignore[attr-defined]
        return cls
    return _wrap


def available_metric_names() -> Iterable[str]:
    """
    Return an iterable of registered public metric names.

    Discovery is triggered on first call so that all modules under
    'utils.complexity.metrics' are imported and their classes registered.

    Returns
    -------
    Iterable[str]
        The set of metric names currently registered.
    """
    discover_metrics()  # ensure populated
    return tuple(_METRIC_REGISTRY.keys())


def get_metric_class(name: str) -> Type[Metric]:
    """
    Backward-compatible lookup of a single Metric class by name.

    If multiple variants are registered under the same name, this returns
    the **first registered** class (typically the trace-based one, unless
    your import order differs). For variant-aware selection (e.g., trace
    vs. distribution), prefer `get_metric_classes(name)` and let your
    orchestrator decide.

    Parameters
    ----------
    name : str
        Public metric name as used during registration.

    Returns
    -------
    Type[Metric]
        The first registered implementation class for the given name.

    Raises
    ------
    KeyError
        If no metric is registered under the given name.
    """
    discover_metrics()
    try:
        return _METRIC_REGISTRY[name][0]
    except (KeyError, IndexError) as e:
        raise KeyError(f"Metric not registered: {name}") from e


# ---- Helper for variant-aware callers ----
def get_metric_classes(name: str) -> List[Type[Metric]]:
    """
    Return all registered implementation classes (variants) for a metric name.

    Use this when you need to choose between multiple variants (e.g.,
    trace-based vs. distribution-based) in a higher-level orchestrator.

    Parameters
    ----------
    name : str
        Public metric name as used during registration.

    Returns
    -------
    List[Type[Metric]]
        All classes registered under this name, in registration order.

    Raises
    ------
    KeyError
        If no metric is registered under the given name.
    """
    discover_metrics()
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Metric not registered: {name}")
    return list(_METRIC_REGISTRY[name])


def discover_metrics(package: str = "utils.complexity.metrics") -> None:
    """
    Import all modules under the given metrics package exactly once so
    that decorated Metric classes are registered.

    This walks the package tree (including subpackages like
    'trace_based' and 'distribution_based'), imports each module, and
    relies on the `@register_metric` decorator to populate the registry.

    Parameters
    ----------
    package : str, optional
        Fully-qualified package path to search (default:
        'utils.complexity.metrics').

    Notes
    -----
    - Discovery is idempotent and guarded by an internal flag.
    - Private/dunder modules are skipped.
    - Import errors are swallowed to avoid breaking discovery; consider
      logging them if needed for debugging.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        return  # not a package (nothing to discover)

    # Walk the whole metrics package tree and import submodules
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, package + "."):
        # Skip private modules
        if any(part.startswith("_") for part in modname.split(".")):
            continue
        importlib.import_module(modname)
        # try:
        #     importlib.import_module(modname)
        # except Exception:
        #     print('Import error for {modname}')
        #     # Optional: log or store the error; don’t crash discovery
        #     # e.g., logging.getLogger(__name__).exception("Failed to import %s", modname)
        #     pass
