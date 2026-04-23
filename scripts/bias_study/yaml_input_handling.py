"""Load and parse bias_study YAML (experiment_settings.yaml, scenarios.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from utils import constants, helpers

_BIAS_STUDY_DIR = Path(__file__).resolve().parent


def _resolve_include_metrics(raw: Any) -> List[str]:
    """YAML null / missing -> all registered metrics; else shorthand or full names."""
    if raw is None:
        return list(constants.ALL_METRIC_NAMES)
    if not isinstance(raw, list):
        raise ValueError("experiment_settings include_metrics must be a list or null")
    return helpers.resolve_metric_names([str(x) for x in raw])


def parse_window_sizes(spec: Union[Dict[str, int], List[int]]) -> Union[range, List[int]]:
    if isinstance(spec, list):
        return [int(x) for x in spec]
    start = int(spec["start"])
    stop = int(spec["stop"])
    step = int(spec["step"])
    return range(start, stop + 1, step)


def load_experiment_settings(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    p = path or _BIAS_STUDY_DIR / "experiment_settings.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "full" not in data or "test" not in data:
        raise ValueError(f"{p} must contain 'full' and 'test' top-level keys")
    return data


def load_scenarios_yaml(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = path or _BIAS_STUDY_DIR / "scenarios.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a mapping of scenario name -> config")
    return data


@dataclass(frozen=True)
class ExperimentSettings:
    """Loaded from experiment_settings.yaml (profile full or test); passed through the pipeline."""

    samples_per_size: int
    random_state: int
    bootstrap_replica_count: int
    include_metrics: List[str]
    window_sizes: Union[range, List[int]]
    correlation_start: int
    correlation_stop: int
    plateau_windows_per_test: int
    plateau_step_between_tests: int
    plateau_alpha: float
    plateau_number_consecutive_non_trending_tests: int
    reliability_sizes: List[int]


def experiment_settings_from_profile(profile: Dict[str, Any]) -> ExperimentSettings:
    ca = profile["correlation_analysis"]
    pa = profile["plateau_analysis"]
    ra = profile["reliability_analysis"]
    ws = profile["window_sizes"]
    return ExperimentSettings(
        samples_per_size=int(profile["samples_per_size"]),
        random_state=int(profile["random_state"]),
        bootstrap_replica_count=int(profile["bootstrap_replica_count"]),
        include_metrics=_resolve_include_metrics(profile.get("include_metrics")),
        window_sizes=parse_window_sizes(ws),
        correlation_start=int(ca["start"]),
        correlation_stop=int(ca["stop"]),
        plateau_windows_per_test=int(pa["windows_per_test"]),
        plateau_step_between_tests=int(pa["step_between_tests"]),
        plateau_alpha=float(pa["alpha"]),
        plateau_number_consecutive_non_trending_tests=int(
            pa["number_consecutive_non_trending_tests"]
        ),
        reliability_sizes=[int(x) for x in ra["window_sizes"]],
    )


def resolve_scenario_dict(
    name: str,
    raw: Dict[str, Any],
    *,
    default_sample_confidence_interval_extractor: Any,
    default_metric_adapters: Any,
    naive_population_extractor: Any,
    chao1_population_extractor: Any,
    default_normalizers: Any,
) -> Dict[str, Any]:
    """Turn YAML scenario row into the dict expected by compute_results."""
    pop = raw["population_extractor"]
    if pop == "naive":
        population_extractor = naive_population_extractor
    elif pop == "chao1":
        population_extractor = chao1_population_extractor
    else:
        raise ValueError(
            f"Scenario {name}: unknown population_extractor {pop!r} (use naive|chao1)"
        )

    norm = raw.get("normalizers")
    if norm is None or norm == "null":
        normalizers = None
    elif norm == "default":
        normalizers = default_normalizers
    else:
        raise ValueError(
            f"Scenario {name}: unknown normalizers {norm!r} (use null|default)"
        )

    base = raw.get("base_scenario_name")
    if base in (None, "null"):
        base_scenario_name = None
    else:
        base_scenario_name = str(base)

    return {
        "logs": list(raw["logs"]),
        "clear_name": str(raw["clear_name"]),
        "population_extractor": population_extractor,
        "metric_adapters": default_metric_adapters,
        "bootstrap_sampler": None,
        "normalizers": normalizers,
        "sample_confidence_interval_extractor": default_sample_confidence_interval_extractor,
        "base_scenario_name": base_scenario_name,
    }


def build_scenarios_registry(
    yaml_data: Dict[str, Dict[str, Any]],
    **resolve_kw: Any,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, raw in yaml_data.items():
        out[name] = resolve_scenario_dict(name, raw, **resolve_kw)
    return out
