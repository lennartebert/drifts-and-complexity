"""Load experiment settings and scenario definitions from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

_BIAS_STUDY_DIR = Path(__file__).resolve().parent


def _parse_correlation_sizes(
    spec: Union[Dict[str, int], List[int]]
) -> Union[range, List[int]]:
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


def apply_experiment_profile(
    profile: Dict[str, Any],
    *,
    globals_dict: Dict[str, Any],
) -> None:
    """Apply one profile (full or test) into the given globals mapping (run_bias_study module)."""
    g = globals_dict
    g["SAMPLES_PER_SIZE"] = int(profile["samples_per_size"])
    g["RANDOM_STATE"] = int(profile["random_state"])
    g["BOOTSTRAP_REPLICA_COUNT"] = int(profile["bootstrap_replica_count"])

    ca = profile["correlation_analysis"]
    pa = profile["plateau_analysis"]
    ra = profile["reliability_analysis"]

    g["CORRELATION_SIZES"] = _parse_correlation_sizes(ca["window_sizes"])
    g["PLATEAU_MIN"] = int(pa["min_window"])
    g["PLATEAU_MAX_CAP"] = int(pa["max_window_cap"])
    g["PLATEAU_STEP"] = int(pa["step"])
    g["PLATEAU_THRESHOLD"] = float(pa["relative_threshold"])
    g["RELIABILITY_SIZES"] = [int(x) for x in ra["window_sizes"]]
    # Same sizes as reliability_analysis: master.csv RelCI {n} columns (see combine_analysis_with_means).
    g["REF_SIZES"] = list(g["RELIABILITY_SIZES"])
    g["BOOTSTRAP_SIZE"] = g["BOOTSTRAP_REPLICA_COUNT"]
    from utils.bootstrapping.bootstrap_samplers.bootstrap_sampler import BootstrapSampler

    g["default_bootstrap_sampler"] = BootstrapSampler(
        B=g["BOOTSTRAP_REPLICA_COUNT"], seed=g["RANDOM_STATE"]
    )


def load_scenarios_yaml(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = path or _BIAS_STUDY_DIR / "scenarios.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a mapping of scenario name -> config")
    return data


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
