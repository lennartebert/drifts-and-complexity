# create directories
mkdir -p dynamic_metrics/{domain,predictors,metrics,bootstrap,normalize,pipeline,config,utils} tests

# root files
cat > dynamic_metrics/__init__.py <<'EOF'
from .types import MetricDict
from .pipeline.runner import WindowRunner, WindowRunnerConfig
from .config.registries import build_runner_config
EOF

cat > dynamic_metrics/types.py <<'EOF'
from __future__ import annotations
from typing import Dict, Any, Mapping, Sequence, Protocol

MetricDict = Dict[str, float]

class MetricAdapter(Protocol):
    name: str
    def compute(self, window: "Window", pop: "WindowPopulation") -> "MetricBundle": ...

class PopulationPredictor(Protocol):
    def predict(self, pop: "WindowPopulation") -> "WindowPopulation": ...

class BootstrapSampler(Protocol):
    def resample(self, pop: "WindowPopulation", rng: "random.Random") -> "Window": ...

class Normalizer(Protocol):
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, float]: ...
EOF

# domain
cat > dynamic_metrics/domain/events.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Any

@dataclass(frozen=True)
class Event:
    activity: str

@dataclass(frozen=True)
class Trace:
    events: Tuple[Event, ...]
    def activities(self) -> Tuple[str, ...]:
        return tuple(e.activity for e in self.events)

@dataclass(frozen=True)
class Window:
    traces: Tuple[Trace, ...]
    meta: dict[str, Any] = field(default_factory=dict)
EOF

cat > dynamic_metrics/domain/population.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from .events import Window, Trace, Event

@dataclass
class WindowPopulation:
    activity_counts: Counter[str]
    variant_counts: Counter[tuple[str, ...]]
    n_traces: int
    @property
    def n_events(self) -> int:
        return sum(len(v) * c for v, c in self.variant_counts.items())
    @property
    def n_distinct_activities(self) -> int: return len(self.activity_counts)
    @property
    def n_distinct_variants(self) -> int: return len(self.variant_counts)

def build_population(window: Window) -> WindowPopulation:
    ac, vc = Counter(), Counter()
    for tr in window.traces:
        v = tuple(e.activity for e in tr.events)
        vc[v] += 1
        ac.update(v)
    return WindowPopulation(ac, vc, n_traces=len(window.traces))
EOF

# predictors
cat > dynamic_metrics/predictors/base.py <<'EOF'
from __future__ import annotations
from ..domain.population import WindowPopulation

class NoOpPredictor:
    def predict(self, pop: WindowPopulation) -> WindowPopulation:
        return pop
EOF

cat > dynamic_metrics/predictors/chao_variant.py <<'EOF'
from __future__ import annotations
from collections import Counter
from ..domain.population import WindowPopulation

class ChaoVariantPredictor:
    def __init__(self, min_count:int=1):
        self.min_count = min_count
    def predict(self, pop: WindowPopulation) -> WindowPopulation:
        f1 = sum(1 for c in pop.variant_counts.values() if c == 1)
        f2 = sum(1 for c in pop.variant_counts.values() if c == 2)
        unseen = 0 if f2 == 0 else int(round((f1*f1)/(2*f2)))
        if unseen <= 0: return pop
        new_v = Counter(pop.variant_counts)
        new_v[("<UNSEEN>",)] += max(unseen, self.min_count)
        return WindowPopulation(
            activity_counts=Counter(pop.activity_counts),
            variant_counts=new_v,
            n_traces=pop.n_traces + max(unseen, self.min_count),
        )
EOF

# metrics
cat > dynamic_metrics/metrics/base.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
from ..domain.events import Window
from ..domain.population import WindowPopulation

@dataclass
class MetricBundle:
    metrics: Dict[str, float]
    extras: Dict[str, Any] = field(default_factory=dict)

class BaseMetricAdapter:
    name = "base"
    def compute(self, window: Window, pop: WindowPopulation) -> MetricBundle:
        raise NotImplementedError
EOF

cat > dynamic_metrics/metrics/basic_counts.py <<'EOF'
from __future__ import annotations
from .base import BaseMetricAdapter, MetricBundle
from ..domain.events import Window
from ..domain.population import WindowPopulation

class BasicCountsAdapter(BaseMetricAdapter):
    name = "basic"
    def compute(self, window: Window, pop: WindowPopulation) -> MetricBundle:
        return MetricBundle(metrics={
            "Number of Traces": float(pop.n_traces),
            "Number of Events": float(pop.n_events),
            "Number of Distinct Activities": float(pop.n_distinct_activities),
            "Number of Distinct Variants": float(pop.n_distinct_variants),
        })
EOF

cat > dynamic_metrics/metrics/vidgof_adapter.py <<'EOF'
from __future__ import annotations
from typing import Callable, Dict
from .base import BaseMetricAdapter, MetricBundle
from ..domain.events import Window
from ..domain.population import WindowPopulation

class VidgofAdapter(BaseMetricAdapter):
    name = "vidgof"
    def __init__(self, fn: Callable[[Window, WindowPopulation], Dict[str, float]]):
        self.fn = fn
    def compute(self, window: Window, pop: WindowPopulation) -> MetricBundle:
        return MetricBundle(metrics=self.fn(window, pop))
EOF

cat > dynamic_metrics/metrics/inext_adapter.py <<'EOF'
from __future__ import annotations
from typing import Callable, Dict
from .base import BaseMetricAdapter, MetricBundle
from ..domain.events import Window
from ..domain.population import WindowPopulation

class INextAdapter(BaseMetricAdapter):
    name = "inext"
    def __init__(self, fn: Callable[[Window, WindowPopulation], Dict[str, float]]):
        self.fn = fn
    def compute(self, window: Window, pop: WindowPopulation) -> MetricBundle:
        return MetricBundle(metrics=self.fn(window, pop))
EOF

# bootstrap
cat > dynamic_metrics/bootstrap/base.py <<'EOF'
from __future__ import annotations
import random, numpy as np
from typing import Dict
from ..domain.events import Window, Trace, Event
from ..domain.population import WindowPopulation, build_population
from ..pipeline.aggregator import MetricAggregator

class BaseBootstrapSampler:
    def resample(self, pop: WindowPopulation, rng: random.Random) -> Window:
        raise NotImplementedError

class Bootstrapper:
    def __init__(self, sampler: "BaseBootstrapSampler", n_boot:int=400, random_state:int|None=None):
        self.sampler = sampler; self.n_boot=n_boot; self.random_state=random_state
    def run(self, base_window: Window, pop: WindowPopulation, aggregator: MetricAggregator) -> Dict[str, np.ndarray]:
        rng = random.Random(self.random_state); samples={}
        for _ in range(self.n_boot):
            w = self.sampler.resample(pop, rng)
            pop_b = build_population(w); m = aggregator.compute_all(w, pop_b)
            for k,v in m.items(): samples.setdefault(k,[]).append(float(v))
        return {k: np.asarray(vs,float) for k,vs in samples.items()}
EOF

cat > dynamic_metrics/bootstrap/ordinary.py <<'EOF'
from __future__ import annotations
import random, numpy as np
from ..domain.events import Window, Trace, Event
from ..domain.population import WindowPopulation
from .base import BaseBootstrapSampler

class OrdinaryBootstrapSampler(BaseBootstrapSampler):
    def resample(self, pop: WindowPopulation, rng: random.Random) -> Window:
        variants, counts = zip(*pop.variant_counts.items())
        probs = np.array(counts,float); probs/=probs.sum()
        idx = rng.choices(range(len(variants)),weights=probs,k=pop.n_traces)
        traces=[Trace(events=tuple(Event(a) for a in variants[i])) for i in idx]
        return Window(traces=tuple(traces))
EOF

cat > dynamic_metrics/bootstrap/rare_weighted.py <<'EOF'
from __future__ import annotations
import random, numpy as np
from ..domain.events import Window, Trace, Event
from ..domain.population import WindowPopulation
from .base import BaseBootstrapSampler

class RareWeightedBootstrapSampler(BaseBootstrapSampler):
    def __init__(self, alpha: float = 1.0): self.alpha=alpha
    def resample(self, pop: WindowPopulation, rng: random.Random) -> Window:
        variants, counts = zip(*pop.variant_counts.items()); counts=np.array(counts,float)
        weights=(1.0/np.maximum(counts,1.0))**self.alpha; probs=weights/weights.sum()
        idx=rng.choices(range(len(variants)),weights=probs,k=pop.n_traces)
        traces=[Trace(events=tuple(Event(a) for a in variants[i])) for i in idx]
        return Window(traces=tuple(traces))
EOF

# normalize
cat > dynamic_metrics/normalize/base.py <<'EOF'
from __future__ import annotations
from typing import Mapping, Dict, Sequence
from ..types import Normalizer

class NormalizationPipeline:
    def __init__(self, normalizers: Sequence[Normalizer]): self.normalizers=list(normalizers)
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        out=dict(metrics)
        for n in self.normalizers: out=n.apply(out)
        return out
EOF

cat > dynamic_metrics/normalize/lz_replication_invariant.py <<'EOF'
from __future__ import annotations
import math
from typing import Mapping, Dict
from ..types import Normalizer

class LZReplicationInvariantNormalizer(Normalizer):
    def __init__(self,
                 lz_key="Lempel-Ziv Complexity",
                 n_events_key="Number of Events",
                 v_key="Number of Distinct Activities",
                 out_key="Lempel-Ziv Complexity (RI)"):
        self.lz_key, self.n_events_key, self.v_key, self.out_key = lz_key,n_events_key,v_key,out_key
    def apply(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        out=dict(metrics); lz,N,V=metrics.get(self.lz_key),metrics.get(self.n_events_key),metrics.get(self.v_key)
        if lz and N and V and N>1 and V>1:
            denom=N/(math.log(N)/math.log(V)); 
            if denom>0: out[self.out_key]=float(lz)/denom
        return out
EOF

# pipeline
cat > dynamic_metrics/pipeline/aggregator.py <<'EOF'
from __future__ import annotations
from typing import Sequence, Dict
from ..types import MetricAdapter
from ..domain.events import Window
from ..domain.population import WindowPopulation

class MetricAggregator:
    def __init__(self, adapters: Sequence[MetricAdapter]): self.adapters=list(adapters)
    def compute_all(self, window: Window, pop: WindowPopulation) -> Dict[str, float]:
        out={}; 
        for ad in self.adapters: out.update(ad.compute(window,pop).metrics)
        return out
EOF

cat > dynamic_metrics/pipeline/result.py <<'EOF'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MetricCI:
    raw: float; norm: float|None; mean: float; std: float; ci_low: float; ci_high: float

@dataclass
class WindowResult:
    window_meta: Dict[str, Any]
    metrics_raw: Dict[str, float]
    metrics_norm: Dict[str, float]
    ci: Dict[str, MetricCI]
EOF

cat > dynamic_metrics/pipeline/runner.py <<'EOF'
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, Any
from ..domain.events import Window
from ..domain.population import build_population, WindowPopulation
from ..types import PopulationPredictor
from ..bootstrap.base import Bootstrapper
from ..normalize.base import NormalizationPipeline
from .aggregator import MetricAggregator
from .result import WindowResult, MetricCI
from ..metrics.basic_counts import BasicCountsAdapter

@dataclass
class WindowRunnerConfig:
    predictors: Sequence[PopulationPredictor] = ()
    adapters: Sequence = (BasicCountsAdapter(),)
    bootstrapper: Optional[Bootstrapper] = None
    normalization: NormalizationPipeline = field(default_factory=lambda: NormalizationPipeline([]))
    ci_alpha: float = 0.05

class WindowRunner:
    def __init__(self, cfg: WindowRunnerConfig):
        self.cfg=cfg; self.aggregator=MetricAggregator(cfg.adapters)
    def run(self, window: Window) -> WindowResult:
        pop=build_population(window)
        for pred in self.cfg.predictors: pop=pred.predict(pop)
        base=self.aggregator.compute_all(window,pop); base_norm=self.cfg.normalization.apply(base)
        ci={}
        if self.cfg.bootstrapper:
            boots=self.cfg.bootstrapper.run(window,pop,self.aggregator); keys=list(boots.keys())
            n=len(next(iter(boots.values()))) if keys else 0; norm_draws={k:np.empty(n) for k in keys}
            for i in range(n):
                rep={k:boots[k][i].item() for k in keys}; rep_norm=self.cfg.normalization.apply(rep)
                for k in keys: norm_draws[k][i]=rep_norm.get(k,rep[k])
            for k in keys:
                arr=norm_draws[k]; lo,hi=np.quantile(arr,[self.cfg.ci_alpha/2,1-self.cfg.ci_alpha/2])
                ci[k]=MetricCI(raw=float(base.get(k,np.nan)),norm=float(base_norm.get(k,base.get(k,np.nan))),
                               mean=float(arr.mean()),std=float(arr.std(ddof=1)) if arr.size>1 else 0.0,
                               ci_low=float(lo),ci_high=float(hi))
        else:
            for k,v in base.items():
                ci[k]=MetricCI(raw=float(v),norm=float(base_norm.get(k,v)),
                               mean=float("nan"),std=float("nan"),ci_low=float("nan"),ci_high=float("nan"))
        return WindowResult(window_meta=window.meta,metrics_raw=base,metrics_norm=base_norm,ci=ci)
EOF

# config
cat > dynamic_metrics/config/registries.py <<'EOF'
from __future__ import annotations
from typing import Dict, Callable
from ..predictors.base import NoOpPredictor
from ..predictors.chao_variant import ChaoVariantPredictor
from ..normalize.base import NormalizationPipeline
from ..normalize.lz_replication_invariant import LZReplicationInvariantNormalizer
from ..bootstrap.base import Bootstrapper
from ..bootstrap.ordinary import OrdinaryBootstrapSampler
from ..bootstrap.rare_weighted import RareWeightedBootstrapSampler
from ..pipeline.runner import WindowRunnerConfig
from ..metrics.basic_counts import BasicCountsAdapter

PREDICTOR_REGISTRY: Dict[str, Callable[..., object]]={"none":NoOpPredictor,"chao-variant":ChaoVariantPredictor}
NORMALIZER_REGISTRY: Dict[str, Callable[..., object]]={"lz-ri":LZReplicationInvariantNormalizer}
BOOTSTRAP_SAMPLER_REGISTRY: Dict[str, Callable[..., object]]={"ordinary":OrdinaryBootstrapSampler,"rare-weighted":RareWeightedBootstrapSampler}

def build_runner_config(predictors:list[dict]|None,normalizers:list[dict]|None,bootstrap:dict|None,adapters:list[object]|None=None,ci_alpha:float=0.05)->WindowRunnerConfig:
    preds=[PREDICTOR_REGISTRY[spec["name"]](**spec.get("kwargs",{})) for spec in (predictors or [])]
    norms=[NORMALIZER_REGISTRY[spec["name"]](**spec.get("kwargs",{})) for spec in (normalizers or [])]
    pipeline=NormalizationPipeline(norms)
    if bootstrap is None: boot=None
    else:
        sctor=BOOTSTRAP_SAMPLER_REGISTRY[bootstrap["sampler"]["name"]]; sampler=sctor(**bootstrap["sampler"].get("kwargs",{}))
        boot=Bootstrapper(sampler,n_boot=bootstrap.get("n_boot",400),random_state=bootstrap.get("random_state"))
    return WindowRunnerConfig(predictors=tuple(preds) if preds else (NoOpPredictor(),),
                              adapters=tuple(adapters or [BasicCountsAdapter()]),
                              bootstrapper=boot,normalization=pipeline,ci_alpha=ci_alpha)
EOF

cat > dynamic_metrics/config/loader.py <<'EOF'
from __future__ import annotations
import yaml
from .registries import build_runner_config

def load_profile(path:str,profile:str):
    with open(path,"r") as f: cfg=yaml.safe_load(f)["profiles"][profile]
    return build_runner_config(predictors=cfg.get("predictors"),normalizers=cfg.get("normalizers"),
                               bootstrap=cfg.get("bootstrap"),adapters=None,ci_alpha=cfg.get("ci_alpha",0.05))
EOF

# utils
cat > dynamic_metrics/utils/stats.py <<'EOF'
import numpy as np
def percentile_ci(arr: np.ndarray, alpha=0.05): return np.quantile(arr,[alpha/2,1-alpha/2])
EOF

cat > dynamic_metrics/utils/hashing.py <<'EOF'
import hashlib,json
def stable_hash(obj)->str:
    return hashlib.sha1(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
EOF
