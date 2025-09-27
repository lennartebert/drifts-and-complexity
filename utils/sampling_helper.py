"""Sampling utilities for trace and window generation."""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Tuple
from pm4py.objects.log.obj import Trace
import numpy as np

from utils.windowing.window import Window

ReplacementPolicy = Literal[
    "within_and_across",  # (1) full replacement: duplicates allowed within a sample and across samples
    "within_only",        # (2) no duplicates within a sample; samples are independent (replacement across samples)
    "none"                # (3) no replacement at all across all samples; each trace used at most once globally
]

def sample_random_traces(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    policy: ReplacementPolicy = "within_and_across",
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Trace]]]:
    """
    Sample random trace sets from an event log under different replacement policies.

    Parameters
    ----------
    event_log
        Iterable of traces (will be materialized to a list).
    sizes
        Iterable of positive integers indicating sample sizes to draw.
    samples_per_size
        Number of samples to draw per size.
    policy
        Replacement policy:
          - "within_and_across": draw with replacement for each sample (duplicates allowed within & across samples).
          - "within_only":        draw without replacement within a sample, but with replacement across samples
                                  (i.e., each sample is an independent no-replacement draw from the full log).
          - "none":               draw without replacement globally across all samples and sizes.
    random_state
        Seed for reproducible sampling.

    Returns
    -------
    List[Tuple[int, str, List[Trace]]]
        List of (window_size, sample_id, trace_list).
    """
    # Materialize and validate
    event_log = list(event_log)
    n_traces = len(event_log)

    if n_traces == 0 or samples_per_size <= 0:
        return []

    sizes = [int(s) for s in sizes if int(s) > 0]
    if not sizes:
        return []

    rng = np.random.default_rng(seed=random_state)
    results: List[Tuple[int, str, List[Trace]]] = []

    if policy == "within_and_across":
        # (1) Full replacement: allow repeats within a sample and across samples.
        for sample_id in range(samples_per_size):
            for s in sizes:
                idxs = rng.integers(low=0, high=n_traces, size=s).tolist()
                chosen_traces = [event_log[i] for i in idxs]
                results.append((s, str(sample_id), chosen_traces))
        return results

    if policy == "within_only":
        # (2) No replacement within a sample; across samples independent.
        #     Each sample requires s <= n_traces.
        max_s = max(sizes)
        if max_s > n_traces:
            raise ValueError(
                f"Requested size {max_s} exceeds number of traces {n_traces} for policy 'within_only'."
            )
        indices = np.arange(n_traces)
        for sample_id in range(samples_per_size):
            for s in sizes:
                # independent sample each time, without replacement
                idxs = rng.choice(indices, size=s, replace=False).tolist()
                chosen_traces = [event_log[i] for i in idxs]
                results.append((s, str(sample_id), chosen_traces))
        return results

    if policy == "none":
        # (3) Global no replacement across all samples and sizes.
        total_needed = int(samples_per_size) * int(sum(sizes))
        if total_needed > n_traces:
            raise ValueError(
                f"Insufficient traces ({n_traces}) for global no-replacement sampling "
                f"of {samples_per_size} * sum(sizes) = {total_needed}."
            )

        # Shuffle a pool of all indices once, then carve it sequentially per (sample_id, size)
        pool = rng.permutation(n_traces).tolist()
        cursor = 0
        for sample_id in range(samples_per_size):
            for s in sizes:
                take = pool[cursor: cursor + s]
                cursor += s
                chosen_traces = [event_log[i] for i in take]
                results.append((s, str(sample_id), chosen_traces))
        return results

    raise ValueError(f"Unknown policy: {policy!r}")


# --- Backwards-compatible wrappers for sampling --------------------------

def sample_random_traces_with_replacement(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Trace]]]:
    """(1) Duplicates allowed within and across samples."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_and_across",
        random_state=random_state,
    )

def sample_random_trace_sets_no_replacement_within_only(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Trace]]]:
    """(2) No duplicates within a sample; samples are independent across runs."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_only",
        random_state=random_state,
    )

def sample_random_trace_sets_no_replacement_global(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, List[Trace]]]:
    """(3) No replacement globally across all samples and sizes."""
    return sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="none",
        random_state=random_state,
    )

# --- Samplers that return windows --------------------------

def sample_random_windows_with_replacement(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, Window]]:
    """(1) Duplicates allowed within and across samples."""
    trace_samples = sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_and_across",
        random_state=random_state,
    )
    window_samples: List[Tuple[int, str, Window]] = [
        (window_size, sample_id, Window(id=sample_id, size=len(trace_list), traces=trace_list))
        for window_size, sample_id, trace_list in trace_samples
    ]
    return window_samples

def sample_random_windows_no_replacement_within_only(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, Window]]:
    """(2) No duplicates within a sample; samples are independent across runs."""
    trace_samples = sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="within_only",
        random_state=random_state,
    )
    window_samples: List[Tuple[int, str, Window]] = [
        (window_size, sample_id, Window(id=sample_id, size=len(trace_list), traces=trace_list))
        for window_size, sample_id, trace_list in trace_samples
    ]
    return window_samples

def sample_random_windows_no_replacement_global(
    event_log: Iterable[Trace],
    sizes: Iterable[int] = range(10, 501, 50),
    samples_per_size: int = 10,
    random_state: Optional[int] = None
) -> List[Tuple[int, str, Window]]:
    """(3) No replacement globally across all samples and sizes."""
    trace_samples = sample_random_traces(
        event_log=event_log,
        sizes=sizes,
        samples_per_size=samples_per_size,
        policy="none",
        random_state=random_state,
    )
    window_samples: List[Tuple[int, str, Window]] = [
        (window_size, sample_id, Window(id=sample_id, size=len(trace_list), traces=trace_list))
        for window_size, sample_id, trace_list in trace_samples
    ]
    return window_samples
