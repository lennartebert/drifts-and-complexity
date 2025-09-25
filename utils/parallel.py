# A tiny, centralized helper to run work in parallel without sprinkling
# ProcessPool/ThreadPool boilerplate across the codebase.

from __future__ import annotations
import os
import math
import contextlib
from typing import Any, Callable, Iterable, List, Literal, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

Backend = Literal["off", "auto", "processes", "threads"]


def _default_n_jobs(n_jobs: int | None) -> int:
    """
    Resolve the worker count:
      1) explicit n_jobs if provided
      2) SLURM_CPUS_PER_TASK if present
      3) os.cpu_count()
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    return max(1, int(n_jobs))


@contextlib.contextmanager
def _blas_safety(backend: Backend, force_single_thread_blas: bool = True):
    """
    Prevent massive oversubscription (processes x BLAS threads).
    This does NOT affect Python threading; it only caps BLAS-backed libs.
    """
    if force_single_thread_blas and backend in ("auto", "processes"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
    yield


def run_parallel(
    items: Sequence[Any],
    worker: Callable[[Any], Any],
    *,
    backend: Backend = "auto",
    n_jobs: int | None = None,
    chunksize: int | None = None,
    unordered: bool = True,
) -> List[Any]:
    """
    Execute `worker(item)` over `items` using a chosen backend.

    Args
    ----
    items: Sequence[Any]
        Input tasks. Prefer lightweight, picklable objects if using processes.
    worker: Callable[[Any], Any]
        Top-level function (picklable) that processes one item.
    backend: "off" | "auto" | "processes" | "threads"
        - "off": run sequentially
        - "auto": process pool if n_jobs>1 else sequential
        - "processes": multiprocessing (good for CPU-bound Python)
        - "threads": threads (only for I/O or C-extensions that release GIL)
    n_jobs: Optional[int]
        Number of workers. Defaults to SLURM_CPUS_PER_TASK or cpu_count().
    chunksize: Optional[int]
        Batch size for map semantics. Auto-tuned if None for processes.
    unordered: bool
        If True, return results as they complete (usually faster).

    Returns
    -------
    List[Any]: list of worker results (order not guaranteed if unordered=True).
    """
    n_jobs = _default_n_jobs(n_jobs)

    # Fast path: sequential
    if backend == "off" or n_jobs == 1:
        return [worker(x) for x in items]

    # Resolve "auto"
    if backend == "auto":
        backend = "processes"

    # Smart default chunksize
    if chunksize is None and backend == "processes":
        # A few batches per worker keeps overhead low but balances load
        chunksize = max(1, math.ceil(len(items) / (n_jobs * 8)))
    elif chunksize is None:
        chunksize = 1

    with _blas_safety(backend):
        Executor = ProcessPoolExecutor if backend == "processes" else ThreadPoolExecutor
        if unordered:
            results: List[Any] = []
            with Executor(max_workers=n_jobs) as ex:
                futs = [ex.submit(worker, it) for it in items]
                for f in as_completed(futs):
                    results.append(f.result())
            return results
        else:
            with Executor(max_workers=n_jobs) as ex:
                # map preserves order
                return list(ex.map(worker, items, chunksize=chunksize))
