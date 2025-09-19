from __future__ import annotations

import math
from typing import List

from lempel_ziv_complexity import lempel_ziv_complexity as _lz76_impl

from utils.complexity.measures.measure_store import MeasureStore
from utils.complexity.metrics.metric import Metric
from utils.complexity.metrics.registry import register_metric
from utils.windowing.window import Window


@register_metric("Lempel-Ziv Complexity")
class LempelZivComplexity(Metric):
    """
    LZ76 phrase-count complexity on a window's event log,
    using Pentland-style row-wise concatenation.
    
    Parameters
    ----------
    normalize : bool, default False
        If True, apply length/alphabet normalization: 
        C* = C * log_a(n) / n
    padding : bool, default True
        If True, pad traces to equal length with a sentinel
        before concatenation (Pentland matrix-to-vector style).
    """

    name = "Lempel-Ziv Complexity"

    def __init__(self, normalize: bool = False, padding: bool = True):
        self.normalize = normalize
        self.padding = padding

    def compute(self, window: "Window", measures: MeasureStore) -> None:
        if measures.has(self.name):
            return

        # --- 1) Extract sequences of activity labels
        sequences: List[List[str]] = []
        for trace in window.traces:
            seq = [ev["concept:name"] for ev in trace]
            sequences.append(seq)

        if not sequences:
            measures.set(self.name, 0.0 if self.normalize else 0)
            return

        # --- 2) Concatenate rows (with optional padding)
        if self.padding:
            PAD = object()  # unique sentinel
            max_len = max(len(s) for s in sequences)
            padded = [s + [PAD] * (max_len - len(s)) for s in sequences]
            flat = [tok for row in padded for tok in row]
        else:
            flat = [tok for row in sequences for tok in row]

        # --- 3) Map tokens to consecutive ints
        vocab, int_seq = {}, []
        for tok in flat:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            int_seq.append(vocab[tok])

        # --- 4) Compute raw LZ complexity
        lz_val = _lz76_impl(tuple(int_seq))

        # --- 5) Apply optional normalization
        if self.normalize:
            n = len(int_seq)
            a = max(1, len(vocab))
            if n == 0 or a <= 1:
                lz_val = 0.0
            else:
                lz_val = lz_val * (math.log(n, a)) / n

        measures.set(self.name, lz_val, hidden=False, meta={"basis": "traces"})