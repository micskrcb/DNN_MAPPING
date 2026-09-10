"""Explicit full-frame timing approximation; not the paper's streaming simulator."""
import math
import numpy as np


def channel_ranges(channels, groups):
    """Balanced disjoint ranges; remainder channels are never counted twice."""
    if not 1 <= groups <= channels:
        raise ValueError("groups must be between 1 and channels")
    return [(i * channels // groups, (i + 1) * channels // groups)
            for i in range(groups)]


def tile_work(cin, cout, height, width, kernel_area, input_groups, output_groups):
    inputs = channel_ranges(cin, input_groups)
    outputs = channel_ranges(cout, output_groups)
    work = []
    kinds = []
    for lo, hi in outputs:
        for il, ih in inputs:
            work.append((ih - il) * (hi - lo) * height * width * kernel_area)
            kinds.append("vmm")
        work.append(max(input_groups - 1, 0) * (hi - lo) * height * width)
        kinds.append("vva")
    return work, kinds


def compute_seconds(operations, kinds, utilization=1.0, vva_ops_per_cycle=1.0,
                    frequency_hz=400e6, macs_per_cycle=128):
    """VMM throughput from Table 1; VVA throughput is a user-declared assumption.

    Ideal arithmetic cycles rounded up once per task; excludes memory stalls,
    activation/pooling/bias overhead and intra-frame streaming startup.
    """
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0,1]")
    if any(not math.isfinite(x) or x <= 0 for x in
           (vva_ops_per_cycle, frequency_hz, macs_per_cycle)):
        raise ValueError("hardware rates must be finite and positive")
    if len(operations) != len(kinds):
        raise ValueError("one operation kind required per task")
    ops = np.asarray(operations, dtype=np.float64)
    if np.any(ops < 0) or not np.all(np.isfinite(ops)):
        raise ValueError("operations must be finite and nonnegative")
    if any(k not in ("vmm", "vva") for k in kinds):
        raise ValueError("unknown operation kind")
    rates = np.array([macs_per_cycle if k == "vmm" else vva_ops_per_cycle for k in kinds])
    return np.ceil(ops / (rates * utilization)) / frequency_hz


def edge_bytes(graph, kinds):
    """32-bit partial sums from VMM; 8-bit activations from VVA (Table 1)."""
    return np.asarray(graph, dtype=np.float64) * np.array(
        [4 if k == "vmm" else 1 for k in kinds], dtype=np.float64)[:, None]
