"""Explicit full-frame timing approximation; not the paper's streaming simulator."""
import math
import numpy as np


PAPER_LOGIC_CORE_TARGETS = {
    "alexnet": {"conv": 183, "linear": 932},
    "vgg16": {"conv": 1024, "linear": 1924},
    "resnet50": {"conv": 512, "linear": 37},
}


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


def paper_partition_grids(layers, target_count, weight_buffer_bytes=64 * 1024,
                          candidate_limit=512):
    """Choose per-layer ``(M, N)`` grids summing to a Figure-6 core count.

    A layer with ``M`` output groups and ``N`` input groups consumes
    ``M*N`` VMM cores plus ``M`` VVA cores. The paper publishes aggregate
    CONV/FC counts and says that work is balanced, but does not publish its
    per-layer grids. This deterministic reconstruction allocates counts in
    proportion to layer MACs, enforces the Table-1 64-KB weight buffer for
    every VMM tile, and finds the minimum-error exact integer allocation.

    ``layers`` contains dictionaries with ``cin``, ``cout``, ``height``,
    ``width`` and ``kernel_area``. The returned list follows input order.
    """
    if not layers or target_count < 2 * len(layers):
        raise ValueError("target_count cannot provide at least one VMM and VVA per layer")
    if weight_buffer_bytes <= 0:
        raise ValueError("weight_buffer_bytes must be positive")

    macs = [layer["cin"] * layer["cout"] * layer["height"] *
            layer["width"] * layer["kernel_area"] for layer in layers]
    total_macs = sum(macs)
    ideals = [target_count * value / total_macs for value in macs]

    choices = []
    for layer, ideal in zip(layers, ideals):
        cin, cout, kernel = layer["cin"], layer["cout"], layer["kernel_area"]
        by_count = {}
        for m_groups in range(1, min(cout, target_count) + 1):
            max_n = min(cin, target_count // m_groups - 1)
            for n_groups in range(1, max_n + 1):
                count = m_groups * (n_groups + 1)
                max_cin = math.ceil(cin / n_groups)
                max_cout = math.ceil(cout / m_groups)
                if max_cin * max_cout * kernel > weight_buffer_bytes:
                    continue
                relative_error = ((count - ideal) / max(ideal, 1.0)) ** 2
                # For equally close counts, prefer fewer VVA reduction cores.
                score = relative_error + 1e-9 * m_groups / count
                previous = by_count.get(count)
                if previous is None or score < previous[0]:
                    by_count[count] = (score, m_groups, n_groups)
        if not by_count:
            raise ValueError(f"No partition for layer {layer.get('name', '?')} fits the weight buffer")
        ranked = sorted((score, count, m_groups, n_groups)
                        for count, (score, m_groups, n_groups) in by_count.items())
        choices.append(ranked[:candidate_limit])

    states = {0: (0.0, [])}
    for layer_choices in choices:
        next_states = {}
        for current_count, (current_score, path) in states.items():
            for score, count, m_groups, n_groups in layer_choices:
                new_count = current_count + count
                if new_count > target_count:
                    continue
                new_score = current_score + score
                if (new_count not in next_states or
                        new_score < next_states[new_count][0]):
                    next_states[new_count] = (new_score,
                                             path + [(m_groups, n_groups)])
        states = next_states
    if target_count not in states:
        raise ValueError("Could not reconstruct an exact paper logic-core allocation")
    return states[target_count][1]
