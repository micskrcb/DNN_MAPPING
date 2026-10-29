"""Legal operator-labelled moves for the existing online candidate scorer.

Reuses the project's structured group move; no additional objective queries.
"""
import numpy as np
from structured_moves import flow_axis_neighbor

OPERATOR_NAMES = ('original', 'relocation', 'swap', 'group')


def operator_neighbor(placement, size, free_pool, rng, index, count,
                      groups, ranks, original_neighbor):
    kind = 0 if index < count // 2 else 1 + (index - count // 2) % 3
    n = len(placement)
    if kind == 1 and free_pool:
        task = rng.randrange(n)
        candidate = placement.copy()
        used, freed = free_pool.choice(rng), int(candidate[task])
        candidate[task] = used
        return candidate, np.array([task]), (used, freed), kind
    if kind == 2 and n >= 2:
        a, b = rng.sample(range(n), 2)
        candidate = placement.copy()
        candidate[a], candidate[b] = candidate[b], candidate[a]
        return candidate, np.array([a, b]), None, kind
    if kind == 3 and groups and ranks:
        candidate, changed, relocation = flow_axis_neighbor(placement, groups, ranks, rng)
        return candidate, changed, relocation, kind
    # Missing free cores or groups: label the actual fallback, not requested type.
    candidate, changed, relocation = original_neighbor(placement, size, free_pool, rng=rng)
    return candidate, changed, relocation, 0


def append_operator_features(pool_features, kinds):
    return np.concatenate((pool_features, np.eye(4, dtype=np.float32)[kinds]), axis=1)
