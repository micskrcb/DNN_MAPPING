"""Communication-group axis reorderings, inspired by DOS's structured search.

This is not a DOS reproduction: no four-dimensional tensor coordinates or
TianjicX time-sharing model are assumed. No objective queries are used here.
"""
import numpy as np


def flow_groups(task_count, sources, destinations, volumes, limit=8):
    """Deterministic bounded BFS, strongest communication edges visited first."""
    adjacency = [dict() for _ in range(task_count)]
    for source, destination, volume in zip(sources, destinations, volumes):
        a, b = int(source), int(destination)
        if a == b or volume <= 0:
            continue
        adjacency[a][b] = adjacency[a].get(b, 0.0) + float(volume)
        adjacency[b][a] = adjacency[b].get(a, 0.0) + float(volume)
    neighbors = [sorted(row, key=lambda j: (-row[j], j)) for row in adjacency]
    groups = []
    for anchor in range(task_count):
        group, seen = [anchor], {anchor}
        cursor = 0
        while cursor < len(group) and len(group) < limit:
            for task in neighbors[group[cursor]]:
                if task not in seen:
                    seen.add(task);group.append(task)
                    if len(group) == limit:
                        break
            cursor += 1
        if len(group) > 1:
            groups.append(np.asarray(group, dtype=np.int64))
    return groups


def axis_ranks(chip_x, chip_y, rows, cols):
    """Two chip-major spatial orderings, each valid on rectangular meshes."""
    ranks = []
    for keys in ((cols, rows, chip_x, chip_y), (rows, cols, chip_y, chip_x)):
        order = np.lexsort(keys)
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        ranks.append(rank)
    return ranks


def flow_axis_neighbor(placement, groups, ranks, rng):
    """Reassign existing slots within a connected group; never alter occupancy."""
    group = groups[rng.randrange(len(groups))]
    rank = ranks[rng.randrange(len(ranks))]
    old = placement[group]
    ordered = old[np.argsort(rank[old], kind='stable')]
    if rng.randrange(2):
        ordered = ordered[::-1]
    if np.array_equal(ordered, old):
        ordered = np.roll(ordered, 1)
    candidate = placement.copy()
    candidate[group] = ordered
    changed = group[ordered != old]
    return candidate, changed.tolist(), None
