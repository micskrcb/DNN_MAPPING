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


def nearby_allowed_cores(allowed, rows, cols, torus=False):
    """Cache at most eight same-chip neighbor slots; respect placement masks."""
    allowed = set(map(int, allowed))
    result = {}
    for core in sorted(allowed):
        chip, local = divmod(core, rows * cols)
        row, col = divmod(local, cols)
        near = set()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            r, c = row + dr, col + dc
            if torus:
                r, c = r % rows, c % cols
            if 0 <= r < rows and 0 <= c < cols:
                other = chip * rows * cols + r * cols + c
                if other in allowed and other != core:
                    near.add(other)
        result[core] = np.asarray(sorted(near), dtype=np.int64)
    return result


def attraction_neighbor(placement, pairs, nearby, inverse, free_pool, distance, rng):
    """Try up to four communicating pairs; prefer a closer free slot, else swap.

    Only the selected pair's distance proxy is improved. The full exact cost
    can worsen, so learning and Metropolis acceptance remain necessary.
    Free-pool state changes only when the caller accepts the returned move.
    """
    for _ in range(4):
        pair = pairs[rng.randrange(len(pairs))]
        source, target = map(int, pair)
        if rng.randrange(2):
            source, target = target, source
        old, anchor = int(placement[source]), int(placement[target])
        slots = nearby[anchor]
        closer = slots[distance(slots, anchor) < float(distance(old, anchor))]
        if not len(closer):
            continue
        free = [int(c) for c in closer if int(c) in free_pool.positions]
        destination = free[rng.randrange(len(free))] if free else int(closer[rng.randrange(len(closer))])
        candidate = placement.copy()
        candidate[source] = destination
        if destination in free_pool.positions:
            return candidate, [source], (destination, old), 1
        displaced = int(inverse[destination])
        if displaced < 0:
            raise ValueError('occupancy and free-pool state disagree')
        candidate[displaced] = old
        return candidate, [source, displaced], None, 2
    return None
