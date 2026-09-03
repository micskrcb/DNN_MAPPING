"""
multi_chip_environment.py
--------------------------
Wraps MultiChipTopology into an RL-compatible environment.

CHANGE (objective function fix, vs. original):
The original `evaluate()` / `_total_cost()` summed Volume[i,j] * Latency[i,j]
over every task pair -- a total communication cost. The 2020 ACM paper
(Eq. 4, Sec 3.1.2) instead optimizes:

    P* = argmin_P { L(P) },   where L(P) = max_k T(k|P)

i.e. the *bottleneck* latency across pipeline stages, since a streaming
pipeline's throughput is capped by its slowest stage, not by the sum of
all stages. `evaluate()` now returns this bottleneck latency. The old
sum-of-pairs metric is preserved as `total_communication_cost()` for
diagnostics / comparison, and is still what `chip_breakdown()` reports.
"""

from typing import List, Optional

import numpy as np

from multi_chip_topology import MultiChipTopology


class MultiChipEnvironment:
    """Wraps multi-chip topology into an RL environment compatible with DDPG."""

    def __init__(
        self,
        num_chips_x: int = 2,
        num_chips_y: int = 2,
        rows_per_chip: int = 4,
        cols_per_chip: int = 4,
        on_chip_latency: float = 1.0,
        off_chip_latency: float = 5.0,
        task_graph: Optional[np.ndarray] = None,
        num_tasks: Optional[int] = None,
        compute_latency: Optional[np.ndarray] = None,
        make_dag: bool = True,
    ):
        """
        Args:
            task_graph: directed communication-volume matrix. task_graph[i, j]
                is the volume sent FROM task i TO task j. Must be a DAG for
                `pipeline_latency()` / `evaluate()` to be meaningful -- a
                symmetric (undirected) matrix has no well-defined stages.
            compute_latency: per-task compute-time proxy (length num_tasks).
                Defaults to zeros (communication-only model) if not supplied;
                pass real per-layer timings/FLOPs here when available for a
                more faithful reproduction of the paper's latency numbers.
            make_dag: when `task_graph` is None, generate a random DAG
                (upper-triangular volumes, i -> j for i < j) instead of a
                symmetric matrix, so the default synthetic workload still
                supports the pipeline objective. Set False to reproduce the
                old symmetric random-graph behavior (not pipeline-valid).
        """
        self.topo = MultiChipTopology(
            num_chips_x, num_chips_y,
            rows_per_chip, cols_per_chip,
            on_chip_latency, off_chip_latency,
        )
        self.total_cores = self.topo.total_cores

        if num_tasks is None:
            num_tasks = self.total_cores
        self.num_tasks = num_tasks

        if task_graph is None:
            rng = np.random.default_rng(42)
            W = rng.integers(0, 10, size=(num_tasks, num_tasks)).astype(np.float32)
            if make_dag:
                # Keep only i -> j edges for i < j: a random DAG respecting
                # task index order, so pipeline_stages() is well-defined.
                W = np.triu(W, k=1)
            else:
                W = (W + W.T) / 2
                np.fill_diagonal(W, 0)
            task_graph = W
        self.task_graph = task_graph
        self.latency = self.topo.latency_matrix()

        if compute_latency is None:
            compute_latency = np.zeros(self.num_tasks, dtype=np.float32)
        assert len(compute_latency) == self.num_tasks, \
            "compute_latency must have length num_tasks"
        self.compute_latency = np.asarray(compute_latency, dtype=np.float32)

        self.placement = np.full(self.num_tasks, -1, dtype=np.int32)
        self.core_occupied = np.zeros(self.total_cores, dtype=bool)

        self.state_dim = self.total_cores + self.num_tasks
        self.action_dim = 2

        # Cached once per placement change; pipeline_stages() only depends
        # on task_graph structure (not on placement), so compute it lazily
        # and reuse across evaluate() calls within a run.
        self._stages_cache: Optional[List[List[int]]] = None

    def reset(self) -> np.ndarray:
        self.placement[:] = -1
        self.core_occupied[:] = False
        return self._get_state()

    def step(self, action: np.ndarray):
        task_id = int(np.clip(
            np.round((action[0] + 1) / 2 * (self.num_tasks - 1)),
            0, self.num_tasks - 1
        ))
        core_id = int(np.clip(
            np.round((action[1] + 1) / 2 * (self.total_cores - 1)),
            0, self.total_cores - 1
        ))

        if self.placement[task_id] != -1 or self.core_occupied[core_id]:
            reward = -100.0
            done = False
            return self._get_state(), reward, done

        self.placement[task_id] = core_id
        self.core_occupied[core_id] = True

        done = np.all(self.placement != -1)
        reward = -self.evaluate() if done else 0.0
        return self._get_state(), reward, done

    def place(self, placement: np.ndarray):
        assert len(placement) == self.num_tasks
        self.placement = placement.copy()
        self.core_occupied[:] = False
        for c in placement:
            if 0 <= c < self.total_cores:
                self.core_occupied[c] = True

    # ------------------------------------------------------------------
    # Pipeline-stage objective (paper Eq. 4)
    # ------------------------------------------------------------------

    def pipeline_stages(self) -> List[List[int]]:
        """Group tasks into pipeline stages via Kahn's topological levels.

        Requires `task_graph` to be a DAG: task_graph[i, j] > 0 means task i
        sends data to task j. If a cycle is found (e.g. a symmetric/
        undirected graph was passed in), the unresolved tasks are dumped
        into one trailing stage rather than silently mis-measuring -- check
        `make_dag=True` if you hit this.
        """
        if self._stages_cache is not None:
            return self._stages_cache

        n = self.num_tasks
        succ: List[List[int]] = [[] for _ in range(n)]
        indeg = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and self.task_graph[i, j] > 0:
                    succ[i].append(j)
                    indeg[j] += 1

        stages: List[List[int]] = []
        remaining = indeg[:]
        frontier = [i for i in range(n) if remaining[i] == 0]
        seen = set()
        while frontier:
            stages.append(frontier)
            seen.update(frontier)
            nxt = []
            for u in frontier:
                for v in succ[u]:
                    remaining[v] -= 1
                    if remaining[v] == 0:
                        nxt.append(v)
            frontier = nxt

        leftover = [i for i in range(n) if i not in seen]
        if leftover:
            stages.append(leftover)

        self._stages_cache = stages
        return stages

    def pipeline_latency(self) -> float:
        """L(P) = max_k T(k|P) -- the paper's Eq. (4) objective.

        T(k|P) is the max, over tasks in stage k, of that task's own
        compute latency plus the communication latency to deliver its
        outputs to its successors' placed cores. Unplaced tasks (-1)
        contribute 0, matching the paper's convention that reward is
        only meaningful for the current (partial or complete) placement.
        """
        stages = self.pipeline_stages()
        stage_latencies = []
        for stage in stages:
            task_latencies = []
            for i in stage:
                ci = self.placement[i]
                if ci < 0:
                    task_latencies.append(0.0)
                    continue
                comm = 0.0
                for j in range(self.num_tasks):
                    vol = self.task_graph[i, j]
                    if vol <= 0:
                        continue
                    cj = self.placement[j]
                    if cj < 0:
                        continue
                    comm += vol * self.topo.comm_cost(ci, cj)
                task_latencies.append(float(self.compute_latency[i]) + comm)
            stage_latencies.append(max(task_latencies) if task_latencies else 0.0)
        return max(stage_latencies) if stage_latencies else 0.0

    def evaluate(self) -> float:
        """Primary objective: L(P), the bottleneck pipeline-stage latency
        (paper Eq. 4). This is what RS/SA/DDPG all optimize against."""
        return self.pipeline_latency()

    # ------------------------------------------------------------------
    # Diagnostics (old objective, kept for comparison -- NOT the reward)
    # ------------------------------------------------------------------

    def total_communication_cost(self) -> float:
        """Sum_i,j Volume[i,j] * Latency[core_i, core_j] over all task
        pairs. This was the pre-fix `evaluate()` / reward signal. It does
        not match the paper's objective (it ignores pipelining/bottleneck
        structure entirely) -- kept only as a secondary diagnostic."""
        return self._total_cost()

    def _total_cost(self) -> float:
        cost = 0.0
        for i in range(self.num_tasks):
            ci = self.placement[i]
            if ci < 0:
                continue
            for j in range(i + 1, self.num_tasks):
                cj = self.placement[j]
                if cj < 0:
                    continue
                volume = self.task_graph[i, j]
                if volume > 0:
                    cost += volume * self.latency[ci, cj]
        return cost

    def _get_state(self) -> np.ndarray:
        occupancy = self.core_occupied.astype(np.float32)
        task_norm = (self.placement / max(self.total_cores, 1)).astype(np.float32)
        return np.concatenate([occupancy, task_norm])

    def chip_breakdown(self) -> dict:
        """Diagnostic on-chip vs. off-chip split of total_communication_cost
        (NOT of the pipeline-latency objective)."""
        on_cost = 0.0
        off_cost = 0.0
        for i in range(self.num_tasks):
            ci = self.placement[i]
            if ci < 0:
                continue
            chip_i, _ = self.topo.chip_and_local(ci)
            for j in range(i + 1, self.num_tasks):
                cj = self.placement[j]
                if cj < 0:
                    continue
                chip_j, _ = self.topo.chip_and_local(cj)
                vol = self.task_graph[i, j]
                if vol == 0:
                    continue
                if chip_i == chip_j:
                    on_cost += vol * self.topo.comm_cost(ci, cj)
                else:
                    off_cost += vol * self.topo.comm_cost(ci, cj)
        return {"on_chip_cost": on_cost, "off_chip_cost": off_cost,
                "total_cost": on_cost + off_cost}

    def __repr__(self):
        return f"MultiChipEnvironment({self.topo})"
