"""
multi_chip_topology.py
----------------------
Defines a multi-chip system as a 2-D grid of chips connected by
off-chip links.  Each chip is a single-chip many-core mesh.

Key concepts (from the ACM paper 10.1145/3418498):
  - On-chip  links  : low-latency mesh links inside one chip
  - Off-chip links  : higher-latency links between chips (non-uniform cost)
  - Hierarchical communication cost modelled as alpha * (on-chip hops)
    + beta * (chip-to-chip hops)

HARDWARE FLEXIBILITY: grid SIZE (num_chips_x/y, rows/cols_per_chip) and
LATENCY VALUES (on_chip_latency/off_chip_latency) were already fully
configurable via the CLI without touching this file. What was previously
hardcoded is TOPOLOGY TYPE -- the routing/distance rule itself. This file
now supports `topology="mesh"` (original, edges only, the default -- no
behavior change) and `topology="torus"` (wraparound edges on both the
on-chip mesh and the inter-chip grid, one of the paper's own mentioned
alternative topologies, Sec 4.4). Genuinely different topology FAMILIES
(HNoC, dragonfly, or an arbitrary custom connectivity graph) are NOT
supported here -- that would need a different distance model entirely
(e.g. shortest-path over an arbitrary adjacency graph rather than a
closed-form hop-count formula), and is tracked as future work.
"""

import numpy as np


def _wrapped_delta(a: int, b: int, size: int) -> int:
    """Distance between two positions on a ring of the given size (torus
    wraparound) -- the shorter of going directly or going the other way
    around the edge."""
    d = abs(a - b)
    return min(d, size - d)


class MultiChipTopology:
    """Rectangular grid of chips, each a rows_per_chip x cols_per_chip mesh."""

    def __init__(
        self,
        num_chips_x: int = 2,
        num_chips_y: int = 2,
        rows_per_chip: int = 4,
        cols_per_chip: int = 4,
        on_chip_latency: float = 1.0,
        off_chip_latency: float = 5.0,
        topology: str = "mesh",
        routing_model: str = "legacy_distance",
    ):
        if topology not in ("mesh", "torus"):
            raise ValueError(f"Unknown topology '{topology}'. Choose 'mesh' or 'torus'.")
        if routing_model not in ("legacy_distance", "paper_xy"):
            raise ValueError("routing_model must be legacy_distance or paper_xy")
        if routing_model == "paper_xy" and topology != "mesh":
            raise ValueError("paper_xy routing currently supports the paper's mesh topology only")

        self.num_chips_x = num_chips_x
        self.num_chips_y = num_chips_y
        self.num_chips = num_chips_x * num_chips_y
        self.rows_per_chip = rows_per_chip
        self.cols_per_chip = cols_per_chip
        self.cores_per_chip = rows_per_chip * cols_per_chip
        self.total_cores = self.num_chips * self.cores_per_chip

        self.on_chip_latency = on_chip_latency
        self.off_chip_latency = off_chip_latency
        self.topology = topology
        self.routing_model = routing_model

    # ------------------------------------------------------------------
    # Core / chip indexing helpers
    # ------------------------------------------------------------------

    def global_core_id(self, chip_id: int, local_core_id: int) -> int:
        return chip_id * self.cores_per_chip + local_core_id

    def chip_and_local(self, global_id: int):
        chip_id = global_id // self.cores_per_chip
        local_id = global_id % self.cores_per_chip
        return chip_id, local_id

    def chip_xy(self, chip_id: int):
        return chip_id % self.num_chips_x, chip_id // self.num_chips_x

    def core_xy_global(self, global_id: int):
        """Return (chip_gx, chip_gy, local_row, local_col)."""
        chip_id, local_id = self.chip_and_local(global_id)
        cx, cy = self.chip_xy(chip_id)
        row = local_id // self.cols_per_chip
        col = local_id % self.cols_per_chip
        return cx, cy, row, col

    # ------------------------------------------------------------------
    # Communication cost (hierarchical, non-uniform)
    # ------------------------------------------------------------------

    def comm_cost(self, src_global: int, dst_global: int) -> float:
        """
        Hierarchical communication cost between two global core IDs.
        On-chip hops cost `on_chip_latency`; crossing a chip boundary
        adds `off_chip_latency` per chip hop.

        For `topology="torus"`, both the on-chip mesh and the inter-chip
        grid wrap around at the edges (row/col 0 is adjacent to the last
        row/col), so the effective distance in each dimension is the
        shorter of the direct and wraparound paths.
        """
        if self.routing_model == "paper_xy":
            route = self.xy_route(src_global, dst_global)
            return sum(self.on_chip_latency if kind == "on" else self.off_chip_latency
                       for kind, _ in route)

        scx, scy, sr, sc = self.core_xy_global(src_global)
        dcx, dcy, dr, dc = self.core_xy_global(dst_global)

        if self.topology == "torus":
            on_chip_hops = (_wrapped_delta(sr, dr, self.rows_per_chip) +
                             _wrapped_delta(sc, dc, self.cols_per_chip))
            chip_hops = (_wrapped_delta(scx, dcx, self.num_chips_x) +
                         _wrapped_delta(scy, dcy, self.num_chips_y))
        else:  # mesh (default, original behavior -- unchanged)
            on_chip_hops = abs(sr - dr) + abs(sc - dc)
            chip_hops = abs(scx - dcx) + abs(scy - dcy)

        return (on_chip_hops * self.on_chip_latency +
                chip_hops * self.off_chip_latency)

    @staticmethod
    def _axis_steps(start: int, end: int):
        step = 1 if end > start else -1
        return range(start, end, step)

    def _local_xy_route(self, chip_id, start_row, start_col, end_row, end_col):
        links = []
        col = start_col
        for current in self._axis_steps(start_col, end_col):
            nxt = current + (1 if end_col > start_col else -1)
            links.append(("on", (chip_id, start_row, current, start_row, nxt)))
            col = nxt
        row = start_row
        for current in self._axis_steps(start_row, end_row):
            nxt = current + (1 if end_row > start_row else -1)
            links.append(("on", (chip_id, current, col, nxt, col)))
            row = nxt
        return links

    def xy_route(self, src_global: int, dst_global: int):
        """Directed links for the paper-mode XY reconstruction.

        Same-chip traffic follows X then Y through the core mesh. Inter-chip
        traffic goes from the source core to a lower-left chip-periphery
        router, follows X then Y through the chip grid, and then travels from
        the destination periphery to its core. The lower-left gateway is an
        explicit interpretation of Figure 3; the paper does not publish a
        cycle-accurate GRS route implementation.
        """
        if not (0 <= src_global < self.total_cores and
                0 <= dst_global < self.total_cores):
            raise ValueError("core IDs must be within the topology")
        if src_global == dst_global:
            return []
        src_chip, src_local = self.chip_and_local(src_global)
        dst_chip, dst_local = self.chip_and_local(dst_global)
        sr, sc = divmod(src_local, self.cols_per_chip)
        dr, dc = divmod(dst_local, self.cols_per_chip)
        if src_chip == dst_chip:
            return self._local_xy_route(src_chip, sr, sc, dr, dc)

        gateway_row, gateway_col = self.rows_per_chip - 1, 0
        links = self._local_xy_route(src_chip, sr, sc, gateway_row, gateway_col)
        scx, scy = self.chip_xy(src_chip)
        dcx, dcy = self.chip_xy(dst_chip)
        chip_x, chip_y = scx, scy
        for current in self._axis_steps(scx, dcx):
            nxt = current + (1 if dcx > scx else -1)
            links.append(("off", (current, chip_y, nxt, chip_y)))
            chip_x = nxt
        for current in self._axis_steps(scy, dcy):
            nxt = current + (1 if dcy > scy else -1)
            links.append(("off", (chip_x, current, chip_x, nxt)))
            chip_y = nxt
        links.extend(self._local_xy_route(dst_chip, gateway_row, gateway_col, dr, dc))
        return links

    # ------------------------------------------------------------------
    # Full latency matrix (cached)
    # ------------------------------------------------------------------

    def latency_matrix(self) -> np.ndarray:
        """Return total_cores x total_cores latency matrix."""
        n = self.total_cores
        L = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                L[i, j] = self.comm_cost(i, j)
        return L

    # ------------------------------------------------------------------
    # Chip membership mask (useful for state encoding)
    # ------------------------------------------------------------------

    def chip_membership(self) -> np.ndarray:
        """Return array of length total_cores giving chip_id for each core."""
        return np.array(
            [g // self.cores_per_chip for g in range(self.total_cores)],
            dtype=np.int32,
        )

    def __repr__(self):
        return (
            f"MultiChipTopology({self.num_chips_x}x{self.num_chips_y} chips, "
            f"{self.rows_per_chip}x{self.cols_per_chip} cores/chip, "
            f"total={self.total_cores} cores, topology={self.topology}, "
            f"routing={self.routing_model})"
        )
