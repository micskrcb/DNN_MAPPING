"""
run_multi_chip.py
-----------------
Entry point to run core placement on a multi-chip system.
Mirrors the interface of the existing single-chip main.py.

Quick start:
    python run_multi_chip.py --algo ddpg
    python run_multi_chip.py --algo sa
    python run_multi_chip.py --algo random
    python run_multi_chip.py --algo ddpg --use_cnn
"""

import argparse
import math
import random
import sys
import os
import time
import hashlib
import json
import subprocess
from datetime import timedelta
import numpy as np

# Make sure the original src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_chip_topology import MultiChipTopology
from multi_chip_environment import MultiChipEnvironment
from compute_model import (PAPER_LOGIC_CORE_TARGETS, channel_ranges, tile_work,
                           compute_seconds, edge_bytes, paper_partition_grids)


def git_provenance():
    """Best-effort revision metadata for reproducible experiment reports."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            text=True, capture_output=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            text=True, capture_output=True).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def configure_cpu_threads(cpu_threads=None, interop_threads=1):
    """Configure PyTorch before model construction without oversubscribing CPUs."""
    if not HAS_TORCH:
        return None
    threads = cpu_threads or int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    if threads <= 0 or interop_threads <= 0:
        raise ValueError("CPU thread counts must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch permits setting this only before parallel work starts. This
        # can already have happened when main() is called repeatedly in tests.
        pass
    return threads


def valid_ppo_gamma(algorithm, gamma):
    """PPO-ASA may use gamma=0 for immediate proposal-ranking rewards."""
    return (0 <= gamma <= 1 if algorithm == "ppo_asa"
            else 0 < gamma <= 1)


def within_pool_ranking_loss(scores, gains):
    """Pairwise ordering only within a fully evaluated common-state pool."""
    difference = gains.unsqueeze(-1) - gains.unsqueeze(-2)
    margin = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    mask = difference > 1e-8
    return F.softplus(-margin[mask]).mean() if mask.any() else scores.sum() * 0.0


def sample_bandit_replay_indices(rng, size, batch_size, recent_fraction=0.0, recent_window=10000):
    """Mix recent samples with full-buffer samples; never oversample by reward."""
    if not 0 <= recent_fraction <= 1 or recent_window < 1 or not 0 < batch_size <= size:
        raise ValueError("invalid replay sampling configuration")
    if recent_fraction == 0:
        return rng.sample(range(size), batch_size)  # Preserve historical RNG behavior.
    window = min(size, recent_window)
    recent_count = min(window, round(batch_size * recent_fraction))
    chosen = rng.sample(range(size-window, size), recent_count)
    seen = set(chosen)
    while len(chosen) < batch_size:
        index = rng.randrange(size)
        if index not in seen:
            chosen.append(index)
            seen.add(index)
    return chosen


def proposal_training_reward(mode, proposed_reward, realized_reward,
                             planted_proxy):
    """Return the declared local reward for learned SA proposal selection."""
    if mode == "planted_proxy":
        return float(planted_proxy)
    if mode == "proposed_delta":
        return float(proposed_reward)
    if mode == "positive_delta":
        return float(max(proposed_reward, 0.0))
    if mode == "realized":
        return float(realized_reward)
    raise ValueError(f"unknown PPO-ASA reward mode: {mode}")


def proposal_focus_label(focus_fraction, focus_stage_slack):
    """Describe task focusing while preserving legacy zero-slack metadata."""
    if focus_fraction <= 0:
        return "uniform_tasks"
    if focus_stage_slack <= 0:
        if focus_fraction == 1.0:
            return "current_bottleneck_stage"
        return f"mixed_bottleneck_{focus_fraction:.3f}"
    if focus_fraction == 1.0:
        return f"stages_within_{focus_stage_slack:.3f}_of_bottleneck"
    return (f"mixed_near_bottleneck_{focus_fraction:.3f}_"
            f"slack_{focus_stage_slack:.3f}")


def proposal_cost_state(initial_cost, current_cost, best_cost, encoding):
    """Encode search progress without early clipping in the opt-in mode."""
    if encoding == "relative_cost":
        scale = max(abs(initial_cost), np.finfo(float).eps)
        return current_cost / scale, best_cost / scale
    scale = max(abs(initial_cost) * 0.05, np.finfo(float).eps)
    return (np.clip((initial_cost - current_cost) / scale, -2.0, 2.0),
            np.clip((initial_cost - best_cost) / scale, -2.0, 2.0))


def reorder_extracted_tasks(task_graph, task_labels, operations, kinds,
                            logic_task_order):
    """Apply an explicit, reproducible logical-core numbering convention.

    The paper specifies physical BS ordering but does not publish the logical
    VMM/VVA index order. Reindexing preserves the graph, work, and objective;
    it only changes which logical tasks sequential placement encounters first.
    """
    if logic_task_order == "output_group_interleaved":
        return task_graph, task_labels, operations, kinds
    if logic_task_order != "all_vmm_then_vva":
        raise ValueError(f"unknown logic task order: {logic_task_order}")

    vmm = [index for index, label in enumerate(task_labels)
           if "_VMM_" in label]
    vva = [index for index, label in enumerate(task_labels)
           if "_VVA_" in label]
    other = [index for index, label in enumerate(task_labels)
             if "_VMM_" not in label and "_VVA_" not in label]
    order = np.asarray(vmm + vva + other, dtype=np.int64)
    if len(order) != len(task_labels):
        raise ValueError("logic task order did not cover every extracted task")
    remapped_graph = task_graph[np.ix_(order, order)]
    remapped_labels = [task_labels[index] for index in order]
    remapped_operations = (operations[order] if operations is not None else None)
    remapped_kinds = ([kinds[index] for index in order]
                      if kinds is not None else None)
    return remapped_graph, remapped_labels, remapped_operations, remapped_kinds


# ---------------------------------------------------------------------------
# REAL WORKLOAD EXTRACTOR (CNN)
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 15 Layers = 15 Tasks to map to your cores
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.features(x)

    def _tv_model(ctor):
        """Handle both old (pretrained=False) and new (weights=None)
        torchvision APIs, and always skip downloading pretrained weights --
        we only need the architecture/shapes, not trained parameters."""
        try:
            return ctor(weights=None)
        except TypeError:
            return ctor(pretrained=False)

    _CONCAT_MERGE_FAMILIES = ("densenet", "unet")
    _ADD_MERGE_FAMILIES = ("resnet", "resnext", "regnet", "efficientnet")

    def _build_model_and_input(model_name: str, custom_model_path: str = None):
        """Returns (model, dummy_input, warning_or_None).

        Resolution order:
          1. model_name == "simple"      -> small built-in demo CNN
          2. custom_model_path is set    -> user-supplied model (see below)
          3. otherwise                   -> looked up dynamically in
                                             torchvision.models by name, so
                                             ANY torchvision classification
                                             model works (alexnet, vgg16,
                                             resnet50, resnet18, densenet121,
                                             mobilenet_v2, efficientnet_b0,
                                             googlenet, squeezenet1_0, ...)
                                             without needing a new branch
                                             added here per model.

        Custom models (--custom_model path/to/file.py): the file must define
            def build_model():
                ...
                return model                      # dummy input defaults to 1x3x224x224
            # or
                return model, dummy_input          # explicit input shape/dtype
        """
        model_name = model_name.lower()

        if model_name == "simple" and custom_model_path is None:
            return SimpleCNN(), torch.randn(1, 3, 32, 32), None

        if custom_model_path is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_model_module", custom_model_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load --custom_model file: {custom_model_path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "build_model"):
                raise RuntimeError(
                    f"{custom_model_path} must define a build_model() function returning "
                    f"either `model` or `(model, dummy_input)`."
                )
            result = mod.build_model()
            if isinstance(result, tuple):
                model, dummy_input = result
            else:
                model, dummy_input = result, torch.randn(1, 3, 224, 224)
            model.eval()
            return model, dummy_input, None

        if not HAS_TORCHVISION:
            raise RuntimeError(
                f"--model {model_name} requires torchvision. Install it with "
                f"'pip install torchvision --break-system-packages'."
            )
        if not hasattr(tv_models, model_name):
            raise ValueError(
                f"Unknown model '{model_name}': not 'simple', no --custom_model given, "
                f"and not found in torchvision.models. Try e.g. alexnet, vgg16, resnet50, "
                f"resnet18, densenet121, mobilenet_v2, efficientnet_b0, googlenet, "
                f"squeezenet1_0 -- or pass --custom_model for your own architecture."
            )

        dummy_input = torch.randn(1, 3, 224, 224)  # standard ImageNet size (paper Sec 4.1)
        model = _tv_model(getattr(tv_models, model_name))
        model.eval()

        warning = None
        if any(k in model_name for k in _CONCAT_MERGE_FAMILIES):
            warning = ("Branch dependencies are traced, but concatenation timing is unsupported.")
        elif any(k in model_name for k in _ADD_MERGE_FAMILIES):
            warning = ("Residual dependencies are traced; add arithmetic is assigned to the "
                       "destination transformation/VVA path without a separate logic core.")
        return model, dummy_input, warning

    def extract_model_task_graph(model_name: str = "simple", channels_per_partition: int = 8,
                                  custom_model_path: str = None, return_work: bool = False,
                                  partition_mode: str = "uniform",
                                  partition_balance_weight: float = 1.0,
                                  vva_ops_per_cycle: float = 1.0):
        """
        PAPER FIX (Sec 3.1.1, Fig. 5-6): partition each CONV/FC layer's
        weights along the input channel C and output channel K into a grid
        of VMM (vector-matrix-multiply) cores, each feeding a VVA (vector-
        vector-accumulation) core that reduces the partial sums for its
        output-channel group.

        Activation (ReLU) and pooling are NOT separate tasks: per the
        paper's core architecture (Sec 2.2), these run inside each core's
        transformation unit, so only Conv2d/Linear layers are partitioned.

        Uses torch.fx for Conv2d/Linear dependency discovery. Channel ranges
        are routed through pooling/flatten using consumer shapes. Residual
        branch traffic is retained and addition is assigned to the destination
        transformation/VVA path without a separate task. Concatenation, grouped
        convolutions, and channel-changing merges are rejected. This is a
        restricted extractor, not an exact simulator for every architecture.

        Args:
            model_name: "simple" (small demo CNN), any model name found in
                `torchvision.models` (e.g. "alexnet", "vgg16", "resnet50",
                "resnet18", "densenet121", "mobilenet_v2", ...), or ignored
                if `custom_model_path` is given.
            channels_per_partition: max output channels per partition group
                (smaller -> more, finer-grained logic cores, closer to the
                paper's per-model core counts in Fig. 6). Set to 0 to
                restore the unpartitioned one-task-per-layer behavior
                (useful for quick smoke tests). Real models have far more
                channels than the demo CNN -- start with a LARGER value
                or you'll generate more logic cores than any reasonable
                grid can hold; see the pre-flight check in main().
            custom_model_path: path to a Python file defining
                `build_model()` (see `_build_model_and_input` docstring)
                for a user-supplied architecture instead of a built-in or
                torchvision one.

        Returns:
            task_graph: directed volume matrix, task_graph[i, j] = elements
                sent from logic core i to logic core j.
            num_tasks: total logic core count.
            labels: human-readable label per task index, for debugging.
        """
        model, dummy_input, warning = _build_model_and_input(model_name, custom_model_path)
        if warning:
            print(f"[WARN] {warning}")

        import torch.fx as fx
        from torch.fx.passes.shape_prop import ShapeProp

        try:
            traced = fx.symbolic_trace(model)
        except Exception as e:
            raise RuntimeError(
                f"torch.fx.symbolic_trace failed for '{model_name}': {e}\n"
                f"This usually means the model has data-dependent control flow (a "
                f"Python if/loop driven by a tensor's VALUE, not just its shape) that "
                f"torch.fx cannot trace symbolically. This is an inherent torch.fx "
                f"limitation -- not every architecture is traceable this way."
            ) from e

        with torch.no_grad():
            ShapeProp(traced).propagate(dummy_input)

        # Identify logic nodes: call_module fx nodes targeting Conv2d/Linear.
        logic_nodes = {}  # fx.Node -> (name, kind, C_in, C_out, H, W)
        kernels = {}
        input_shapes = {}
        for node in traced.graph.nodes:
            if node.op != "call_module":
                continue
            submodule = traced.get_submodule(node.target)
            shape = node.meta["tensor_meta"].shape
            if isinstance(submodule, nn.Conv2d):
                if submodule.groups != 1:
                    raise ValueError("Grouped/depthwise Conv2d partitioning is not implemented")
                kernels[node] = math.prod(submodule.kernel_size)
                input_shapes[node] = tuple(node.args[0].meta["tensor_meta"].shape)
                logic_nodes[node] = (node.target, "conv", submodule.in_channels,
                                      shape[1], shape[2], shape[3])
            elif isinstance(submodule, nn.Linear):
                kernels[node] = 1
                input_shapes[node] = tuple(node.args[0].meta["tensor_meta"].shape)
                logic_nodes[node] = (node.target, "linear", submodule.in_features,
                                      submodule.out_features, 1, 1)

        if not logic_nodes:
            raise RuntimeError(
                f"No Conv2d/Linear layers found via torch.fx trace of '{model_name}'."
            )

        # node_list preserves fx graph order, which is already a valid
        # topological order of the traced computation.
        node_list = [n for n in traced.graph.nodes if n in logic_nodes]
        if return_work:
            import operator
            if any(n.op == "call_function" and n.target is torch.cat
                   for n in traced.graph.nodes):
                raise ValueError("Full-frame timing does not model concatenation merges")
            if any(n.op == "call_function" and n.target in (operator.add, torch.add)
                   or n.op == "call_method" and n.target in ("add", "add_")
                   for n in traced.graph.nodes):
                print("[ASSUMPTION] Residual-add traffic is traced from both branches; "
                      "the add executes in the destination transformation/VVA path and "
                      "does not consume a separate Figure-6 logic core.")

        def downstream_logic_nodes(start):
            """BFS forward through non-logic nodes to find the TRUE next
            logic layer(s) -- validated separately against a mock graph
            with a branch+merge (see conversation record); correctly
            surfaces multiple predecessors/successors at branch/merge
            points instead of assuming a single linear successor."""
            found = []
            visited = set()
            frontier = list(start.users.keys())
            while frontier:
                n = frontier.pop()
                if n in visited:
                    continue
                visited.add(n)
                if n in logic_nodes:
                    found.append(n)
                    continue
                frontier.extend(n.users.keys())
            return found

        if channels_per_partition <= 0:
            if return_work:
                raise ValueError("Full-frame timing requires channel partitioning > 0")
            # Legacy behavior: one task per Conv2d/Linear layer, wired
            # along the TRUE graph edges (still generalized -- a branching
            # model gets branching edges even in legacy/unpartitioned mode).
            id_of = {n: i for i, n in enumerate(node_list)}
            num_tasks = len(node_list)
            task_graph = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            labels = []
            for n in node_list:
                name, kind, C_in, C_out, H, W = logic_nodes[n]
                labels.append(f"{name}_{kind}")
                vol = C_out * H * W
                for dst in downstream_logic_nodes(n):
                    task_graph[id_of[n], id_of[dst]] += vol
            return task_graph, num_tasks, labels

        if partition_mode not in ("uniform", "paper_targets"):
            raise ValueError("partition_mode must be uniform or paper_targets")
        paper_grids = {}
        if partition_mode == "paper_targets":
            normalized_model = model_name.lower()
            if custom_model_path is not None or normalized_model not in PAPER_LOGIC_CORE_TARGETS:
                raise ValueError("paper_targets supports alexnet, vgg16 and resnet50 only")
            for region_kind in ("conv", "linear"):
                region_nodes = [node for node in node_list if logic_nodes[node][1] == region_kind]
                layer_specs = []
                for node in region_nodes:
                    name, _, cin, cout, height, width = logic_nodes[node]
                    layer_specs.append({"name": name, "cin": cin, "cout": cout,
                                        "height": height, "width": width,
                                        "kernel_area": kernels[node]})
                target = PAPER_LOGIC_CORE_TARGETS[normalized_model][region_kind]
                grids = paper_partition_grids(
                    layer_specs, target,
                    compute_balance_weight=partition_balance_weight,
                    vva_ops_per_cycle=vva_ops_per_cycle)
                paper_grids.update(zip(region_nodes, grids))

        # --- Channel-partitioned extraction (paper Sec 3.1.1), generalized
        # to arbitrary graph topology. M/N are now computed independently
        # per layer from its OWN (C_in, C_out) -- not tied to a specific
        # predecessor's M -- since a layer can have zero, one, or several
        # true predecessors once branching/merging is possible. This can
        # change task counts substantially versus the old hook extractor;
        # counts are not calibrated to the paper's compute-balanced split.
        layer_ids = {}   # node -> (vmm_ids[M][N], vva_ids[M], M, N, partial_vol)
        labels = []
        breakdown_rows = []
        operations, task_kinds = [], []
        next_id = 0

        for n in node_list:
            name, kind, C_in, C_out, H, W = logic_nodes[n]
            if partition_mode == "paper_targets":
                M, N = paper_grids[n]
            else:
                M = max(1, math.ceil(C_out / channels_per_partition))
                N = max(1, math.ceil(C_in / channels_per_partition))
            work, kinds = tile_work(C_in, C_out, H, W, kernels[n], N, M)
            operations.extend(work)
            task_kinds.extend(kinds)

            vmm_ids = [[None] * N for _ in range(M)]
            vva_ids = [None] * M
            out_per_group = math.ceil(C_out / M)
            partial_vol = out_per_group * H * W

            edges_here = []
            for m in range(M):
                lo, hi = channel_ranges(C_out, M)[m]
                partial_vol = (hi - lo) * H * W
                for k in range(N):
                    vmm_ids[m][k] = next_id
                    labels.append(f"{name}_{kind}_VMM_m{m}_n{k}")
                    next_id += 1
                vva_ids[m] = next_id
                labels.append(f"{name}_{kind}_VVA_m{m}")
                next_id += 1
                for k in range(N):
                    edges_here.append((vmm_ids[m][k], vva_ids[m], partial_vol))

            layer_ids[n] = (vmm_ids, vva_ids, M, N, partial_vol, edges_here)
            breakdown_rows.append((name, kind, C_in, C_out, M, N, M * N, M))

        partition_label = ("Figure-6 aggregate targets with explicit reconstructed per-layer grids"
                           if partition_mode == "paper_targets"
                           else f"channels_per_partition={channels_per_partition}")
        print(f">> Per-layer VMM/VVA breakdown ({model_name}, {partition_label}, "
              f"extracted via torch.fx true computational-graph tracing):")
        print(f"   {'Layer':<30} {'Type':<8} {'Cin':>6} {'Cout':>6} {'M':>4} {'N':>4} {'VMM':>6} {'VVA':>5}")
        for name, kind, cin, cout, M, N, vmm, vva in breakdown_rows:
            print(f"   {name:<30} {kind:<8} {cin:>6} {cout:>6} {M:>4} {N:>4} {vmm:>6} {vva:>5}")

        # Within-layer VMM->VVA edges, plus TRUE cross-layer edges found via
        # graph traversal -- every real predecessor/successor pair gets
        # connected (all-to-all between predecessor's M output groups and
        # successor's M*N input-side cores), correctly representing branch
        # and merge points instead of assuming one linear predecessor.
        edges = []
        for n in node_list:
            _, _, _, _, _, edges_here = layer_ids[n]
            edges.extend(edges_here)

        for src in node_list:
            _, vva_src, M_src, _, _, _ = layer_ids[src]
            for dst in downstream_logic_nodes(src):
                vmm_dst, _, M_dst, N_dst, _, _ = layer_ids[dst]
                src_channels = logic_nodes[src][3]
                dst_channels = logic_nodes[dst][2]
                # Flatten after pooling: contiguous spatial features belong to
                # their source channel. Use consumer shapes, not pre-pool H/W.
                flatten = logic_nodes[src][1] == "conv" and logic_nodes[dst][1] == "linear"
                if flatten and dst_channels % src_channels:
                    raise ValueError("Cannot map flattened channel ranges exactly")
                factor = dst_channels // src_channels if flatten else 1
                if not flatten and src_channels != dst_channels:
                    raise ValueError("Channel-changing merge is unsupported; refusing all-to-all approximation")
                spatial = math.prod(input_shapes[dst][2:]) if len(input_shapes[dst]) > 2 else 1
                for m in range(M_src):
                    slo, shi = channel_ranges(src_channels, M_src)[m]
                    slo, shi = slo * factor, shi * factor
                    for m2 in range(M_dst):
                        for k2 in range(N_dst):
                            dlo, dhi = channel_ranges(dst_channels, N_dst)[k2]
                            overlap = max(0, min(shi, dhi) - max(slo, dlo))
                            if overlap:
                                edges.append((vva_src[m], vmm_dst[m2][k2], overlap * spatial))

        num_tasks = next_id
        task_graph = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        for src, dst, vol in edges:
            task_graph[src, dst] += vol

        if return_work:
            return task_graph, num_tasks, labels, np.asarray(operations), task_kinds
        return task_graph, num_tasks, labels

    def extract_cnn_task_graph(channels_per_partition: int = 8):
        """Backward-compatible alias for extract_model_task_graph('simple', ...)."""
        return extract_model_task_graph("simple", channels_per_partition)


# ---------------------------------------------------------------------------
# DDPG Architecture (Following the ACM Paper Pseudocode)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim=2, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
                nn.Tanh()  # Action bounds [-1, 1] for continuous space
            )

        def forward(self, state):
            return self.net(state)

    class Critic(nn.Module):
        def __init__(self, state_dim, action_dim=2, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)  # Q-Value
            )

        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            return self.net(x)

    class SpatialEncoder(nn.Module):
        """Encode the policy's contiguous 2-D placement grid.

        The mapper owns this grid representation and converts it to
        chip-major physical IDs only when evaluating a placement. Keeping the
        encoder on the policy grid prevents the coordinate mismatch in the
        earlier CNN prototype.
        """
        def __init__(self, rows, cols):
            super().__init__()
            if rows < 4 or cols < 4:
                raise ValueError("CNN agent requires a policy grid of at least 4x4")
            self.rows, self.cols = rows, cols
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            with torch.no_grad():
                self.feature_dim = int(self.net(torch.zeros(1, 1, rows, cols)).numel())

        def forward(self, grid):
            return torch.flatten(self.net(grid), start_dim=1)

    class CNNActor(nn.Module):
        """Spatial actor ported from the junior prototype, with current state semantics."""
        def __init__(self, rows, cols, num_tasks, action_dim=2, hidden=256):
            super().__init__()
            self.rows, self.cols = rows, cols
            self.encoder = SpatialEncoder(rows, cols)
            self.head = nn.Sequential(
                nn.Linear(self.encoder.feature_dim + num_tasks, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim), nn.Tanh(),
            )

        def forward(self, state):
            grid_size = self.rows * self.cols
            grid = state[:, :grid_size].reshape(-1, 1, self.rows, self.cols)
            return self.head(torch.cat([self.encoder(grid), state[:, grid_size:]], dim=1))

    class CNNCritic(nn.Module):
        """Spatial critic that combines grid features, task communication and action."""
        def __init__(self, rows, cols, num_tasks, action_dim=2, hidden=256):
            super().__init__()
            self.rows, self.cols = rows, cols
            self.encoder = SpatialEncoder(rows, cols)
            self.head = nn.Sequential(
                nn.Linear(self.encoder.feature_dim + num_tasks + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
            )

        def forward(self, state, action):
            grid_size = self.rows * self.cols
            grid = state[:, :grid_size].reshape(-1, 1, self.rows, self.cols)
            features = torch.cat([self.encoder(grid), state[:, grid_size:], action], dim=1)
            return self.head(features)

    class PaperSpatialEncoder(nn.Module):
        """Figure 9 spatial stack; padding and LRN parameters are assumptions."""
        def __init__(self, rows, cols):
            super().__init__()
            if rows < 4 or cols < 4:
                raise ValueError("Paper CNN agent requires a policy grid of at least 4x4")
            self.rows, self.cols = rows, cols
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.LocalResponseNorm(5),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2), nn.LocalResponseNorm(5),
            )
            with torch.no_grad():
                self.feature_dim = int(self.net(torch.zeros(1, 1, rows, cols)).numel())

        def forward(self, grid):
            return torch.flatten(self.net(grid), start_dim=1)

    class PaperCNNActor(nn.Module):
        """Actor from paper Figure 9: CONV32/64, FC600/300, FC(2z)."""
        def __init__(self, rows, cols, num_tasks, action_dim=2):
            super().__init__()
            self.rows, self.cols = rows, cols
            self.encoder = PaperSpatialEncoder(rows, cols)
            self.fc1 = nn.Linear(self.encoder.feature_dim, 600)
            self.bn1 = nn.BatchNorm1d(600)
            self.fc2 = nn.Linear(600, 300)
            self.bn2 = nn.BatchNorm1d(300)
            self.output = nn.Linear(300, action_dim)

        def forward(self, state):
            grid_size = self.rows * self.cols
            grid = state[:, :grid_size].reshape(-1, 1, self.rows, self.cols)
            x = F.relu(self.bn1(self.fc1(self.encoder(grid))))
            x = F.relu(self.bn2(self.fc2(x)))
            return torch.tanh(self.output(x))

    class PaperCNNCritic(nn.Module):
        """Critic from Figure 9; action merges after FC600 as stated in text."""
        def __init__(self, rows, cols, num_tasks, action_dim=2):
            super().__init__()
            self.rows, self.cols = rows, cols
            self.encoder = PaperSpatialEncoder(rows, cols)
            self.fc1 = nn.Linear(self.encoder.feature_dim, 600)
            self.bn1 = nn.BatchNorm1d(600)
            self.fc2 = nn.Linear(600 + action_dim, 300)
            self.bn2 = nn.BatchNorm1d(300)
            self.output = nn.Linear(300, 1)

        def forward(self, state, action):
            grid_size = self.rows * self.cols
            grid = state[:, :grid_size].reshape(-1, 1, self.rows, self.cols)
            x = F.relu(self.bn1(self.fc1(self.encoder(grid))))
            x = F.relu(self.bn2(self.fc2(torch.cat([x, action], dim=1))))
            return self.output(x)

    class ReplayBuffer:
        def __init__(self, capacity=50000):
            self.buffer = []
            self.ptr = 0
            self.capacity = capacity

        def add(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.ptr] = (state, action, reward, next_state, done)
            self.ptr = (self.ptr + 1) % self.capacity

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)

    class GuidedReplayBuffer:
        """Prioritized replay for complete-episode return targets.

        Guided DDPG deliberately trains the critic on Monte-Carlo returns.
        This makes every transition in a sparse-reward placement trajectory
        informative and avoids bootstrapping a terminal reward through tens
        or hundreds of placement decisions.  Demonstration transitions are
        sampled explicitly so that a single ASA trajectory cannot disappear
        in a much larger online replay buffer.
        """
        def __init__(self, capacity=100000, demo_fraction=0.25, alpha=0.6,
                     seed=None):
            self.buffer = []
            self.priorities = np.zeros(capacity, dtype=np.float64)
            self.ptr = 0
            self.capacity = capacity
            self.demo_fraction = demo_fraction
            self.alpha = alpha
            # Keep replay sampling separate from environment exploration.
            # Training performs many more replay draws than a no-learning
            # control; sharing NumPy's global RNG would therefore give the
            # two conditions different exploration despite the same seed.
            self.rng = np.random.default_rng(seed)

        def add(self, state, action, return_target, is_demo=False, priority=None):
            entry = (np.asarray(state, dtype=np.float32),
                     np.asarray(action, dtype=np.float32),
                     float(return_target), bool(is_demo))
            if len(self.buffer) < self.capacity:
                self.buffer.append(entry)
            else:
                # DDPG-from-demonstrations keeps expert transitions in replay.
                # Without this guard, a 100k buffer would erase the AlexNet
                # demonstration around placement 1,638, while behavior cloning
                # is still active by default through placement 2,000.
                start = self.ptr
                while self.buffer[self.ptr][3]:
                    self.ptr = (self.ptr + 1) % self.capacity
                    if self.ptr == start:
                        raise RuntimeError(
                            "guided replay is full of permanent demonstrations")
                self.buffer[self.ptr] = entry
            if priority is None:
                priority = self.priorities[:len(self.buffer)].max(initial=1.0)
            self.priorities[self.ptr] = max(float(priority), 1e-6)
            self.ptr = (self.ptr + 1) % self.capacity

        def sample(self, batch_size, beta=0.4):
            if not self.buffer:
                raise ValueError("cannot sample an empty replay buffer")
            size = len(self.buffer)
            demo_indices = np.array(
                [i for i, item in enumerate(self.buffer) if item[3]], dtype=np.int64)
            demo_count = min(len(demo_indices), int(round(batch_size * self.demo_fraction)))
            chosen = []
            if demo_count:
                chosen.extend(self.rng.choice(
                    demo_indices, demo_count, replace=True).tolist())
            remaining = batch_size - len(chosen)
            scaled = np.maximum(self.priorities[:size], 1e-6) ** self.alpha
            probabilities = scaled / scaled.sum()
            if remaining:
                chosen.extend(self.rng.choice(
                    size, remaining, replace=True, p=probabilities).tolist())
            indices = np.asarray(chosen, dtype=np.int64)
            batch = [self.buffer[index] for index in indices]
            states, actions, returns, demos = map(np.asarray, zip(*batch))
            sample_probabilities = np.maximum(probabilities[indices], 1e-12)
            weights = (size * sample_probabilities) ** (-beta)
            weights /= weights.max(initial=1.0)
            return (states, actions, returns.astype(np.float32),
                    demos.astype(bool), indices, weights.astype(np.float32))

        def update_priorities(self, indices, priorities):
            for index, priority in zip(indices, priorities):
                self.priorities[int(index)] = max(float(priority), 1e-6)

        def __len__(self):
            return len(self.buffer)

    class DDPGAgent:
        def __init__(self, state_dim, action_dim=2, lr_actor=2e-4, lr_critic=1e-3,
                     gamma=0.98, tau=0.005, device=None, agent_arch="mlp",
                     rows=None, cols=None, num_tasks=None, stable=False):
            # PERF FIX: this agent previously never checked for a GPU, even
            # if one was available -- for a partitioned CNN workload the
            # state vector is total_cores + num_tasks (e.g. 4096+906=5002
            # dims), so the first Linear layer alone has ~1.3M parameters.
            # On a CUDA GPU this is dramatically faster per training step.
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            if agent_arch not in ("mlp", "cnn", "paper_cnn"):
                raise ValueError(f"Unknown agent architecture: {agent_arch}")
            self.agent_arch = agent_arch
            self.stable = stable
            self.training_updates = 0

            self.action_dim = action_dim
            if agent_arch in ("cnn", "paper_cnn"):
                if rows is None or cols is None or num_tasks is None:
                    raise ValueError("CNN agent requires rows, cols and num_tasks")
                actor_cls, critic_cls = ((PaperCNNActor, PaperCNNCritic)
                                         if agent_arch == "paper_cnn" else (CNNActor, CNNCritic))
                actor_args = (rows, cols, num_tasks, action_dim)
                critic_args = (rows, cols, num_tasks, action_dim)
            else:
                actor_cls, critic_cls = Actor, Critic
                actor_args, critic_args = (state_dim, action_dim), (state_dim, action_dim)
            self.actor = actor_cls(*actor_args).to(self.device)
            self.actor_target = actor_cls(*actor_args).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

            self.critic = critic_cls(*critic_args).to(self.device)
            self.critic_target = critic_cls(*critic_args).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

            if stable:
                self.critic2 = critic_cls(*critic_args).to(self.device)
                self.critic2_target = critic_cls(*critic_args).to(self.device)
                self.critic2_target.load_state_dict(self.critic2.state_dict())
                self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

            self.gamma = gamma
            self.tau = tau
            self.ou_state = np.zeros(action_dim, dtype=np.float32)

        def score_actions(self, state, actions):
            """Return conservative Q estimates for legal candidate actions."""
            states = torch.as_tensor(state, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
            actions_tensor = torch.as_tensor(actions, dtype=torch.float32,
                                             device=self.device)
            states = states.expand(len(actions_tensor), -1)
            self.critic.eval()
            if self.stable:
                self.critic2.eval()
            with torch.inference_mode():
                values = self.critic(states, actions_tensor)
                if self.stable:
                    values = torch.minimum(values, self.critic2(states, actions_tensor))
            self.critic.train()
            if self.stable:
                self.critic2.train()
            return values.squeeze(1).cpu().numpy()

        def train_guided(self, replay_buffer, batch_size=64, bc_weight=1.0,
                         policy_delay=2, max_grad_norm=1.0):
            """Stable twin-critic update using prioritized episode returns."""
            if len(replay_buffer) < 2:
                return None
            batch_size = min(batch_size, max(2, len(replay_buffer)))
            states, actions, returns, demos, indices, weights = replay_buffer.sample(batch_size)
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            targets = torch.as_tensor(returns, dtype=torch.float32,
                                      device=self.device).unsqueeze(1)
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32,
                                             device=self.device).unsqueeze(1)

            self.critic.train()
            self.critic2.train()
            current1 = self.critic(states, actions)
            current2 = self.critic2(states, actions)
            loss1_values = F.smooth_l1_loss(current1, targets, reduction="none")
            loss2_values = F.smooth_l1_loss(current2, targets, reduction="none")
            critic1_loss = (loss1_values * weights_tensor).mean()
            critic2_loss = (loss2_values * weights_tensor).mean()

            self.critic_optimizer.zero_grad()
            critic1_loss.backward()
            critic1_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            critic2_grad = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_grad_norm)
            self.critic2_optimizer.step()

            td_error = torch.maximum((current1.detach() - targets).abs(),
                                     (current2.detach() - targets).abs())
            replay_buffer.update_priorities(indices,
                                            td_error.squeeze(1).cpu().numpy() + 1e-6)

            self.training_updates += 1
            actor_loss_value = None
            bc_loss_value = None
            actor_grad_value = None
            if self.training_updates % policy_delay == 0:
                self.critic.eval()
                self.critic.requires_grad_(False)
                predicted = self.actor(states)
                policy_loss = -self.critic(states, predicted).mean()
                demo_mask = torch.as_tensor(demos, dtype=torch.bool, device=self.device)
                if demo_mask.any():
                    bc_loss = F.mse_loss(predicted[demo_mask], actions[demo_mask])
                else:
                    bc_loss = torch.zeros((), device=self.device)
                actor_loss = policy_loss + float(bc_weight) * bc_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                self.actor_optimizer.step()
                self.critic.requires_grad_(True)
                self.critic.train()
                actor_loss_value = float(actor_loss.detach().cpu())
                bc_loss_value = float(bc_loss.detach().cpu())
                actor_grad_value = float(torch.as_tensor(actor_grad).detach().cpu())

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for online, target in ((self.critic, self.critic_target),
                                   (self.critic2, self.critic2_target)):
                for param, target_param in zip(online.parameters(), target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            with torch.no_grad():
                pairs = [(self.actor, self.actor_target),
                         (self.critic, self.critic_target),
                         (self.critic2, self.critic2_target)]
                for online, target in pairs:
                    for source, destination in zip(online.buffers(), target.buffers()):
                        destination.copy_(source)

            q_abs_max = float(torch.maximum(current1.detach().abs().max(),
                                            current2.detach().abs().max()).cpu())
            if not math.isfinite(q_abs_max):
                raise FloatingPointError("guided DDPG critic produced a non-finite Q value")
            return {
                "actor_loss": actor_loss_value,
                "critic_loss": float((critic1_loss + critic2_loss).detach().cpu() / 2),
                "bc_loss": bc_loss_value,
                "q_abs_max": q_abs_max,
                "critic_grad_norm": float(max(torch.as_tensor(critic1_grad),
                                               torch.as_tensor(critic2_grad)).detach().cpu()),
                "actor_grad_norm": actor_grad_value,
            }

        def select_action(self, state, noise_scale=0.1, explore=True, rng=None):
            state = torch.as_tensor(state, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            self.actor.eval()
            with torch.inference_mode():
                action = self.actor(state).squeeze(0).cpu().numpy()
            self.actor.train()
            if not explore:
                return action
            # Add exploration noise.
            # Ornstein-Uhlenbeck exploration used by the paper (Sec. 3.2).
            noise = (np.random.randn(self.action_dim) if rng is None else
                     rng.standard_normal(self.action_dim))
            self.ou_state += 0.15 * (0.0 - self.ou_state) + 0.2 * noise
            action += noise_scale * self.ou_state
            return np.clip(action, -1.0, 1.0)

        def train(self, replay_buffer, batch_size=64):
            if len(replay_buffer) < batch_size:
                return None

            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32,
                                      device=self.device).unsqueeze(1)
            next_states = torch.as_tensor(next_states, dtype=torch.float32,
                                          device=self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32,
                                    device=self.device).unsqueeze(1)

            # Target values must use fixed running statistics, not depend on
            # the other transitions sampled into the minibatch.
            self.actor_target.eval()
            self.critic_target.eval()
            self.critic.train()
            self.actor.train()
            # Critic Update
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                target_Q = self.critic_target(next_states, next_actions)
                target_Q = rewards + (1 - dones) * self.gamma * target_Q

            current_Q = self.critic(states, actions)
            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Evaluate the critic without updating its BatchNorm statistics
            # or accumulating critic gradients during the actor-only step.
            self.critic.eval()
            self.critic.requires_grad_(False)
            # Actor Update
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True)
            self.critic.train()

            # Soft Update Targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # BatchNorm buffers are state too; synchronize running means,
            # variances and counters with the online models.
            with torch.no_grad():
                for online, target in ((self.actor, self.actor_target),
                                       (self.critic, self.critic_target)):
                    for source, destination in zip(online.buffers(), target.buffers()):
                        destination.copy_(source)
            return {"actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu())}

        def state_dict(self) -> dict:
            """Everything needed to exactly resume this agent's networks/optimizers."""
            result = {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "stable": self.stable,
                "training_updates": self.training_updates,
            }
            if self.stable:
                result.update({
                    "critic2": self.critic2.state_dict(),
                    "critic2_target": self.critic2_target.state_dict(),
                    "critic2_optimizer": self.critic2_optimizer.state_dict(),
                })
            return result

        def load_state_dict(self, sd: dict) -> None:
            self.actor.load_state_dict(sd["actor"])
            self.actor_target.load_state_dict(sd["actor_target"])
            self.critic.load_state_dict(sd["critic"])
            self.critic_target.load_state_dict(sd["critic_target"])
            self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
            self.critic_optimizer.load_state_dict(sd["critic_optimizer"])
            self.training_updates = sd.get("training_updates", 0)
            if self.stable:
                self.critic2.load_state_dict(sd["critic2"])
                self.critic2_target.load_state_dict(sd["critic2_target"])
                self.critic2_optimizer.load_state_dict(sd["critic2_optimizer"])


# ---------------------------------------------------------------------------
# MultiChipCoreMapper (Translates continuous actions to discrete grid)
# ---------------------------------------------------------------------------

class MultiChipCoreMapper:
    def __init__(self, env: MultiChipEnvironment, baseline_latency: float = None, batch_z: int = 3,
                 reward_mode: str = "sparse", shaping_gamma: float = 0.98,
                 reward_scale: float = 1.0, compact_state: bool = False):
        """
        Args:
            baseline_latency: B in the paper's reward r_t = sqrt(B) - sqrt(L(P))
                (Algorithm 1, line 11) -- the best latency found by random
                search, computed once before training and held fixed. If
                None, the reward falls back to -sqrt(L(P)) (unnormalized)
                and a one-time warning is printed, since this is NOT what
                the paper specifies.
            batch_z: number of unplaced logic cores assigned per action,
                per the paper's action representation [x1,y1,...,xz,yz]
                (Sec 3.2, 'Representation of Assigning Placements'). The
                paper doesn't fix a specific z; pick one that divides
                reasonably into num_tasks, or leave the remainder-batch
                handling (below) to place fewer than z on the final step.
        """
        self.env = env
        topo = env.topo
        self.total_cols = topo.cols_per_chip * topo.num_chips_x
        self.total_rows = topo.rows_per_chip * topo.num_chips_y
        self.num_tasks  = env.num_tasks
        self.batch_z    = max(1, batch_z)
        self._placement = np.full(self.num_tasks, -1, dtype=np.int32)
        self._occupied  = set()
        physical = env.allowed_cores
        chip = physical // topo.cores_per_chip
        local = physical % topo.cores_per_chip
        chip_x = chip % topo.num_chips_x
        chip_y = chip // topo.num_chips_x
        grid_x = chip_x * topo.cols_per_chip + local % topo.cols_per_chip
        grid_y = chip_y * topo.rows_per_chip + local // topo.cols_per_chip
        self._allowed_grid = np.sort(grid_y * self.total_cols + grid_x)
        self._allowed_grid_mask = np.zeros(self.total_rows * self.total_cols, dtype=bool)
        self._allowed_grid_mask[self._allowed_grid] = True
        self._region_min_x, self._region_max_x = int(grid_x.min()), int(grid_x.max())
        self._region_min_y, self._region_max_y = int(grid_y.min()), int(grid_y.max())
        self.compact_state = compact_state
        self.state_cols = (self._region_max_x - self._region_min_x + 1
                           if compact_state else self.total_cols)
        self.state_rows = (self._region_max_y - self._region_min_y + 1
                           if compact_state else self.total_rows)
        self._state_index_by_grid = np.full(self.total_rows * self.total_cols, -1,
                                            dtype=np.int64)
        if compact_state:
            local_x = grid_x - self._region_min_x
            local_y = grid_y - self._region_min_y
            self._state_index_by_grid[self._allowed_grid] = local_y * self.state_cols + local_x
        else:
            self._state_index_by_grid[self._allowed_grid] = self._allowed_grid
        self._task_ptr  = 0
        self.collision_repairs = 0
        self.occupied_collision_repairs = 0
        self.masked_region_repairs = 0
        self.intended_cores = set()
        if reward_mode not in ("sparse", "potential"):
            raise ValueError("reward_mode must be sparse or potential")
        self.reward_mode = reward_mode
        self.shaping_gamma = shaping_gamma
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.reward_scale = reward_scale
        self._potential = 0.0
        self.baseline_latency = baseline_latency
        self._potential_normalizer = (
            math.sqrt(max(self.reward_scale * baseline_latency, 0.0))
            if baseline_latency is not None else 1.0)
        if self._potential_normalizer == 0:
            self._potential_normalizer = 1.0
        if baseline_latency is None:
            print("[WARN] MultiChipCoreMapper created without baseline_latency -- "
                  "reward will use unnormalized -sqrt(L(P)), not the paper's "
                  "sqrt(B) - sqrt(L(P)) (Algorithm 1, line 11).")

    def reset(self):
        self._placement[:] = -1
        self._occupied.clear()
        self._task_ptr = 0
        self.collision_repairs = 0
        self.occupied_collision_repairs = 0
        self.masked_region_repairs = 0
        self.intended_cores.clear()
        self._potential = 0.0
        self.env.reset()
        return self._occ_map()

    def _occ_map(self) -> np.ndarray:
        # PAPER FIX (Sec 3.2, 'Representation of Core Placements'): occupied
        # cores are encoded by the INDEX of their assigned logic core, not a
        # bare 0/1 flag -- otherwise the agent can never tell WHERE an
        # already-placed predecessor task ended up, which is exactly the
        # information needed to minimize communication cost to it.
        #
        # PERF FIX: replaced a Python `for t, c in enumerate(...)` loop over
        # num_tasks (called every environment step -- e.g. ~482 times/episode
        # for AlexNet at batch_z=3) with vectorized numpy assignment. Same
        # result, no interpreted-Python loop.
        m = np.full(self.state_rows * self.state_cols, -1.0, dtype=np.float32)
        allowed_state_indices = self._state_index_by_grid[self._allowed_grid]
        m[allowed_state_indices] = 0.0
        valid = self._placement >= 0
        if np.any(valid):
            state_indices = self._state_index_by_grid[self._placement[valid]]
            m[state_indices] = (np.nonzero(valid)[0] + 1) / self.num_tasks

        # PAPER FIX: expose BOTH directions of communication volume,
        # aggregated over the WHOLE upcoming batch of up to batch_z tasks
        # (not just a single "current task"), since one action now assigns
        # all of them at once. PERF FIX: vectorized via row/column slicing +
        # sum instead of a per-task Python accumulation loop.
        remaining = self.num_tasks - self._task_ptr
        n_batch = min(self.batch_z, remaining) if remaining > 0 else 0
        if n_batch > 0:
            batch_slice = slice(self._task_ptr, self._task_ptr + n_batch)
            task_comm = (self.env.task_graph[batch_slice, :].sum(axis=0) +
                         self.env.task_graph[:, batch_slice].sum(axis=1))
        else:
            task_comm = np.zeros(self.num_tasks, dtype=np.float32)

        max_vol = task_comm.max()
        if max_vol > 0:
            task_comm = task_comm / max_vol

        return np.concatenate([m, task_comm])

    def _grid_core_to_action_pair(self, core_id):
        """Encode a legal grid core at the centre of its continuous bin."""
        x, y = int(core_id) % self.total_cols, int(core_id) // self.total_cols
        width = self._region_max_x - self._region_min_x + 1
        height = self._region_max_y - self._region_min_y + 1
        ax = 2.0 * ((x - self._region_min_x + 0.5) / width) - 1.0
        ay = 2.0 * ((y - self._region_min_y + 0.5) / height) - 1.0
        return float(ax), float(ay)

    def topology_to_grid_placement(self, placement):
        """Convert chip-major physical IDs into the policy's row-major grid."""
        physical = np.asarray(placement, dtype=np.int64)
        topo = self.env.topo
        chip = physical // topo.cores_per_chip
        local = physical % topo.cores_per_chip
        x = ((chip % topo.num_chips_x) * topo.cols_per_chip +
             local % topo.cols_per_chip)
        y = ((chip // topo.num_chips_x) * topo.rows_per_chip +
             local // topo.cols_per_chip)
        return (y * self.total_cols + x).astype(np.int32)

    def legal_action_candidates(self, proto_action, top_k=8):
        """Generate legal batched actions near a continuous proto-action.

        Each returned candidate contains unique, currently free physical
        cores.  The action vector is the centre-coordinate encoding of those
        executed cores, so replay describes the transition that actually
        occurred rather than the pre-repair request.
        """
        proto_action = np.asarray(proto_action, dtype=np.float32)
        remaining = self.num_tasks - self._task_ptr
        n_this_step = min(self.batch_z, remaining)
        free = np.asarray([core for core in self._allowed_grid
                           if int(core) not in self._occupied], dtype=np.int64)
        if len(free) < n_this_step:
            raise RuntimeError("not enough legal cores remain for guided action")
        candidate_count = max(1, min(int(top_k), len(free)))
        candidates, core_batches, seen = [], [], set()
        width = self._region_max_x - self._region_min_x + 1
        height = self._region_max_y - self._region_min_y + 1
        for candidate_index in range(candidate_count):
            available = free.copy()
            selected = []
            for pair_index in range(n_this_step):
                ax, ay = proto_action[2 * pair_index:2 * pair_index + 2]
                target_x = self._region_min_x + ((float(ax) + 1.0) / 2.0) * width
                target_y = self._region_min_y + ((float(ay) + 1.0) / 2.0) * height
                target_x = min(max(target_x, self._region_min_x),
                               np.nextafter(self._region_max_x + 1.0, -np.inf))
                target_y = min(max(target_y, self._region_min_y),
                               np.nextafter(self._region_max_y + 1.0, -np.inf))
                cx, cy = available % self.total_cols, available // self.total_cols
                order = np.lexsort((available,
                                    np.abs(cx - target_x) + np.abs(cy - target_y)))
                rank = (0 if candidate_index == 0 else
                        (candidate_index + pair_index - 1) %
                        min(candidate_count, len(order)))
                selected_core = int(available[order[rank]])
                selected.append(selected_core)
                available = available[available != selected_core]
            key = tuple(selected)
            if key in seen:
                continue
            seen.add(key)
            encoded = []
            for core in selected:
                encoded.extend(self._grid_core_to_action_pair(core))
            while len(encoded) < 2 * self.batch_z:
                encoded.extend((0.0, 0.0))
            candidates.append(np.asarray(encoded, dtype=np.float32))
            core_batches.append(np.asarray(selected, dtype=np.int32))
        return np.stack(candidates), core_batches

    def step_legal(self, grid_core_ids):
        """Place an already validated batch without invoking collision repair."""
        remaining = self.num_tasks - self._task_ptr
        n_this_step = min(self.batch_z, remaining)
        core_ids = np.asarray(grid_core_ids, dtype=np.int64)
        if len(core_ids) != n_this_step or len(set(core_ids.tolist())) != n_this_step:
            raise ValueError("guided action must contain one unique core per pending task")
        for core_id in core_ids:
            core_id = int(core_id)
            if not self._allowed_grid_mask[core_id] or core_id in self._occupied:
                raise ValueError("guided action selected an unavailable core")
            self.intended_cores.add(core_id)
            self._placement[self._task_ptr] = core_id
            self._occupied.add(core_id)
            self._task_ptr += 1
        return self._reward_after_step()

    def legal_action_mask(self) -> np.ndarray:
        """Boolean mask over the compact list of allowed physical cores."""
        mask = np.ones(len(self._allowed_grid), dtype=bool)
        if self._occupied:
            occupied = np.fromiter(self._occupied, dtype=np.int64,
                                   count=len(self._occupied))
            indices = np.searchsorted(self._allowed_grid, occupied)
            mask[indices] = False
        return mask

    def step_masked(self, action_index: int):
        """Execute one categorical legal-core action with no repair path."""
        if self.batch_z != 1:
            raise ValueError("masked categorical placement requires batch_z=1")
        action_index = int(action_index)
        mask = self.legal_action_mask()
        if not 0 <= action_index < len(mask) or not mask[action_index]:
            raise ValueError("masked policy selected an unavailable core")
        return self.step_legal([int(self._allowed_grid[action_index])])

    def _place_one(self, target_x: float, target_y: float):
        """Place the task at self._task_ptr onto a core, given a single
        (target_x, target_y) intended position. PAPER IMPLEMENTATION (Sec
        3.2, p.11-12): floor the continuous target to get the intended
        integer grid position; place there directly if free; on an actual
        collision, search by MINIMUM MANHATTAN DISTANCE to the ORIGINAL
        intended position, ties broken by first-found (core-index order).

        PERF FIX: the collision-search branch used to be a Python `for c in
        range(total_cores)` loop (up to 4096 iterations, run on every
        collision -- which can be a large fraction of placements on a
        crowded grid). Replaced with vectorized numpy: build a boolean
        occupied mask, compute Manhattan distance to every free core at
        once, and take the argmin. `np.argmin` returns the FIRST occurrence
        of the minimum when there are ties, and free-core indices are kept
        in ascending order by construction, so this preserves the exact
        same first-found tie-break semantics as the original loop."""
        intended_x = int(math.floor(target_x))
        intended_y = int(math.floor(target_y))
        intended_x = min(max(intended_x, 0), self.total_cols - 1)
        intended_y = min(max(intended_y, 0), self.total_rows - 1)
        intended_core = intended_y * self.total_cols + intended_x
        self.intended_cores.add(intended_core)

        if self._allowed_grid_mask[intended_core] and intended_core not in self._occupied:
            core_id = intended_core
        else:
            self.collision_repairs += 1
            if not self._allowed_grid_mask[intended_core]:
                self.masked_region_repairs += 1
            else:
                self.occupied_collision_repairs += 1
            occupied_mask = ~self._allowed_grid_mask.copy()
            if self._occupied:
                occupied_mask[np.fromiter(self._occupied, dtype=np.int64, count=len(self._occupied))] = True
            free_cores = np.nonzero(~occupied_mask)[0]  # ascending order, preserves tie-break
            cx = free_cores % self.total_cols
            cy = free_cores // self.total_cols
            dist = np.abs(cx - intended_x) + np.abs(cy - intended_y)
            core_id = int(free_cores[np.argmin(dist)])

        self._placement[self._task_ptr] = core_id
        self._occupied.add(core_id)

    def step(self, action):
        """Place up to `batch_z` unplaced logic cores per call, from a
        batched action [x1,y1,x2,y2,...,xz,yz] (paper Sec 3.2). If fewer
        than batch_z tasks remain (num_tasks not evenly divisible by z),
        only the first `remaining` (x,y) pairs of the action are used --
        the actor's output is still fixed-size 2*batch_z, the extras are
        simply ignored on the final, partial batch."""
        remaining = self.num_tasks - self._task_ptr
        n_this_step = min(self.batch_z, remaining)

        for k in range(n_this_step):
            ax, ay = action[2 * k], action[2 * k + 1]
            target_x = self._region_min_x + ((ax + 1.0) / 2.0) * (
                self._region_max_x - self._region_min_x)
            target_y = self._region_min_y + ((ay + 1.0) / 2.0) * (
                self._region_max_y - self._region_min_y)
            self._place_one(target_x, target_y)
            self._task_ptr += 1

        return self._reward_after_step()

    def _reward_after_step(self):
        """Evaluate sparse or shaped reward after either placement path."""

        done = (self._task_ptr >= self.num_tasks)

        # PAPER IMPLEMENTATION: sparse reward, r_t = 0 for every non-terminal
        # step (Algorithm 1, line 18); only a completed placement gets a
        # reward, r_t = sqrt(B) - sqrt(L(P)) (Algorithm 1, line 11), where B
        # is the fixed random-search baseline and L(P) is the pipeline
        # bottleneck latency (env.evaluate(), Eq. 4).
        if not done and self.reward_mode == "sparse":
            return 0.0, done, "", 0.0

        self.env.place(self.get_placement())
        final_cost = self.env.evaluate()
        if self.reward_mode == "potential":
            # Potential-based shaping F(s,s') = gamma*Phi(s') - Phi(s),
            # with Phi(terminal)=0. Across a fixed episode the discounted
            # shaping terms telescope to zero, so the sparse objective is
            # preserved while the critic receives intermediate feedback.
            # Normalize the shaping potential to approximately unit scale.
            # The gamma*Phi(s')-Phi(s) terms still telescope to zero over a
            # fixed episode, while avoiding cycle-scaled intermediate rewards
            # two orders of magnitude larger than the terminal objective.
            next_potential = (
                0.0 if done else
                -math.sqrt(max(self.reward_scale * final_cost, 0.0)) /
                self._potential_normalizer)
            base_reward = 0.0
            if done:
                base_reward = (
                    math.sqrt(max(self.reward_scale * self.baseline_latency, 0.0)) -
                    math.sqrt(max(self.reward_scale * final_cost, 0.0))
                ) if self.baseline_latency is not None else \
                    -math.sqrt(max(self.reward_scale * final_cost, 0.0))
            step_reward = base_reward + self.shaping_gamma * next_potential - self._potential
            self._potential = next_potential
            return step_reward, done, self._render() if done else "", final_cost if done else 0.0

        grid = self._render()

        if self.baseline_latency is not None:
            step_reward = (
                math.sqrt(max(self.reward_scale * self.baseline_latency, 0.0)) -
                math.sqrt(max(self.reward_scale * final_cost, 0.0)))
        else:
            step_reward = -math.sqrt(max(self.reward_scale * final_cost, 0.0))

        return step_reward, done, grid, final_cost

    def _render(self) -> str:
        rows, cols = self.total_rows, self.total_cols
        g = [['.' for _ in range(cols)] for _ in range(rows)]
        for t, c in enumerate(self._placement):
            if 0 <= c < rows * cols:
                g[c // cols][c % cols] = str(t % 10)
        return '\n'.join(' '.join(r) for r in g)

    def get_placement(self):
        # The policy grid is row-major; topology IDs are chip-major.
        p = self._placement.copy()
        valid = p >= 0
        y, x = p[valid] // self.total_cols, p[valid] % self.total_cols
        topo = self.env.topo
        chips = (y // topo.rows_per_chip) * topo.num_chips_x + x // topo.cols_per_chip
        local = (y % topo.rows_per_chip) * topo.cols_per_chip + x % topo.cols_per_chip
        p[valid] = chips * topo.cores_per_chip + local
        return p


# ---------------------------------------------------------------------------
# Experimental masked categorical PPO
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class MaskedPPOActorCritic(nn.Module):
        """Small actor/value network over exact legal physical-core actions."""

        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.policy_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            nn.init.zeros_(self.policy_head.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

        def forward(self, states):
            features = self.encoder(states)
            return self.policy_head(features), self.value_head(features).squeeze(-1)

        def distribution_and_value(self, states, masks):
            logits, values = self(states)
            masks = masks.to(dtype=torch.bool)
            if not bool(torch.all(masks.any(dim=-1))):
                raise ValueError("every masked categorical row needs a legal action")
            masked_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
            return torch.distributions.Categorical(logits=masked_logits), values

        def action_and_value(self, states, masks, deterministic=False, generator=None):
            distribution, values = self.distribution_and_value(states, masks)
            if deterministic:
                actions = distribution.logits.argmax(dim=-1)
            else:
                actions = torch.multinomial(
                    distribution.probs, 1, generator=generator).squeeze(-1)
            return actions, distribution.log_prob(actions), distribution.entropy(), values


    class ProposalPPOActorCritic(nn.Module):
        """Permutation-equivariant scorer for a small set of legal SA moves.

        Every row receives global annealing state plus a set of candidate-move
        features.  The zero-initialized score head makes the initial proposal
        distribution exactly uniform, so a frozen instance is a clean control
        for the learned proposal policy.
        """

        def __init__(self, global_dim: int, candidate_dim: int,
                     hidden_dim: int = 128):
            super().__init__()
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, hidden_dim), nn.Tanh())
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_dim, hidden_dim), nn.Tanh())
            self.score_head = nn.Linear(hidden_dim, 1)
            self.value_head = nn.Linear(hidden_dim, 1)
            for module in (self.global_encoder[0], self.candidate_encoder[0]):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            nn.init.zeros_(self.score_head.weight)
            nn.init.zeros_(self.score_head.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

        def forward(self, global_states, candidate_features):
            global_embedding = self.global_encoder(global_states)
            candidate_embedding = self.candidate_encoder(candidate_features)
            joint = torch.tanh(candidate_embedding + global_embedding[:, None, :])
            logits = self.score_head(joint).squeeze(-1)
            pooled = candidate_embedding.mean(dim=1)
            values = self.value_head(
                torch.tanh(global_embedding + pooled)).squeeze(-1)
            return logits, values

        def distribution_and_value(self, global_states, candidate_features):
            logits, values = self(global_states, candidate_features)
            return torch.distributions.Categorical(logits=logits), values

        def action_and_value(self, global_states, candidate_features,
                             generator=None):
            distribution, values = self.distribution_and_value(
                global_states, candidate_features)
            actions = torch.multinomial(
                distribution.probs, 1, generator=generator).squeeze(-1)
            return (actions, distribution.log_prob(actions),
                    distribution.entropy(), values)


    class ProposalPPOActor(nn.Module):
        """Candidate scorer with no value-function parameters.

        This separation follows the actor/critic ownership used by the
        official RL-Based-SA PPO implementation.  It prevents critic gradients
        from changing the representation used to rank proposal candidates.
        """

        def __init__(self, global_dim: int, candidate_dim: int,
                     hidden_dim: int = 128):
            super().__init__()
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, hidden_dim), nn.Tanh())
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_dim, hidden_dim), nn.Tanh())
            self.score_head = nn.Linear(hidden_dim, 1)
            for module in (self.global_encoder[0], self.candidate_encoder[0]):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            nn.init.zeros_(self.score_head.weight)
            nn.init.zeros_(self.score_head.bias)

        def distribution(self, global_states, candidate_features):
            global_embedding = self.global_encoder(global_states)
            candidate_embedding = self.candidate_encoder(candidate_features)
            joint = torch.tanh(
                candidate_embedding + global_embedding[:, None, :])
            logits = self.score_head(joint).squeeze(-1)
            return torch.distributions.Categorical(logits=logits)

        def action(self, global_states, candidate_features, generator=None):
            distribution = self.distribution(global_states, candidate_features)
            actions = torch.multinomial(
                distribution.probs, 1, generator=generator).squeeze(-1)
            return (actions, distribution.log_prob(actions),
                    distribution.entropy())


    class ProposalPPOCritic(nn.Module):
        """Value network independent from the proposal-scoring actor."""

        def __init__(self, global_dim: int, candidate_dim: int,
                     hidden_dim: int = 128):
            super().__init__()
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, hidden_dim), nn.Tanh())
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_dim, hidden_dim), nn.Tanh())
            self.value_head = nn.Linear(hidden_dim, 1)
            for module in (self.global_encoder[0], self.candidate_encoder[0]):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.zeros_(self.value_head.bias)

        def forward(self, global_states, candidate_features):
            global_embedding = self.global_encoder(global_states)
            candidate_embedding = self.candidate_encoder(candidate_features)
            pooled = candidate_embedding.mean(dim=1)
            return self.value_head(
                torch.tanh(global_embedding + pooled)).squeeze(-1)


    class ProposalBanditRegressor(nn.Module):
        """Predict the signed immediate gain of each legal SA proposal.

        The zero-initialized head gives every candidate the same score before
        observations arrive.  This makes the frozen model an exact uniform
        proposal control and lets epsilon-greedy exploration collect labels
        without spending extra objective evaluations.
        """

        def __init__(self, global_dim: int, candidate_dim: int,
                     hidden_dim: int = 128, scoring: str = "direct",
                     residual_limit: float = 0.25):
            super().__init__()
            self.scoring = scoring
            self.residual_limit = residual_limit
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, hidden_dim), nn.Tanh())
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_dim, hidden_dim), nn.Tanh())
            self.score_head = nn.Linear(hidden_dim, 1)
            for module in (self.global_encoder[0], self.candidate_encoder[0]):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            nn.init.zeros_(self.score_head.weight)
            nn.init.zeros_(self.score_head.bias)

        def forward(self, global_states, candidate_features):
            global_embedding = self.global_encoder(global_states)
            candidate_embedding = self.candidate_encoder(candidate_features)
            joint = torch.tanh(
                candidate_embedding + global_embedding[:, None, :])
            scores = self.score_head(joint).squeeze(-1)
            if self.scoring == "proxy_residual":
                # The zero head starts exactly at the proxy ranking. Limit
                # each correction so an uncalibrated learner cannot replace it.
                return (candidate_features[..., 7] +
                        self.residual_limit * torch.tanh(scores))
            return scores


def run_masked_ppo(env: MultiChipEnvironment, n_episodes: int = 500,
                   baseline_trials: int = 1000, device: str = None,
                   learning_rate: float = 3e-4, rollout_episodes: int = 10,
                   update_epochs: int = 4, minibatch_size: int = 512,
                   gamma: float = 0.98, gae_lambda: float = 0.95,
                   clip_ratio: float = 0.2, entropy_coefficient: float = 0.01,
                   value_coefficient: float = 0.5, max_grad_norm: float = 0.5,
                   hidden_dim: int = 256, reward_mode: str = "potential",
                   reward_scale: float = 1.0, diagnostics_every: int = 30,
                   diagnostics_path: str = None, save_checkpoint: str = None,
                   load_checkpoint: str = None, checkpoint_every: int = 100,
                   disable_learning: bool = False, seed: int = None,
                   run_metadata: dict = None) -> float:
    """Train a legal-action PPO policy as an experimental DDPG alternative.

    The next logic core is fixed by the workload order.  The policy therefore
    selects one currently free physical core from ``env.allowed_cores``.  A
    pre-softmax boolean mask makes illegal actions structurally impossible;
    no collision-repair heuristic participates in the transition.
    """
    if not HAS_TORCH:
        raise RuntimeError("masked PPO requires PyTorch")
    if min(n_episodes, baseline_trials, rollout_episodes, update_epochs,
           minibatch_size, hidden_dim, diagnostics_every, checkpoint_every) <= 0:
        raise ValueError("masked PPO budgets and sizes must be positive")
    if (not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1 or
            not 0 < clip_ratio < 1 or learning_rate <= 0 or
            entropy_coefficient < 0 or value_coefficient < 0 or max_grad_norm <= 0):
        raise ValueError("invalid masked PPO hyperparameters")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_value = 0 if seed is None else int(seed)

    signature = hashlib.sha256()
    signature.update(np.ascontiguousarray(env.task_graph).tobytes())
    signature.update(np.ascontiguousarray(env.compute_latency).tobytes())
    signature.update(np.ascontiguousarray(env.allowed_cores).tobytes())
    signature.update(repr((env.topo, env.topo.on_chip_latency,
                           env.topo.off_chip_latency, learning_rate,
                           rollout_episodes, update_epochs, minibatch_size,
                           gamma, gae_lambda, clip_ratio, entropy_coefficient,
                           value_coefficient, max_grad_norm, hidden_dim,
                           reward_mode, reward_scale, disable_learning,
                           seed_value, "masked-ppo-v1")).encode())
    fingerprint = signature.hexdigest()
    checkpoint = None
    if load_checkpoint and os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Masked PPO checkpoint is from a different configuration")
        baseline_latency = checkpoint["baseline_latency"]
        baseline_placement = np.asarray(
            checkpoint["baseline_placement"], dtype=np.int32).copy()
        print(f"[PPO] Resuming from placement {checkpoint['episode']}; "
              f"baseline B={baseline_latency:.6g}")
    else:
        print(f"[PPO] Computing random-search baseline B ({baseline_trials} trials)...")
        baseline_latency = run_random(env, n_trials=baseline_trials)
        baseline_placement = env.placement.copy()
        print(f"[PPO] Baseline B = {baseline_latency:.6g}")

    mapper = MultiChipCoreMapper(
        env, baseline_latency=baseline_latency, batch_z=1,
        reward_mode=reward_mode, shaping_gamma=gamma,
        reward_scale=reward_scale, compact_state=True)
    state_dim = mapper.state_rows * mapper.state_cols + env.num_tasks
    action_dim = len(mapper._allowed_grid)
    model = MaskedPPOActorCritic(state_dim, action_dim, hidden_dim).to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    action_generator = torch.Generator(device=torch_device)
    action_generator.manual_seed(seed_value + 104729)
    update_generator = torch.Generator(device=torch_device)
    update_generator.manual_seed(seed_value + 130363)

    best_cost = baseline_latency
    best_placement = baseline_placement.copy()
    best_grid = None
    best_candidate_source = "random_baseline"
    start_episode = deterministic_evaluations = update_count = 0
    initial_deterministic_cost = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_cost = checkpoint["best_cost"]
        best_placement = np.asarray(checkpoint["best_placement"], dtype=np.int32).copy()
        best_grid = checkpoint.get("best_grid")
        best_candidate_source = checkpoint.get("best_candidate_source", "ppo_training")
        start_episode = int(checkpoint["episode"])
        if start_episode > n_episodes:
            raise ValueError("Masked PPO checkpoint is beyond the requested placement budget")
        deterministic_evaluations = int(
            checkpoint.get("deterministic_candidate_evaluations", 0))
        update_count = int(checkpoint.get("update_count", 0))
        initial_deterministic_cost = checkpoint.get("initial_deterministic_cost")
        action_generator.set_state(checkpoint["action_generator_state"])
        update_generator.set_state(checkpoint["update_generator_state"])

    diagnostics_stream = None
    if diagnostics_path:
        parent = os.path.dirname(diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        diagnostics_stream = open(
            diagnostics_path, "a" if checkpoint is not None else "w", encoding="utf-8")

    def save(ep):
        if not save_checkpoint:
            return
        parent = os.path.dirname(save_checkpoint)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({
            "fingerprint": fingerprint, "episode": ep,
            "baseline_latency": baseline_latency,
            "baseline_placement": baseline_placement,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_cost": best_cost, "best_placement": best_placement,
            "best_grid": best_grid, "best_candidate_source": best_candidate_source,
            "deterministic_candidate_evaluations": deterministic_evaluations,
            "update_count": update_count,
            "initial_deterministic_cost": initial_deterministic_cost,
            "action_generator_state": action_generator.get_state(),
            "update_generator_state": update_generator.get_state(),
        }, save_checkpoint)

    def collect_episode(deterministic=False, keep_trajectory=True):
        state = mapper.reset()
        trajectory = {key: [] for key in
                      ("states", "masks", "actions", "log_probs", "rewards",
                       "values", "dones")}
        done = False
        grid = ""
        final_cost = 0.0
        while not done:
            mask = mapper.legal_action_mask()
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=torch_device).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.bool, device=torch_device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = model.action_and_value(
                    state_tensor, mask_tensor, deterministic=deterministic,
                    generator=None if deterministic else action_generator)
            reward, done, grid, final_cost = mapper.step_masked(int(action.item()))
            if keep_trajectory:
                trajectory["states"].append(state.copy())
                trajectory["masks"].append(mask.copy())
                trajectory["actions"].append(int(action.item()))
                trajectory["log_probs"].append(float(log_prob.item()))
                trajectory["rewards"].append(float(reward))
                trajectory["values"].append(float(value.item()))
                trajectory["dones"].append(bool(done))
            state = mapper._occ_map()
        return trajectory, grid, final_cost, mapper.get_placement().copy()

    def update_policy(trajectories):
        nonlocal update_count
        if disable_learning or not trajectories:
            return None
        combined = {key: sum((trajectory[key] for trajectory in trajectories), [])
                    for key in trajectories[0]}
        rewards = np.asarray(combined["rewards"], dtype=np.float32)
        values = np.asarray(combined["values"], dtype=np.float32)
        dones = np.asarray(combined["dones"], dtype=bool)
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            nonterminal = 0.0 if dones[index] else 1.0
            next_value = (values[index + 1]
                          if index + 1 < len(values) and not dones[index] else 0.0)
            delta = rewards[index] + gamma * next_value * nonterminal - values[index]
            last_advantage = (delta + gamma * gae_lambda * nonterminal *
                              last_advantage)
            advantages[index] = last_advantage
        returns = advantages + values
        states = torch.as_tensor(np.asarray(combined["states"]),
                                 dtype=torch.float32, device=torch_device)
        masks = torch.as_tensor(np.asarray(combined["masks"]),
                                dtype=torch.bool, device=torch_device)
        actions = torch.as_tensor(combined["actions"], dtype=torch.long,
                                  device=torch_device)
        old_log_probs = torch.as_tensor(combined["log_probs"], dtype=torch.float32,
                                        device=torch_device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32,
                                            device=torch_device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32,
                                         device=torch_device)
        if len(advantages_tensor) > 1:
            advantages_tensor = ((advantages_tensor - advantages_tensor.mean()) /
                                 (advantages_tensor.std(unbiased=False) + 1e-8))
        metrics = []
        for _ in range(update_epochs):
            permutation = torch.randperm(
                len(states), generator=update_generator, device=torch_device)
            for start in range(0, len(states), minibatch_size):
                indices = permutation[start:start + minibatch_size]
                distribution, new_values = model.distribution_and_value(
                    states[indices], masks[indices])
                new_log_probs = distribution.log_prob(actions[indices])
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages_tensor[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * \
                    advantages_tensor[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(new_values, returns_tensor[indices])
                entropy = distribution.entropy().mean()
                loss = (policy_loss + value_coefficient * value_loss -
                        entropy_coefficient * entropy)
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                metrics.append({
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approximate_kl": float(approximate_kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                })
        update_count += 1
        return {key: float(np.mean([metric[key] for metric in metrics]))
                for key in metrics[0]}

    if initial_deterministic_cost is None:
        _, initial_grid, initial_deterministic_cost, initial_placement = \
            collect_episode(deterministic=True, keep_trajectory=False)
        deterministic_evaluations += 1
        if initial_deterministic_cost < best_cost:
            best_cost, best_grid = initial_deterministic_cost, initial_grid
            best_placement = initial_placement.copy()
            best_candidate_source = "untrained_masked_policy_deterministic"

    print(f"[PPO] Device={torch_device}; state={state_dim}; legal actions={action_dim}; "
          f"reward={reward_mode}; learning={'off' if disable_learning else 'on'}")
    start_time = time.time()
    pending = []
    latest_metrics = None
    final_cost = best_cost
    for episode in range(start_episode + 1, n_episodes + 1):
        trajectory, grid, final_cost, placement = collect_episode()
        if final_cost < best_cost:
            best_cost, best_grid = final_cost, grid
            best_placement = placement.copy()
            best_candidate_source = ("untrained_masked_policy_rollout"
                                     if disable_learning else "ppo_training")
        if not disable_learning:
            pending.append(trajectory)
            if len(pending) >= rollout_episodes or episode == n_episodes:
                latest_metrics = update_policy(pending)
                pending.clear()

        write_diagnostics = (episode % diagnostics_every == 0 or episode == n_episodes)
        if write_diagnostics:
            _, eval_grid, deterministic_cost, deterministic_placement = \
                collect_episode(deterministic=True, keep_trajectory=False)
            deterministic_evaluations += 1
            retained = False
            if deterministic_cost < best_cost:
                best_cost, best_grid = deterministic_cost, eval_grid
                best_placement = deterministic_placement.copy()
                best_candidate_source = ("untrained_masked_policy_deterministic"
                                         if disable_learning else
                                         "ppo_deterministic")
                retained = True
            env.place(placement)
            record = {
                "episode": episode, "current_cost": final_cost,
                "best_cost": best_cost, "deterministic_cost": deterministic_cost,
                "learning_enabled": not disable_learning,
                "collision_repairs": mapper.collision_repairs,
                "deterministic_collision_repairs": 0,
                "retained_deterministic_candidate": retained,
                "valid_action_fraction_final": float(
                    (action_dim - env.num_tasks + 1) / action_dim),
                "update_count": update_count,
            }
            for key in ("policy_loss", "value_loss", "entropy", "approximate_kl",
                        "clip_fraction", "gradient_norm"):
                record[key] = latest_metrics.get(key) if latest_metrics else None
            if diagnostics_stream is not None:
                diagnostics_stream.write(json.dumps(record, allow_nan=False) + "\n")
                diagnostics_stream.flush()
        if episode % 10 == 0 or episode == n_episodes:
            elapsed = time.time() - start_time
            seconds_per_episode = elapsed / max(1, episode - start_episode)
            eta = timedelta(seconds=int(seconds_per_episode * (n_episodes - episode)))
            print(f"# of PPO placements: {episode:7d} | Current Cost: "
                  f"{final_cost:.6g} | Best Cost: {best_cost:.6g} | "
                  f"{seconds_per_episode:.2f}s/ep | ETA: {eta}")
        if (save_checkpoint and episode % checkpoint_every == 0 and
                (disable_learning or not pending)):
            save(episode)
            print(f"[PPO] Checkpoint saved to {save_checkpoint} (placement {episode})")

    save(n_episodes)
    if diagnostics_stream is not None:
        diagnostics_stream.close()
    env.place(best_placement)
    if run_metadata is not None:
        partial_evaluations = (n_episodes * max(0, env.num_tasks - 1)
                               if reward_mode == "potential" else 0)
        run_metadata.update({
            "method": "masked_categorical_ppo",
            "baseline_cost": baseline_latency,
            "training_candidate_evaluations": n_episodes,
            "deterministic_candidate_evaluations": deterministic_evaluations,
            "total_candidate_evaluations": (baseline_trials + n_episodes +
                                            deterministic_evaluations),
            "partial_objective_evaluations": partial_evaluations,
            "best_candidate_source": best_candidate_source,
            "best_cost": best_cost, "learning_enabled": not disable_learning,
            "random_seed": seed_value, "state_dimension": state_dim,
            "action_dimension": action_dim, "reward_mode": reward_mode,
            "rollout_episodes": rollout_episodes, "update_epochs": update_epochs,
            "minibatch_size": minibatch_size, "learning_rate": learning_rate,
            "gamma": gamma, "gae_lambda": gae_lambda,
            "clip_ratio": clip_ratio, "entropy_coefficient": entropy_coefficient,
            "value_coefficient": value_coefficient, "max_grad_norm": max_grad_norm,
            "hidden_dimension": hidden_dim, "update_count": update_count,
            "collision_repairs": 0, "diagnostics_path": diagnostics_path,
            "initial_deterministic_cost": initial_deterministic_cost,
        })
    return best_cost


# ---------------------------------------------------------------------------
# DDPG RL Training Loop (Based on Paper Pseudocode)
# ---------------------------------------------------------------------------

def run_ddpg(env: MultiChipEnvironment, n_episodes: int = 500, batch_size: int = 64,
             baseline_trials: int = 1000, batch_z: int = 3, train_every: int = 5,
             device: str = None, save_checkpoint: str = None, load_checkpoint: str = None,
             checkpoint_every: int = 100, diagnostics_path: str = None,
             run_metadata: dict = None, agent_arch: str = "mlp",
             reward_mode: str = "sparse", diagnostics_every: int = 100,
             reward_scale: float = 1.0,
             exploration_decay_placements: int = None,
             retain_deterministic_candidates: bool = False,
             ddpg_gamma: float = .98, ddpg_disable_learning: bool = False) -> float:
    if not 0 < ddpg_gamma <= 1:
        raise ValueError("DDPG gamma must be in (0,1]")
    if not HAS_TORCH:
        raise RuntimeError("DDPG requires PyTorch; refusing a random-search fallback")
    if diagnostics_every <= 0:
        raise ValueError("diagnostics_every must be positive")
    if exploration_decay_placements is None:
        exploration_decay_placements = max(1, int(0.8 * n_episodes))
    if exploration_decay_placements <= 0:
        raise ValueError("exploration_decay_placements must be positive")

    signature = hashlib.sha256()
    signature.update(np.ascontiguousarray(env.task_graph).tobytes())
    signature.update(np.ascontiguousarray(env.compute_latency).tobytes())
    signature.update(np.ascontiguousarray(env.allowed_cores).tobytes())
    signature.update(repr((env.topo, env.topo.on_chip_latency, env.topo.off_chip_latency,
                           batch_z, env.timing_units, agent_arch, reward_mode,
                           reward_scale, exploration_decay_placements,
                           retain_deterministic_candidates,
                           "chip-major-v7-target-normalization")).encode())
    if ddpg_gamma != .98 or ddpg_disable_learning:
        signature.update(repr((ddpg_gamma, ddpg_disable_learning)).encode())
    fingerprint = signature.hexdigest()
    # Reject old or incompatible checkpoints, including changed objective units.
    # instead of starting fresh -- restores the trained networks, optimizer
    # state, best result so far, noise schedule position, and (importantly)
    # the random-search baseline B, so resuming doesn't need to recompute an
    # expensive baseline_trials pass or lose the point noise_scale decayed to.
    checkpoint = None
    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        print(f"[DDPG] Loading checkpoint from {load_checkpoint} ...")
        checkpoint = torch.load(load_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint is from a different workload/configuration or older code; start a new checkpoint")
        baseline_latency = checkpoint["baseline_latency"]
        print(f"[DDPG] Resuming from episode {checkpoint['episode']}, "
              f"Baseline B = {baseline_latency:.6g} (loaded, not recomputed)")
    else:
        if load_checkpoint is not None:
            print(f"[DDPG] --load_checkpoint {load_checkpoint} not found -- starting fresh "
                  f"(it will be created at this path once training saves a checkpoint).")
        # Paper Algorithm 1: B is the latency of the best placement found by
        # random search, computed once up front and held fixed as the reward
        # normalizer. run_random() leaves env.placement mutated -- harmless,
        # since every training episode starts with mapper.reset() -> env.reset().
        print(f"[DDPG] Computing random-search baseline B ({baseline_trials} trials)...")
        baseline_latency = run_random(env, n_trials=baseline_trials)
        baseline_placement = env.placement.copy()
        print(f"[DDPG] Baseline B = {baseline_latency:.6g}")

    baseline_search_evaluations = (checkpoint.get("baseline_search_evaluations") if checkpoint is not None else baseline_trials)
    invocation_gradient_updates = 0

    topo  = env.topo
    rows  = topo.rows_per_chip * topo.num_chips_y
    cols  = topo.cols_per_chip * topo.num_chips_x
    state_dim = (rows * cols) + env.num_tasks

    # PAPER (Sec 3.2, 'Representation of Assigning Placements'): one action
    # assigns a batch of z unplaced logic cores at once -> action_dim = 2*z.
    batch_z = max(1, min(batch_z, env.num_tasks))
    action_dim = 2 * batch_z
    print(f"[DDPG] Batched action: placing {batch_z} logic core(s) per step "
          f"(action_dim={action_dim})")

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device,
                      agent_arch=agent_arch, rows=rows, cols=cols, num_tasks=env.num_tasks, gamma=ddpg_gamma)
    print(f"[DDPG] Agent architecture: {agent_arch}")
    print(f"[DDPG] Using device: {agent.device}"
          + (f" ({torch.get_num_threads()} CPU threads)" if agent.device.type == "cpu" else ""))
    replay_buffer = ReplayBuffer()
    mapper = MultiChipCoreMapper(env, baseline_latency=baseline_latency, batch_z=batch_z,
                                 reward_mode=reward_mode, shaping_gamma=agent.gamma,
                                 reward_scale=reward_scale)
    diagnostics_stream = None
    if diagnostics_path is not None:
        parent = os.path.dirname(diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        diagnostics_stream = open(diagnostics_path, "w", encoding="utf-8")
        print(f"[DDPG] Writing per-episode diagnostics to {diagnostics_path}")

    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if agent.device.type == "cuda" and checkpoint.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
        best_cost = checkpoint["best_cost"]
        best_placement = checkpoint["best_placement"]
        best_grid = checkpoint.get("best_grid")
        noise_scale = checkpoint["noise_scale"]
        global_step = checkpoint["global_step"]
        deterministic_candidate_evaluations = checkpoint.get(
            "deterministic_candidate_evaluations", 0)
        best_candidate_source = checkpoint.get("best_candidate_source", "training")
        start_episode = checkpoint["episode"]
    else:
        best_cost = baseline_latency
        best_placement = baseline_placement.copy()
        best_grid = None
        noise_scale = 1.0
        global_step = 0
        deterministic_candidate_evaluations = 0
        best_candidate_source = "random_baseline"
        start_episode = 0

    # Compute exploration from the absolute placement index. An explicit
    # horizon stays identical across cumulative checkpoint stages; deriving
    # it from each stage's temporary target would exhaust exploration in the
    # first stage and leave every resumed stage at the minimum noise level.
    def noise_at(episode):
        return max(0.01, 0.01 ** (episode / exploration_decay_placements))

    def _save_checkpoint(ep):
        if save_checkpoint is None:
            return
        torch.save({
            "fingerprint": fingerprint,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if agent.device.type == "cuda" else None,
            "agent": agent.state_dict(),
            "best_cost": best_cost,
            "best_placement": best_placement,
            "best_grid": best_grid,
            "noise_scale": noise_scale,
            "global_step": global_step,
            "deterministic_candidate_evaluations": deterministic_candidate_evaluations,
            "best_candidate_source": best_candidate_source,
            "episode": ep,
            "baseline_latency": baseline_latency,
            "baseline_search_evaluations": baseline_search_evaluations,
        }, save_checkpoint)

    # PERF FIX: agent.train() previously ran on EVERY environment step
    # (num_tasks/batch_z steps per episode -- e.g. 302 for a 906-task
    # workload at batch_z=3), each a full actor+critic forward/backward
    # pass. train_every spaces these out; replay_buffer.add() still runs
    # every step so no experience is lost, just the gradient-update
    # frequency is reduced.
    start_time = time.time()

    if start_episode >= n_episodes:
        print(f"[DDPG] Checkpoint episode ({start_episode}) already >= --epochs "
              f"({n_episodes}) -- nothing to do. Raise --epochs to continue training.")

    for ep in range(start_episode + 1, n_episodes + 1):
        state = mapper.reset()
        agent.ou_state.fill(0)
        done = False
        actions, losses = [], []
        episode_return = 0.0
        episode_discounted_return = 0.0
        episode_step = 0

        while not done:
            action = agent.select_action(state, noise_scale=noise_scale)
            actions.append(action.copy())
            reward, done, grid, final_cost = mapper.step(action)
            episode_return += reward
            episode_discounted_return += (agent.gamma ** episode_step) * reward
            episode_step += 1
            next_state = mapper._occ_map()

            replay_buffer.add(state, action, reward, next_state, done)
            global_step += 1
            if not ddpg_disable_learning and global_step % train_every == 0:
                loss = agent.train(replay_buffer, batch_size)
                if loss is not None:
                    losses.append(loss)
                    invocation_gradient_updates += 1
            state = next_state

        noise_scale = noise_at(ep)

        if final_cost < best_cost:
            best_cost = final_cost          
            best_grid   = grid
            best_placement = mapper.get_placement()
            best_candidate_source = "noisy_training"

        write_diagnostics = (diagnostics_stream is not None and
                             (ep % diagnostics_every == 0 or ep == n_episodes))
        if write_diagnostics:
            # Evaluate the actor without OU noise. This is deliberately a
            # separate rollout: it answers whether the trained policy itself
            # is useful, rather than whether a lucky noisy action was useful.
            evaluation_mapper = MultiChipCoreMapper(env, baseline_latency=baseline_latency,
                                                     batch_z=batch_z, reward_mode=reward_mode,
                                                     shaping_gamma=agent.gamma,
                                                     reward_scale=reward_scale)
            evaluation_state = evaluation_mapper.reset()
            evaluation_done = False
            evaluation_grid = ""
            while not evaluation_done:
                deterministic_action = agent.select_action(evaluation_state, explore=False)
                _, evaluation_done, evaluation_grid, deterministic_cost = evaluation_mapper.step(
                    deterministic_action)
                evaluation_state = evaluation_mapper._occ_map()
            deterministic_candidate_evaluations += 1
            retained_deterministic_candidate = False
            if retain_deterministic_candidates and deterministic_cost < best_cost:
                best_cost = deterministic_cost
                best_grid = evaluation_grid
                best_placement = evaluation_mapper.get_placement()
                best_candidate_source = "deterministic_diagnostic"
                retained_deterministic_candidate = True
            # Preserve the noisy episode placement for checkpointing/final output.
            env.place(mapper.get_placement())
            action_values = np.concatenate(actions) if actions else np.array([], dtype=np.float32)
            record = {
                "episode": ep,
                "current_cost": final_cost,
                "best_cost": best_cost,
                "deterministic_cost": deterministic_cost,
                "retained_deterministic_candidate": retained_deterministic_candidate,
                "reward": reward,
                "episode_return": episode_return,
                "episode_discounted_return": episode_discounted_return,
                "noise_scale": noise_scale,
                "collision_repairs": mapper.collision_repairs,
                "occupied_collision_repairs": mapper.occupied_collision_repairs,
                "masked_region_repairs": mapper.masked_region_repairs,
                "unique_intended_cores": len(mapper.intended_cores),
                "deterministic_collision_repairs": evaluation_mapper.collision_repairs,
                "deterministic_occupied_collision_repairs": evaluation_mapper.occupied_collision_repairs,
                "deterministic_masked_region_repairs": evaluation_mapper.masked_region_repairs,
                "deterministic_unique_intended_cores": len(evaluation_mapper.intended_cores),
                "action_mean": float(action_values.mean()) if len(action_values) else None,
                "action_std": float(action_values.std()) if len(action_values) else None,
                "action_saturated_fraction": float(np.mean(np.abs(action_values) >= 0.999)) if len(action_values) else None,
                "actor_loss_mean": float(np.mean([item["actor_loss"] for item in losses])) if losses else None,
                "critic_loss_mean": float(np.mean([item["critic_loss"] for item in losses])) if losses else None,
                "gradient_updates": len(losses),
            }
            diagnostics_stream.write(json.dumps(record, allow_nan=False) + "\n")
            diagnostics_stream.flush()

        if ep % 10 == 0 or ep == n_episodes:
            elapsed = time.time() - start_time
            per_ep = elapsed / max(1, ep - start_episode)
            eta_sec = per_ep * (n_episodes - ep)
            eta_str = str(timedelta(seconds=int(eta_sec)))
            print(f"# of placements: {ep:7d} | Current Cost: {final_cost:.6g} | "
                  f"Best Cost: {best_cost:.6g} | {per_ep:.2f}s/ep | ETA: {eta_str}")

        if save_checkpoint is not None and ep % checkpoint_every == 0:
            _save_checkpoint(ep)
            print(f"[DDPG] Checkpoint saved to {save_checkpoint} (episode {ep})")

    if save_checkpoint is not None and n_episodes > start_episode:
        _save_checkpoint(n_episodes)
        print(f"[DDPG] Final checkpoint saved to {save_checkpoint}")

    if diagnostics_stream is not None:
        diagnostics_stream.close()
    if run_metadata is not None:
        run_metadata.update({"gamma": ddpg_gamma, "learning_enabled": not ddpg_disable_learning,
                             "placement_steps": math.ceil(env.num_tasks / batch_z),
                             "first_step_terminal_discount": ddpg_gamma ** (math.ceil(env.num_tasks / batch_z)-1),
                             "baseline_search_evaluations": baseline_search_evaluations,
                             "search_evaluations_including_baseline": (None if baseline_search_evaluations is None else baseline_search_evaluations + n_episodes + deterministic_candidate_evaluations),
                             "invocation_gradient_updates": invocation_gradient_updates,
                             "agent_arch": agent_arch, "reward_mode": reward_mode,
                             "reward_scale": reward_scale,
                             "exploration_decay_placements": exploration_decay_placements,
                             "retain_deterministic_candidates": retain_deterministic_candidates,
                             "baseline_cost": baseline_latency, "start_episode": start_episode,
                             "end_episode": n_episodes, "global_steps": global_step,
                             "training_candidate_evaluations": n_episodes,
                             "deterministic_candidate_evaluations": deterministic_candidate_evaluations,
                             "total_candidate_evaluations": n_episodes + deterministic_candidate_evaluations,
                             "best_candidate_source": best_candidate_source,
                             "final_noise_scale": noise_scale, "diagnostics_path": diagnostics_path})

    if best_grid:
        print("--- Current Best Layout ---")
        print(best_grid)
        print("---------------------------")

    if best_placement is not None:
        env.place(best_placement)
        
    return best_cost


# ---------------------------------------------------------------------------
# Simulated Annealing, Adaptive SA, and Random Baseline
# ---------------------------------------------------------------------------

class _FreeCorePool:
    """O(1) random selection and replacement of currently unused cores."""
    def __init__(self, allowed_cores, placement):
        occupied = set(int(core) for core in placement)
        self.items = [int(core) for core in allowed_cores if int(core) not in occupied]
        self.positions = {core: index for index, core in enumerate(self.items)}

    def __bool__(self):
        return bool(self.items)

    def choice(self, rng=None):
        rng = random if rng is None else rng
        return self.items[rng.randrange(len(self.items))]

    def accept_relocation(self, used_core, freed_core):
        index = self.positions.pop(int(used_core))
        self.items[index] = int(freed_core)
        self.positions[int(freed_core)] = index


def _placement_neighbor(placement, n_perturb, free_pool, rng=None,
                        anchor_tasks=None):
    """Return a valid swap/relocation neighbor and its changed task indices.

    When ``anchor_tasks`` is supplied, the first selected task is sampled from
    that set and is guaranteed to move. This supports a matched
    bottleneck-focused proposal neighborhood without changing legality or
    evaluating extra candidates.
    """
    rng = random if rng is None else rng
    n = len(placement)
    count = min(n_perturb, n)
    if anchor_tasks is None:
        indices = np.asarray(rng.sample(range(n), count), dtype=np.int64)
    else:
        anchors = [int(index) for index in anchor_tasks
                   if 0 <= int(index) < n]
        if not anchors:
            raise ValueError("anchor_tasks must contain a valid task index")
        anchor = anchors[rng.randrange(len(anchors))]
        remaining = [index for index in range(n) if index != anchor]
        indices = np.asarray(
            [anchor] + rng.sample(remaining, count - 1), dtype=np.int64)
    candidate = placement.copy()
    original = candidate[indices].copy()
    permuted = original.tolist()
    rng.shuffle(permuted)
    if len(permuted) >= 2 and np.array_equal(permuted, original):
        permuted[0], permuted[1] = permuted[1], permuted[0]
    if anchor_tasks is not None and len(permuted) >= 2 and \
            permuted[0] == int(original[0]):
        swap_index = next(
            index for index in range(1, len(permuted))
            if permuted[index] != int(original[0]))
        permuted[0], permuted[swap_index] = (
            permuted[swap_index], permuted[0])
    candidate[indices] = permuted

    relocation = None
    if free_pool and (n == 1 or rng.random() < 0.5):
        used_core = free_pool.choice(rng)
        freed_core = int(candidate[indices[0]])
        candidate[indices[0]] = used_core
        relocation = (used_core, freed_core)
    changed = indices[candidate[indices] != placement[indices]]
    return candidate, changed, relocation


class IncrementalPipelineEvaluator:
    """Exact affected-stage evaluation for a MultiChipEnvironment candidate.

    Moving task ``u`` can change its own stage and every predecessor stage
    whose outgoing edge ends at ``u``. Other stage values are unchanged, so
    recomputing only those stages preserves exactly the environment objective.
    """
    def __init__(self, env):
        required = ("pipeline_stages", "_succ_cache", "pipeline_model",
                    "compute_latency", "topo")
        if not all(hasattr(env, name) for name in required):
            raise TypeError("environment does not expose the pipeline objective")
        self.env = env
        self.stages = env.pipeline_stages()
        self.task_stage = np.empty(env.num_tasks, dtype=np.int64)
        for stage_index, stage in enumerate(self.stages):
            self.task_stage[stage] = stage_index
        self.predecessors = [[] for _ in range(env.num_tasks)]
        self.predecessor_edges = [[] for _ in range(env.num_tasks)]
        for source, successors in enumerate(env._succ_cache):
            for destination, volume in successors:
                self.predecessors[destination].append(source)
                self.predecessor_edges[destination].append((source, volume))
        self.stage_details = [
            self._stage_details(index) for index in range(len(self.stages))]
        self.stage_costs = np.asarray(
            [details[0] for details in self.stage_details], dtype=np.float64)
        self.pending = None
        self.pending_details = None

    def _stage_details(self, stage_index):
        env = self.env
        stage = self.stages[stage_index]
        if env.pipeline_model == "task_sum":
            maximum = 0.0
            for task in stage:
                source = int(env.placement[task])
                if source < 0:
                    continue
                communication = 0.0
                for successor, volume in env._succ_cache[task]:
                    destination = int(env.placement[successor])
                    if destination >= 0:
                        communication += volume * env.topo.comm_cost(source, destination)
                maximum = max(maximum, float(env.compute_latency[task]) + communication)
            return maximum, 0.0, 0.0, {}, {}

        compute = max((float(env.compute_latency[task]) for task in stage
                       if env.placement[task] >= 0), default=0.0)
        loads = {}
        source_times = {}
        for task in stage:
            source = int(env.placement[task])
            if source < 0:
                continue
            routed_time = 0.0
            for successor, volume in env._succ_cache[task]:
                destination = int(env.placement[successor])
                if destination < 0:
                    continue
                for kind, link in env.topo.xy_route(source, destination):
                    key = (kind, link)
                    loads[key] = loads.get(key, 0.0) + volume
                    routed_time += volume * (
                        env.topo.on_chip_latency if kind == "on"
                        else env.topo.off_chip_latency)
            source_times[task] = routed_time
        link_bottleneck = max(
            (volume * (env.topo.on_chip_latency if kind == "on"
                       else env.topo.off_chip_latency)
             for (kind, _), volume in loads.items()), default=0.0)
        source_bottleneck = max(source_times.values(), default=0.0)
        communication = max(link_bottleneck, source_bottleneck)
        return (compute + communication, link_bottleneck,
                source_bottleneck, loads, source_times)

    def _stage_cost(self, stage_index):
        return self._stage_details(stage_index)[0]

    def _affected_stages(self, changed_tasks):
        affected = set()
        for task in np.asarray(changed_tasks, dtype=np.int64).tolist():
            affected.add(int(self.task_stage[task]))
            affected.update(int(self.task_stage[pred])
                            for pred in self.predecessors[task])
        return affected

    def candidate_contention_features(self, candidate, changed_tasks):
        """Return cheap contention-aware deltas for an unevaluated candidate.

        Only routes incident to moved tasks can change.  The method applies
        those route deltas to cached per-stage link loads and source
        serialization times without calling the placement objective.  Positive
        deltas mean the candidate reduces the corresponding bottleneck.
        """
        env = self.env
        changed = set(np.asarray(changed_tasks, dtype=np.int64).tolist())
        affected = self._affected_stages(changed_tasks)
        global_cost = max(float(self.stage_costs.max(initial=0.0)),
                          np.finfo(float).eps)
        if not affected:
            return 0.0, 0.0, 0.0

        link_deltas = {stage: {} for stage in affected}
        source_deltas = {stage: {} for stage in affected}
        edges = set()
        for task in changed:
            edges.update((task, int(destination), float(volume))
                         for destination, volume in env._succ_cache[task])
            edges.update((int(source), task, float(volume))
                         for source, volume in self.predecessor_edges[task])

        for source, destination, volume in edges:
            stage = int(self.task_stage[source])
            if stage not in affected:
                continue
            old_source = int(env.placement[source])
            old_destination = int(env.placement[destination])
            new_source = int(candidate[source])
            new_destination = int(candidate[destination])
            old_time = new_time = 0.0
            for sign, source_core, destination_core in (
                    (-1.0, old_source, old_destination),
                    (1.0, new_source, new_destination)):
                for kind, link in env.topo.xy_route(
                        source_core, destination_core):
                    key = (kind, link)
                    delta = sign * volume
                    link_deltas[stage][key] = (
                        link_deltas[stage].get(key, 0.0) + delta)
                    latency = (env.topo.on_chip_latency if kind == "on"
                               else env.topo.off_chip_latency)
                    if sign < 0:
                        old_time += volume * latency
                    else:
                        new_time += volume * latency
            source_deltas[stage][source] = (
                source_deltas[stage].get(source, 0.0) +
                new_time - old_time)

        old_link_peak = new_link_peak = 0.0
        old_source_peak = new_source_peak = 0.0
        for stage in affected:
            _, link_peak, source_peak, loads, source_times = \
                self.stage_details[stage]
            old_link_peak = max(old_link_peak, link_peak)
            old_source_peak = max(old_source_peak, source_peak)

            adjusted_loads = dict(loads)
            for key, delta in link_deltas[stage].items():
                adjusted_loads[key] = adjusted_loads.get(key, 0.0) + delta
            stage_link_peak = max((
                max(0.0, volume) *
                (env.topo.on_chip_latency if kind == "on"
                 else env.topo.off_chip_latency)
                for (kind, _), volume in adjusted_loads.items()), default=0.0)
            new_link_peak = max(new_link_peak, stage_link_peak)

            adjusted_sources = dict(source_times)
            for source, delta in source_deltas[stage].items():
                adjusted_sources[source] = max(
                    0.0, adjusted_sources.get(source, 0.0) + delta)
            new_source_peak = max(
                new_source_peak, max(adjusted_sources.values(), default=0.0))

        stage_slack_ratio = max(
            float(self.stage_costs[stage]) for stage in affected) / global_cost
        return (
            float(np.clip((old_link_peak - new_link_peak) / global_cost,
                          -1.0, 1.0)),
            float(np.clip(stage_slack_ratio, 0.0, 1.0)),
            float(np.clip((old_source_peak - new_source_peak) / global_cost,
                          -1.0, 1.0)),
        )

    def candidate_cost(self, changed_tasks):
        affected = self._affected_stages(changed_tasks)
        pending_details = {
            stage: self._stage_details(stage) for stage in affected}
        pending = {
            stage: details[0] for stage, details in pending_details.items()}
        if not pending:
            self.pending = {}
            self.pending_details = {}
            return float(self.stage_costs.max(initial=0.0))
        candidate = self.stage_costs.copy()
        for stage, value in pending.items():
            candidate[stage] = value
        self.pending = pending
        self.pending_details = pending_details
        return float(candidate.max(initial=0.0))

    def accept(self):
        for stage, value in (self.pending or {}).items():
            self.stage_costs[stage] = value
            self.stage_details[stage] = self.pending_details[stage]
        self.pending = None
        self.pending_details = None

    def reject(self):
        self.pending = None
        self.pending_details = None


def run_adaptive_sa(env: MultiChipEnvironment, n_iter: int = 100000,
                    initial_placement=None, initial_acceptance: float = 0.8,
                    target_acceptance: float = 0.30, adapt_window: int = 100,
                    min_perturb_frac: float = 0.005,
                    max_perturb_frac: float = 0.05,
                    stall_windows: int = 3, calibration_trials: int = 32,
                    diagnostics_path: str = None, metadata: dict = None,
                    incremental: bool = True) -> float:
    """Adaptive SA with measured temperature, feedback control, and reheating.

    Temperature is calibrated from observed uphill deltas. Every adaptation
    window then uses the acceptance ratio as feedback. Stagnation reheats the
    chain and expands its neighborhood; improvements contract the neighborhood.
    All calibration proposals count toward ``n_iter``.
    """
    if n_iter <= 0 or adapt_window <= 0 or stall_windows <= 0:
        raise ValueError("ASA budgets and window sizes must be positive")
    if not (0 < initial_acceptance < 1 and 0 < target_acceptance < 1):
        raise ValueError("ASA acceptance targets must be in (0, 1)")
    if not (0 < min_perturb_frac <= max_perturb_frac <= 1):
        raise ValueError("ASA perturbation fractions must satisfy 0 < min <= max <= 1")

    n = env.num_tasks
    if initial_placement is None:
        placement = np.asarray(random.sample(env.allowed_cores.tolist(), n),
                               dtype=np.int32)
        initialization = "random"
    else:
        placement = np.asarray(initial_placement, dtype=np.int32).copy()
        initialization = "supplied"
    env.place(placement)
    current_cost = float(env.evaluate())
    initial_cost = current_cost
    best_cost, best_placement = current_cost, placement.copy()
    free_pool = _FreeCorePool(env.allowed_cores, placement)

    evaluator = None
    if incremental:
        try:
            evaluator = IncrementalPipelineEvaluator(env)
        except (TypeError, AttributeError):
            evaluator = None

    diagnostics = None
    if diagnostics_path:
        parent = os.path.dirname(diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        diagnostics = open(diagnostics_path, "w", encoding="utf-8")

    evaluations = accepted = improving = reheats = 0
    positive_deltas = []
    perturb_fraction = min_perturb_frac
    progress_interval = max(1, n_iter // 50)
    next_progress = progress_interval

    def report_progress():
        nonlocal next_progress
        if evaluations >= next_progress or evaluations == n_iter:
            print(f"[ASA] proposals={evaluations}/{n_iter} "
                  f"best={best_cost:.9g}", flush=True)
            while next_progress <= evaluations:
                next_progress += progress_interval

    def evaluate(candidate, changed):
        env.place(candidate)
        return (evaluator.candidate_cost(changed) if evaluator is not None
                else float(env.evaluate()))

    def finish_proposal(candidate, new_cost, relocation, accept):
        nonlocal placement, current_cost, best_cost, best_placement, accepted, improving
        if accept:
            if new_cost < current_cost:
                improving += 1
            accepted += 1
            placement, current_cost = candidate, new_cost
            if evaluator is not None:
                evaluator.accept()
            if relocation is not None:
                free_pool.accept_relocation(*relocation)
            if new_cost < best_cost:
                best_cost, best_placement = new_cost, candidate.copy()
        else:
            env.place(placement)
            if evaluator is not None:
                evaluator.reject()

    # Calibrate in objective units. Greedy acceptance during this brief phase
    # avoids assuming a temperature before the scale of uphill moves is known.
    calibration = min(n_iter, max(1, calibration_trials))
    for _ in range(calibration):
        n_perturb = max(2 if n > 1 else 1,
                        min(n, round(perturb_fraction * n)))
        candidate, changed, relocation = _placement_neighbor(
            placement, n_perturb, free_pool)
        new_cost = evaluate(candidate, changed)
        evaluations += 1
        delta = new_cost - current_cost
        if delta > 0:
            positive_deltas.append(delta)
        finish_proposal(candidate, new_cost, relocation, delta <= 0)
        report_progress()

    scale = (float(np.median(positive_deltas)) if positive_deltas
             else max(abs(current_cost) * 1e-3, np.finfo(float).eps))
    initial_temperature = max(-scale / math.log(initial_acceptance),
                              np.finfo(float).eps)
    temperature = initial_temperature
    min_temperature = initial_temperature * 1e-6
    max_temperature = initial_temperature * 10.0
    stagnant = 0
    window_accepted = window_improving = window_evaluations = 0

    while evaluations < n_iter:
        n_perturb = max(2 if n > 1 else 1,
                        min(n, round(perturb_fraction * n)))
        candidate, changed, relocation = _placement_neighbor(
            placement, n_perturb, free_pool)
        new_cost = evaluate(candidate, changed)
        evaluations += 1
        window_evaluations += 1
        delta = new_cost - current_cost
        accept = delta <= 0 or random.random() < math.exp(
            -delta / max(temperature, np.finfo(float).eps))
        previous_accepted, previous_improving = accepted, improving
        finish_proposal(candidate, new_cost, relocation, accept)
        report_progress()
        window_accepted += accepted - previous_accepted
        window_improving += improving - previous_improving

        end_window = window_evaluations >= adapt_window or evaluations == n_iter
        if not end_window:
            continue
        acceptance_ratio = window_accepted / window_evaluations
        if acceptance_ratio < target_acceptance * 0.5:
            temperature *= 1.5
        elif acceptance_ratio > min(0.95, target_acceptance * 1.5):
            temperature *= 0.75
        else:
            temperature *= 0.95

        if window_improving == 0:
            stagnant += 1
        else:
            stagnant = 0
            perturb_fraction = max(min_perturb_frac, perturb_fraction * 0.9)
        if stagnant >= stall_windows:
            temperature = max(temperature, initial_temperature * 0.5)
            perturb_fraction = min(max_perturb_frac,
                                   max(min_perturb_frac, perturb_fraction * 1.5))
            reheats += 1
            stagnant = 0
        temperature = min(max_temperature, max(min_temperature, temperature))

        record = {"evaluations": evaluations, "temperature": temperature,
                  "acceptance_ratio": acceptance_ratio,
                  "perturb_fraction": perturb_fraction,
                  "current_cost": current_cost, "best_cost": best_cost,
                  "reheats": reheats}
        if diagnostics is not None:
            diagnostics.write(json.dumps(record, allow_nan=False) + "\n")
            diagnostics.flush()
        window_accepted = window_improving = window_evaluations = 0

    if diagnostics is not None:
        diagnostics.close()
    env.place(best_placement)
    if metadata is not None:
        metadata.update({
            "method": "adaptive_simulated_annealing",
            "initialization": initialization,
            "initial_cost": initial_cost,
            "best_cost": best_cost,
            "candidate_evaluations": evaluations,
            "initial_placement_evaluations": 1,
            "total_objective_evaluations": evaluations + 1,
            "accepted_moves": accepted,
            "improving_moves": improving,
            "reheats": reheats,
            "initial_temperature": initial_temperature,
            "final_temperature": temperature,
            "final_perturb_fraction": perturb_fraction,
            "incremental_evaluation": evaluator is not None,
            "diagnostics_path": diagnostics_path,
        })
    return best_cost


def run_ppo_asa(env: MultiChipEnvironment, n_iter: int = 100000,
                candidate_count: int = 16, rollout_steps: int = 256,
                device: str = None, learning_rate: float = 3e-4,
                weight_decay: float = 0.0,
                update_epochs: int = 4, minibatch_size: int = 256,
                gamma: float = 0.98, gae_lambda: float = 0.95,
                clip_ratio: float = 0.2, entropy_coefficient: float = 0.01,
                value_coefficient: float = 0.5, max_grad_norm: float = 0.5,
                hidden_dim: int = 128, disable_learning: bool = False,
                proposal_learner: str = "ppo",
                bandit_control: str = "uniform",
                bandit_replay_capacity: int = 50000,
                bandit_warmup: int = 512,
                bandit_recent_fraction: float = 0.0,
                bandit_recent_window: int = 10000,
                bandit_epsilon_final: float = 0.50,
                bandit_epsilon_decay_fraction: float = 1.00,
                bandit_exploration: str = "epsilon_all",
                bandit_top_m: int = 4,
                bandit_global_epsilon: float = 0.05,
                bandit_feature_set: str = "legacy9",
                bandit_shuffle_labels: bool = False,
                bandit_scoring: str = "direct",
                bandit_residual_limit: float = 0.25,
                state_encoding: str = "legacy_clipped",
                bandit_pool_supervision_stride: int = 0,
                bandit_ranking_weight: float = 0.0,
                update_style: str = "shared_gae",
                reward_mode: str = "realized",
                focus_bottleneck: bool = False,
                focus_fraction: float = 0.0,
                focus_stage_slack: float = 0.0,
                pool_focus_fraction: float = None,
                pool_small_move_fraction: float = 0.0,
                restart_interval: int = 0,
                update_at_chain_end: bool = False,
                seed: int = None, initial_acceptance: float = 0.8,
                target_acceptance: float = 0.30, adapt_window: int = 100,
                min_perturb_frac: float = 0.005,
                max_perturb_frac: float = 0.05,
                stall_windows: int = 3, calibration_trials: int = 32,
                annealing_schedule: str = "adaptive",
                fixed_cooldown: float = 0.99,
                fixed_perturb_fraction: float = 0.01,
                final_temperature_ratio: float = 1e-6,
                diagnostics_path: str = None, metadata: dict = None,
                incremental: bool = True, save_checkpoint: str = None,
                load_checkpoint: str = None,
                checkpoint_every: int = 1000,
                oracle_diagnostics_path: str = None,
                oracle_pools: int = 0, oracle_stride: int = 1,
                oracle_dataset_path: str = None,
                allow_finetune_loaded: bool = False) -> float:
    """Learned-proposal SA with either adaptive or paper-style fixed annealing.

    The PPO option follows the verified structure of Qiu and Liang's
    RL-Based-SA.  The contextual-bandit option instead regresses immediate
    signed move gain and explores with an epsilon-greedy policy.
    In both cases, simulated annealing owns Metropolis acceptance and cooling,
    and only the selected legal swap/relocation candidate calls the true
    placement objective.  A disabled-learning run stays exactly uniform.
    No external implementation code is copied. Reference:
    https://github.com/nathanqiu07/RL-Based-SA-Public
    """
    if not HAS_TORCH:
        raise RuntimeError("PPO-guided ASA requires PyTorch")
    if min(n_iter, candidate_count, rollout_steps, update_epochs,
           minibatch_size, hidden_dim, adapt_window, stall_windows,
           calibration_trials, checkpoint_every) <= 0:
        raise ValueError("PPO-ASA budgets and sizes must be positive")
    if candidate_count < 2:
        raise ValueError("PPO-ASA requires at least two proposal candidates")
    if proposal_learner not in {"ppo", "bandit"}:
        raise ValueError("unknown proposal learner")
    if bandit_control not in {
            "uniform", "proxy_greedy", "bottleneck_random", "checkpoint"}:
        raise ValueError("unknown bandit control")
    if bandit_control != "uniform" and (
            proposal_learner != "bandit" or not disable_learning):
        raise ValueError(
            "non-uniform bandit controls require the bandit learner with "
            "learning disabled")
    if not 0 <= pool_small_move_fraction <= 1:
        raise ValueError("pool small-move fraction must be in [0, 1]")
    if pool_focus_fraction is not None and not 0 <= pool_focus_fraction <= 1:
        raise ValueError("pool focus fraction must be in [0, 1]")
    if bandit_control == "checkpoint" and not load_checkpoint:
        raise ValueError("checkpoint bandit control requires a loaded checkpoint")
    if not 0 <= bandit_recent_fraction <= 1 or bandit_recent_window < 1:
        raise ValueError("invalid recent replay settings")
    if bandit_replay_capacity <= 0 or bandit_warmup < 0 or (bandit_warmup == 0 and not disable_learning):
        raise ValueError("replay capacity must be positive; zero warmup requires disabled learning")
    if bandit_warmup > bandit_replay_capacity:
        raise ValueError("bandit warmup cannot exceed replay capacity")
    if not 0.0 <= bandit_epsilon_final <= 1.0:
        raise ValueError("bandit final epsilon must be in [0, 1]")
    if not 0.0 < bandit_epsilon_decay_fraction <= 1.0:
        raise ValueError("bandit epsilon decay fraction must be in (0, 1]")
    if bandit_exploration not in {"epsilon_all", "epsilon_top_m"}:
        raise ValueError("unknown bandit exploration rule")
    if not 1 <= bandit_top_m <= candidate_count:
        raise ValueError("bandit top-m must be in [1, candidate_count]")
    if not 0.0 <= bandit_global_epsilon <= 1.0:
        raise ValueError("bandit global epsilon must be in [0, 1]")
    if bandit_feature_set not in {"legacy9", "contention12", "topology16"}:
        raise ValueError("unknown bandit candidate feature set")
    if bandit_scoring not in {"direct", "proxy_residual"}:
        raise ValueError("unknown bandit scoring mode")
    if not 0 < bandit_residual_limit <= 1:
        raise ValueError("bandit residual limit must be in (0, 1]")
    if bandit_scoring != "direct" and proposal_learner != "bandit":
        raise ValueError("proxy residual scoring requires the bandit learner")
    if state_encoding not in {"legacy_clipped", "relative_cost"}:
        raise ValueError("unknown proposal state encoding")
    if not math.isfinite(bandit_ranking_weight) or bandit_ranking_weight < 0:
        raise ValueError("ranking weight must be finite and nonnegative")
    if bandit_ranking_weight and (not bandit_pool_supervision_stride or bandit_shuffle_labels):
        raise ValueError("ranking requires supervised pools and unshuffled labels")
    if bandit_pool_supervision_stride < 0:
        raise ValueError("bandit pool-supervision stride cannot be negative")
    if bandit_pool_supervision_stride and (
            proposal_learner != "bandit" or disable_learning):
        raise ValueError("pool supervision requires an enabled bandit learner")
    if bandit_pool_supervision_stride and reward_mode not in {
            "proposed_delta", "positive_delta", "planted_proxy"}:
        raise ValueError(
            "pool supervision requires a proposal-local reward mode")
    if bandit_pool_supervision_stride and oracle_pools:
        raise ValueError(
            "pool supervision and oracle diagnostics cannot share one run")
    if bandit_shuffle_labels and (
            proposal_learner != "bandit" or disable_learning):
        raise ValueError(
            "shuffled labels require an enabled contextual bandit")
    if not 0.0 <= focus_fraction <= 1.0:
        raise ValueError("PPO-ASA focus fraction must be in [0, 1]")
    if not 0.0 <= focus_stage_slack < 1.0:
        raise ValueError("PPO-ASA focus stage slack must be in [0, 1)")
    if update_style not in {"shared_gae", "reference_separate"}:
        raise ValueError("unknown PPO-ASA update style")
    if reward_mode not in {
            "realized", "proposed_delta", "positive_delta",
            "planted_proxy"}:
        raise ValueError("unknown PPO-ASA reward mode")
    if annealing_schedule not in {"adaptive", "fixed"}:
        raise ValueError("unknown PPO-ASA annealing schedule")
    if oracle_pools < 0 or oracle_stride <= 0:
        raise ValueError("invalid PPO-ASA oracle diagnostic budget")
    if oracle_pools and not oracle_diagnostics_path:
        raise ValueError(
            "oracle diagnostics path is required when oracle pools are enabled")
    if oracle_dataset_path and not oracle_pools:
        raise ValueError(
            "oracle dataset export requires a positive oracle pool budget")
    if load_checkpoint and not disable_learning and not allow_finetune_loaded:
        raise ValueError(
            "learning from a loaded PPO-ASA checkpoint requires explicit "
            "allow_finetune_loaded=True")
    if restart_interval < 0:
        raise ValueError("PPO-ASA restart interval cannot be negative")
    if annealing_schedule == "fixed" and restart_interval:
        raise ValueError(
            "fixed PPO-SA currently requires one continuous annealing chain")
    if focus_bottleneck:
        focus_fraction = 1.0
    if (not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1 or
            not 0 < clip_ratio < 1 or learning_rate <= 0 or
            weight_decay < 0 or
            entropy_coefficient < 0 or value_coefficient < 0 or
            max_grad_norm <= 0):
        raise ValueError("invalid PPO-ASA hyperparameters")
    if not (0 < initial_acceptance < 1 and 0 < target_acceptance < 1):
        raise ValueError("ASA acceptance targets must be in (0, 1)")
    if not (0 < min_perturb_frac <= max_perturb_frac <= 1):
        raise ValueError("ASA perturbation fractions must satisfy 0 < min <= max <= 1")
    if not (0 < fixed_cooldown < 1):
        raise ValueError("fixed PPO-SA cooldown must be in (0, 1)")
    if not (0 < fixed_perturb_fraction <= 1):
        raise ValueError("fixed PPO-SA perturbation fraction must be in (0, 1]")
    if not (0 < final_temperature_ratio < 1):
        raise ValueError("fixed PPO-SA final temperature ratio must be in (0, 1)")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_value = 0 if seed is None else int(seed)
    proposal_rng = random.Random(seed_value)
    acceptance_rng = random.Random(seed_value + 1_000_003)
    torch.manual_seed(seed_value + 2_000_003)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed_value + 2_000_003)
    action_generator = torch.Generator(device=torch_device)
    action_generator.manual_seed(seed_value + 3_000_003)
    update_generator = torch.Generator(device=torch_device)
    update_generator.manual_seed(seed_value + 4_000_003)
    bandit_rng = random.Random(seed_value + 5_000_003)

    n = env.num_tasks
    placement = np.asarray(
        proposal_rng.sample(env.allowed_cores.tolist(), n), dtype=np.int32)
    env.place(placement)
    current_cost = float(env.evaluate())
    initial_cost = current_cost
    chain_initial_cost = current_cost
    chain_best_cost = current_cost
    best_cost, best_placement = current_cost, placement.copy()
    free_pool = _FreeCorePool(env.allowed_cores, placement)
    evaluator = None
    if incremental:
        try:
            evaluator = IncrementalPipelineEvaluator(env)
        except (TypeError, AttributeError):
            evaluator = None
    if (focus_fraction > 0 or bandit_feature_set in {
            "contention12", "topology16"}) and \
            evaluator is None:
        raise ValueError(
            "focused/contention-aware PPO-ASA requires the incremental "
            "pipeline evaluator")

    diagnostics = None
    if diagnostics_path:
        parent = os.path.dirname(diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        diagnostics = open(diagnostics_path, "w", encoding="utf-8")

    oracle_feature_names = [
        "changed_fraction", "is_relocation", "pressure_mean",
        "pressure_max", "motion", "chip_change",
        "bottleneck_fraction", "proxy_gain", "affected_volume",
    ]
    if bandit_feature_set in {"contention12", "topology16"}:
        oracle_feature_names.extend([
            "max_link_delta", "stage_slack_ratio",
            "max_source_byte_hop_delta",
        ])
    if bandit_feature_set == "topology16":
        oracle_feature_names.extend([
            "offchip_latency_share", "normalized_chip_count",
            "placement_load", "is_torus",
        ])
    oracle_feature_rows = []
    oracle_global_rows = []
    oracle_gain_rows = []
    oracle_score_rows = []
    oracle_probability_rows = []
    oracle_selected_rows = []
    oracle_chain_rows = []
    oracle_pool_rows = []
    oracle_candidate_evaluations = 0

    def rank_values(values):
        """Average ranks for a one-dimensional array, including ties."""
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while (stop < len(values) and
                   values[order[stop]] == values[order[start]]):
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    def safe_correlation(left, right, rank=False):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if rank:
            left, right = rank_values(left), rank_values(right)
        if (len(left) < 2 or np.std(left) <= np.finfo(float).eps or
                np.std(right) <= np.finfo(float).eps):
            return None
        return float(np.corrcoef(left, right)[0, 1])

    edge_sources, edge_destinations = np.nonzero(env.task_graph)
    edge_sources = edge_sources.astype(np.int64, copy=False)
    edge_destinations = edge_destinations.astype(np.int64, copy=False)
    edge_volumes = env.task_graph[edge_sources, edge_destinations].astype(
        np.float64, copy=False)
    incident_edges = [[] for _ in range(n)]
    for edge_index, (source, destination) in enumerate(
            zip(edge_sources.tolist(), edge_destinations.tolist())):
        incident_edges[source].append(edge_index)
        incident_edges[destination].append(edge_index)
    task_pressure = (env.task_graph.sum(axis=0, dtype=np.float64) +
                     env.task_graph.sum(axis=1, dtype=np.float64))
    task_pressure = np.log1p(task_pressure)
    pressure_scale = max(float(task_pressure.max(initial=0.0)),
                         np.finfo(float).eps)
    task_pressure /= pressure_scale
    total_edge_volume = max(float(edge_volumes.sum()), np.finfo(float).eps)
    latency_total = max(
        abs(float(env.topo.on_chip_latency)) +
        abs(float(env.topo.off_chip_latency)), np.finfo(float).eps)
    topology_features = [
        abs(float(env.topo.off_chip_latency)) / latency_total,
        min(1.0, math.log1p(env.topo.num_chips) / math.log1p(64.0)),
        min(1.0, n / max(1, len(env.allowed_cores))),
        float(env.topo.topology == "torus"),
    ]

    def bottleneck_tasks(use_focus):
        if not use_focus:
            return None
        maximum = float(evaluator.stage_costs.max(initial=0.0))
        threshold = maximum * (1.0 - focus_stage_slack)
        return [
            task
            for stage_index, stage in enumerate(evaluator.stages)
            if evaluator.stage_costs[stage_index] >= threshold
            for task in stage
        ]

    topo = env.topo
    core_ids = np.arange(env.total_cores, dtype=np.int64)
    core_chips = core_ids // topo.cores_per_chip
    local_ids = core_ids % topo.cores_per_chip
    core_rows = local_ids // topo.cols_per_chip
    core_cols = local_ids % topo.cols_per_chip
    chip_x = core_chips % topo.num_chips_x
    chip_y = core_chips // topo.num_chips_x
    motion_denominator = max(
        1, topo.rows_per_chip + topo.cols_per_chip +
        topo.num_chips_x + topo.num_chips_y - 4)

    def proxy_latencies(source_cores, destination_cores):
        row_delta = np.abs(core_rows[source_cores] - core_rows[destination_cores])
        col_delta = np.abs(core_cols[source_cores] - core_cols[destination_cores])
        chip_x_delta = np.abs(chip_x[source_cores] - chip_x[destination_cores])
        chip_y_delta = np.abs(chip_y[source_cores] - chip_y[destination_cores])
        if topo.topology == "torus":
            row_delta = np.minimum(row_delta, topo.rows_per_chip - row_delta)
            col_delta = np.minimum(col_delta, topo.cols_per_chip - col_delta)
            chip_x_delta = np.minimum(
                chip_x_delta, topo.num_chips_x - chip_x_delta)
            chip_y_delta = np.minimum(
                chip_y_delta, topo.num_chips_y - chip_y_delta)
        on_hops = row_delta + col_delta
        chip_hops = chip_x_delta + chip_y_delta
        return (on_hops * topo.on_chip_latency +
                chip_hops * topo.off_chip_latency)

    def candidate_features(candidate, changed, relocation, expected_changes,
                           bottleneck_stage):
        changed = np.asarray(changed, dtype=np.int64)
        if len(changed):
            old_cores = placement[changed]
            new_cores = candidate[changed]
            motion = (np.abs(core_rows[old_cores] - core_rows[new_cores]) +
                      np.abs(core_cols[old_cores] - core_cols[new_cores]) +
                      np.abs(chip_x[old_cores] - chip_x[new_cores]) +
                      np.abs(chip_y[old_cores] - chip_y[new_cores]))
            chip_change = np.mean(core_chips[old_cores] != core_chips[new_cores])
            pressure_mean = float(task_pressure[changed].mean())
            pressure_max = float(task_pressure[changed].max())
            edge_indices = np.unique(np.concatenate(
                [np.asarray(incident_edges[index], dtype=np.int64)
                 for index in changed if incident_edges[index]])) \
                if any(incident_edges[index] for index in changed) \
                else np.empty(0, dtype=np.int64)
        else:
            motion = np.zeros(1, dtype=np.float64)
            chip_change = pressure_mean = pressure_max = 0.0
            edge_indices = np.empty(0, dtype=np.int64)

        if len(edge_indices):
            sources = edge_sources[edge_indices]
            destinations = edge_destinations[edge_indices]
            volumes = edge_volumes[edge_indices]
            old_proxy = float(np.sum(
                volumes * proxy_latencies(placement[sources], placement[destinations])))
            new_proxy = float(np.sum(
                volumes * proxy_latencies(candidate[sources], candidate[destinations])))
            proxy_gain = np.clip(
                (old_proxy - new_proxy) / max(abs(old_proxy), np.finfo(float).eps),
                -1.0, 1.0)
            affected_volume = float(volumes.sum()) / total_edge_volume
        else:
            proxy_gain = affected_volume = 0.0
        if evaluator is not None and len(changed):
            bottleneck_fraction = float(np.mean(
                evaluator.task_stage[changed] == bottleneck_stage))
        else:
            bottleneck_fraction = 0.0
        feature_values = [
            len(changed) / max(1, expected_changes),
            float(relocation is not None),
            pressure_mean,
            pressure_max,
            float(np.mean(motion)) / motion_denominator,
            float(chip_change),
            bottleneck_fraction,
            float(proxy_gain),
            float(min(1.0, affected_volume)),
        ]
        if bandit_feature_set in {"contention12", "topology16"}:
            feature_values.extend(
                evaluator.candidate_contention_features(candidate, changed))
        if bandit_feature_set == "topology16":
            feature_values.extend(topology_features)
        return np.asarray(feature_values, dtype=np.float32)

    def evaluate(candidate, changed):
        env.place(candidate)
        return (evaluator.candidate_cost(changed) if evaluator is not None
                else float(env.evaluate()))

    def evaluate_oracle_pool(candidates, changed_sets):
        """Evaluate every candidate and restore the current placement.

        Pending incremental updates are retained so the selected candidate can
        reuse its exact cost without an extra objective evaluation.
        """
        costs, pending_updates = [], []
        for candidate, changed in zip(candidates, changed_sets):
            env.place(candidate)
            if evaluator is not None:
                cost = evaluator.candidate_cost(changed)
                pending_updates.append((
                    dict(evaluator.pending or {}),
                    dict(evaluator.pending_details or {}),
                ))
                evaluator.reject()
            else:
                cost = float(env.evaluate())
                pending_updates.append(None)
            costs.append(cost)
            env.place(placement)
        return np.asarray(costs, dtype=np.float64), pending_updates

    accepted = improving = reheats = neutral_proposals = evaluations = 0
    chain_count = initialization_evaluations = 1
    positive_deltas = []
    perturb_fraction = (fixed_perturb_fraction
                        if annealing_schedule == "fixed"
                        else min_perturb_frac)

    def finish_proposal(candidate, new_cost, relocation, accept):
        nonlocal placement, current_cost, best_cost, best_placement
        nonlocal accepted, improving, chain_best_cost
        if accept:
            if new_cost < current_cost:
                improving += 1
            accepted += 1
            placement, current_cost = candidate, new_cost
            if evaluator is not None:
                evaluator.accept()
            if relocation is not None:
                free_pool.accept_relocation(*relocation)
            chain_best_cost = min(chain_best_cost, new_cost)
            if new_cost < best_cost:
                best_cost, best_placement = new_cost, candidate.copy()
        else:
            env.place(placement)
            if evaluator is not None:
                evaluator.reject()

    calibration = min(n_iter, max(1, calibration_trials))
    if restart_interval and restart_interval <= calibration:
        raise ValueError(
            "PPO-ASA restart interval must exceed calibration trials")
    for _ in range(calibration):
        n_perturb = max(2 if n > 1 else 1,
                        min(n, round(perturb_fraction * n)))
        use_focus = (focus_fraction > 0 and
                     proposal_rng.random() < focus_fraction)
        candidate, changed, relocation = _placement_neighbor(
            placement, n_perturb, free_pool, rng=proposal_rng,
            anchor_tasks=bottleneck_tasks(use_focus))
        new_cost = evaluate(candidate, changed)
        evaluations += 1
        delta = new_cost - current_cost
        if delta > 0:
            positive_deltas.append(delta)
        elif delta == 0:
            neutral_proposals += 1
        finish_proposal(candidate, new_cost, relocation, delta <= 0)

    scale = (float(np.median(positive_deltas)) if positive_deltas
             else max(abs(current_cost) * 1e-3, np.finfo(float).eps))
    initial_temperature = max(-scale / math.log(initial_acceptance),
                              np.finfo(float).eps)
    temperature = initial_temperature
    min_temperature = initial_temperature * 1e-6
    max_temperature = initial_temperature * 10.0
    fixed_temperature_steps = max(
        1, math.ceil(
            math.log(final_temperature_ratio) / math.log(fixed_cooldown)))
    fixed_iterations_per_temperature = max(
        1, math.ceil(max(0, n_iter - calibration) /
                     fixed_temperature_steps))
    reward_scale = max(abs(initial_cost) * 1e-3, np.finfo(float).eps)

    global_dim = 8
    candidate_dim = len(oracle_feature_names)
    model = actor = critic = bandit = None
    if proposal_learner == "bandit":
        bandit = ProposalBanditRegressor(
            global_dim, candidate_dim, hidden_dim=hidden_dim,
            scoring=bandit_scoring,
            residual_limit=bandit_residual_limit).to(torch_device)
    elif update_style == "reference_separate":
        actor = ProposalPPOActor(
            global_dim, candidate_dim, hidden_dim=hidden_dim).to(torch_device)
        critic = ProposalPPOCritic(
            global_dim, candidate_dim, hidden_dim=hidden_dim).to(torch_device)
    else:
        model = ProposalPPOActorCritic(
            global_dim, candidate_dim, hidden_dim=hidden_dim).to(torch_device)
    loaded_checkpoint = None
    if load_checkpoint:
        if not os.path.exists(load_checkpoint):
            raise FileNotFoundError(
                f"PPO-ASA checkpoint not found: {load_checkpoint}")
        loaded_checkpoint = torch.load(
            load_checkpoint, map_location=torch_device, weights_only=False)
        expected_algorithm = (
            "bandit_sa_proposal_v1" if proposal_learner == "bandit" else
            "ppo_asa_reference_separate_v2"
            if update_style == "reference_separate" else
            "ppo_asa_proposal_v1")
        if loaded_checkpoint.get("algorithm") != expected_algorithm:
            raise ValueError("checkpoint is not a PPO-ASA proposal model")
        if loaded_checkpoint.get("candidate_count") != candidate_count:
            raise ValueError(
                "checkpoint candidate count does not match this run")
        if loaded_checkpoint.get("state_encoding", "legacy_clipped") != state_encoding:
            raise ValueError("checkpoint state encoding does not match this run")
        if proposal_learner == "bandit":
            if loaded_checkpoint.get("bandit_scoring", "direct") != bandit_scoring:
                raise ValueError("checkpoint bandit scoring does not match this run")
            if (bandit_scoring == "proxy_residual" and
                    loaded_checkpoint.get("bandit_residual_limit") != bandit_residual_limit):
                raise ValueError("checkpoint residual limit does not match this run")
            if loaded_checkpoint.get(
                    "bandit_pool_supervision_stride", 0) != \
                    bandit_pool_supervision_stride:
                raise ValueError(
                    "checkpoint pool-supervision stride does not match this run")
        if loaded_checkpoint.get("bandit_ranking_weight", 0.0) != bandit_ranking_weight:
            raise ValueError("checkpoint ranking weight does not match this run")
        if loaded_checkpoint.get("annealing_schedule", "adaptive") != \
                annealing_schedule:
            raise ValueError(
                "checkpoint annealing schedule does not match this run")
        if not math.isclose(
                float(loaded_checkpoint.get("focus_fraction", 0.0)),
                focus_fraction, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "checkpoint proposal focus does not match this run")
        if not math.isclose(
                float(loaded_checkpoint.get("focus_stage_slack", 0.0)),
                focus_stage_slack, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "checkpoint focus-stage slack does not match this run")
        if loaded_checkpoint.get("pool_small_move_fraction", 0.0) != pool_small_move_fraction:
            raise ValueError("checkpoint small-move fraction does not match this run")
        saved_pool_focus = loaded_checkpoint.get("pool_focus_fraction", loaded_checkpoint.get("focus_fraction", 0.0))
        requested_pool_focus = focus_fraction if pool_focus_fraction is None else pool_focus_fraction
        if saved_pool_focus != requested_pool_focus:
            raise ValueError("checkpoint pool focus fraction does not match this run")
        if loaded_checkpoint.get("bandit_feature_set", "legacy9") != \
                bandit_feature_set:
            raise ValueError(
                "checkpoint candidate feature set does not match this run")
        if proposal_learner == "bandit":
            bandit.load_state_dict(loaded_checkpoint["bandit"])
        elif update_style == "reference_separate":
            actor.load_state_dict(loaded_checkpoint["actor"])
            critic.load_state_dict(loaded_checkpoint["critic"])
        else:
            model.load_state_dict(loaded_checkpoint["model"])
    optimizer = actor_optimizer = critic_optimizer = bandit_optimizer = None
    if proposal_learner == "bandit":
        bandit_optimizer = optim.Adam(
            bandit.parameters(), lr=learning_rate, eps=1e-5,
            weight_decay=weight_decay)
        if (loaded_checkpoint is not None and allow_finetune_loaded and
                "bandit_optimizer" in loaded_checkpoint):
            bandit_optimizer.load_state_dict(
                loaded_checkpoint["bandit_optimizer"])
    elif update_style == "reference_separate":
        actor_optimizer = optim.Adam(
            actor.parameters(), lr=learning_rate, eps=1e-5,
            weight_decay=weight_decay)
        critic_optimizer = optim.Adam(
            critic.parameters(), lr=learning_rate, eps=1e-5,
            weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-5,
            weight_decay=weight_decay)
    update_count = 0

    def make_global_state(progress, last_delta, last_accepted,
                          recent_acceptance):
        temperature_log = math.log(max(temperature, np.finfo(float).eps) /
                                   initial_temperature)
        span = max(max_perturb_frac - min_perturb_frac, np.finfo(float).eps)
        cost_state = proposal_cost_state(
            chain_initial_cost, current_cost, chain_best_cost, state_encoding)
        return np.asarray([
            progress,
            np.clip(temperature_log / 14.0, -1.0, 1.0),
            *cost_state,
            np.clip(last_delta / reward_scale, -10.0, 10.0) / 10.0,
            float(last_accepted),
            float(recent_acceptance),
            np.clip((perturb_fraction - min_perturb_frac) / span, 0.0, 1.0),
        ], dtype=np.float32)

    def current_progress():
        if restart_interval:
            return min(1.0, chain_evaluations / restart_interval)
        return evaluations / n_iter

    effective_pool_focus = focus_fraction if pool_focus_fraction is None else pool_focus_fraction
    if not 0 <= effective_pool_focus <= 1:
        raise ValueError("pool focus fraction must be in [0, 1]")
    focused_candidate_count = round(candidate_count * effective_pool_focus)
    small_candidate_count = round(candidate_count * pool_small_move_fraction)
    selected_small_move_count = 0

    def make_proposal_pool(n_perturb):
        candidates, changed_sets, relocations, features = [], [], [], []
        focused_tasks = bottleneck_tasks(focused_candidate_count > 0)
        bottleneck_stage = (int(np.argmax(evaluator.stage_costs))
                            if evaluator is not None else None)
        for candidate_index in range(candidate_count):
            move_size = min(n, 2) if candidate_index < small_candidate_count else n_perturb
            candidate, changed, relocation = _placement_neighbor(
                placement, move_size, free_pool, rng=proposal_rng,
                anchor_tasks=(focused_tasks
                              if candidate_index < focused_candidate_count else None))
            candidates.append(candidate)
            changed_sets.append(changed)
            relocations.append(relocation)
            features.append(candidate_features(
                candidate, changed, relocation, n_perturb, bottleneck_stage))
        return candidates, changed_sets, relocations, np.stack(features)

    latest_metrics = None
    pending = []
    bandit_replay = []
    ranking_replay = []
    bandit_steps_since_update = 0
    bandit_observations = 0
    epsilon = 1.0

    def update_bandit():
        nonlocal update_count, bandit_steps_since_update
        if (disable_learning or len(bandit_replay) < bandit_warmup or
                bandit_steps_since_update < rollout_steps):
            return None
        metrics = []
        positive_indices = [
            index for index, row in enumerate(bandit_replay)
            if row[2] > 0.0]
        for _ in range(update_epochs):
            batch_size = min(minibatch_size, len(bandit_replay))
            # Default replay is uniform; optional recency sampling tracks drift.
            # Oversampling improving moves made the first bandit severely
            # overpredict gains and exploit harmful proposals.
            chosen = sample_bandit_replay_indices(bandit_rng, len(bandit_replay),
                batch_size, bandit_recent_fraction, bandit_recent_window)
            global_states = torch.as_tensor(
                np.asarray([bandit_replay[i][0] for i in chosen]),
                dtype=torch.float32, device=torch_device)
            candidate_rows = torch.as_tensor(
                np.asarray([bandit_replay[i][1] for i in chosen]),
                dtype=torch.float32, device=torch_device).unsqueeze(1)
            target_values = [bandit_replay[i][2] for i in chosen]
            if bandit_shuffle_labels:
                bandit_rng.shuffle(target_values)
            targets = torch.as_tensor(
                target_values, dtype=torch.float32, device=torch_device)
            if bandit_scoring == "proxy_residual":
                # Match the bounded proxy scale, preserving reward sign/order.
                # This is an explicit heuristic normalization, not a calibrated
                # claim that byte-hop reduction equals true latency reduction.
                targets = torch.tanh(targets)
            predictions = bandit(global_states, candidate_rows).squeeze(1)
            loss = F.smooth_l1_loss(predictions, targets)
            ranking_loss = predictions.sum() * 0.0
            if bandit_ranking_weight and ranking_replay:
                rows = bandit_rng.sample(ranking_replay, min(8, len(ranking_replay)))
                gs = torch.as_tensor(np.asarray([r[0] for r in rows]), dtype=torch.float32, device=torch_device)
                xs = torch.as_tensor(np.asarray([r[1] for r in rows]), dtype=torch.float32, device=torch_device)
                ys = torch.as_tensor(np.asarray([r[2] for r in rows]), dtype=torch.float32, device=torch_device)
                ranking_loss = within_pool_ranking_loss(bandit(gs, xs), ys)
                loss = loss + bandit_ranking_weight * ranking_loss
            bandit_optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                bandit.parameters(), max_grad_norm)
            bandit_optimizer.step()
            metrics.append({
                "bandit_loss": float(loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm),
                "score_head_weight_norm": float(
                    bandit.score_head.weight.detach().norm().cpu()),
                "mean_replay_age": float(np.mean([len(bandit_replay)-1-i for i in chosen])),
                "sampled_recent_fraction": float(np.mean([i >= len(bandit_replay)-bandit_recent_window for i in chosen])),
                "mean_prediction": float(predictions.detach().mean().cpu()),
                "mean_target": float(targets.detach().mean().cpu()),
                "positive_replay_fraction": (
                    len(positive_indices) / len(bandit_replay)),
            })
        bandit_steps_since_update = 0
        update_count += 1
        return {key: float(np.mean([entry[key] for entry in metrics]))
                for key in metrics[0]}

    def update_policy(transitions):
        nonlocal update_count
        if disable_learning or not transitions:
            return None
        global_states = torch.as_tensor(
            np.asarray([row["global_state"] for row in transitions]),
            dtype=torch.float32, device=torch_device)
        candidates = torch.as_tensor(
            np.asarray([row["candidate_features"] for row in transitions]),
            dtype=torch.float32, device=torch_device)
        actions = torch.as_tensor(
            [row["action"] for row in transitions], dtype=torch.long,
            device=torch_device)
        old_log_probs = torch.as_tensor(
            [row["log_prob"] for row in transitions], dtype=torch.float32,
            device=torch_device)
        rewards = np.asarray([row["reward"] for row in transitions],
                             dtype=np.float32)
        values = np.asarray([row["value"] for row in transitions],
                            dtype=np.float32)
        next_values = np.asarray([row["next_value"] for row in transitions],
                                 dtype=np.float32)
        dones = np.asarray([row["done"] for row in transitions], dtype=bool)
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            nonterminal = 0.0 if dones[index] else 1.0
            delta = (rewards[index] + gamma * next_values[index] * nonterminal -
                     values[index])
            last_advantage = (delta + gamma * gae_lambda * nonterminal *
                              last_advantage)
            advantages[index] = last_advantage
        if update_style == "reference_separate":
            returns = np.zeros_like(rewards)
            # The reference implementation updates on complete episodes.  Our
            # adapter can also update on fixed-size rollout fragments, so
            # bootstrap the final fragment unless it ended at a chain boundary.
            discounted_reward = (
                0.0 if dones[-1] else float(next_values[-1]))
            for index in range(len(rewards) - 1, -1, -1):
                nonterminal = 0.0 if dones[index] else 1.0
                discounted_reward = (
                    rewards[index] + gamma * nonterminal * discounted_reward)
                returns[index] = discounted_reward
        else:
            returns = advantages + values
        advantages_tensor = torch.as_tensor(
            advantages, dtype=torch.float32, device=torch_device)
        returns_tensor = torch.as_tensor(
            returns, dtype=torch.float32, device=torch_device)
        if len(advantages_tensor) > 1:
            advantages_tensor = ((advantages_tensor - advantages_tensor.mean()) /
                                 (advantages_tensor.std(unbiased=False) + 1e-8))

        metrics = []
        for _ in range(update_epochs):
            permutation = torch.randperm(
                len(transitions), generator=update_generator, device=torch_device)
            for start in range(0, len(transitions), minibatch_size):
                indices = permutation[start:start + minibatch_size]
                if update_style == "reference_separate":
                    distribution = actor.distribution(
                        global_states[indices], candidates[indices])
                    new_values = critic(
                        global_states[indices], candidates[indices])
                else:
                    distribution, new_values = model.distribution_and_value(
                        global_states[indices], candidates[indices])
                new_log_probs = distribution.log_prob(actions[indices])
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(
                    ratio * advantages_tensor[indices],
                    torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) *
                    advantages_tensor[indices]).mean()
                value_loss = F.mse_loss(new_values, returns_tensor[indices])
                entropy = distribution.entropy().mean()
                actor_objective = (
                    policy_loss - entropy_coefficient * entropy)
                critic_objective = value_coefficient * value_loss
                policy_module = (actor if update_style == "reference_separate"
                                 else model)
                value_module = (critic if update_style == "reference_separate"
                                else model)
                policy_head_gradients = torch.autograd.grad(
                    actor_objective, tuple(policy_module.score_head.parameters()),
                    retain_graph=True, allow_unused=True)
                value_head_gradients = torch.autograd.grad(
                    critic_objective, tuple(value_module.value_head.parameters()),
                    retain_graph=True, allow_unused=True)

                def diagnostic_gradient_norm(gradients):
                    terms = [gradient.detach().pow(2).sum()
                             for gradient in gradients
                             if gradient is not None]
                    if not terms:
                        return 0.0
                    return float(torch.sqrt(torch.stack(terms).sum()).cpu())

                policy_head_gradient_norm = diagnostic_gradient_norm(
                    policy_head_gradients)
                value_head_gradient_norm = diagnostic_gradient_norm(
                    value_head_gradients)
                if update_style == "reference_separate":
                    actor_optimizer.zero_grad()
                    actor_objective.backward()
                    actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), max_grad_norm)
                    actor_optimizer.step()
                    critic_optimizer.zero_grad()
                    critic_objective.backward()
                    critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), max_grad_norm)
                    critic_optimizer.step()
                    gradient_norm = max(
                        float(actor_gradient_norm),
                        float(critic_gradient_norm))
                else:
                    loss = actor_objective + critic_objective
                    optimizer.zero_grad()
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                    optimizer.step()
                    actor_gradient_norm = gradient_norm
                    critic_gradient_norm = gradient_norm
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
                metrics.append({
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approximate_kl": float(approximate_kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "gradient_norm": float(gradient_norm),
                    "actor_gradient_norm": float(actor_gradient_norm),
                    "critic_gradient_norm": float(critic_gradient_norm),
                    "policy_head_gradient_norm": policy_head_gradient_norm,
                    "value_head_gradient_norm": value_head_gradient_norm,
                    "score_head_weight_norm": float(
                        policy_module.score_head.weight.detach().norm().cpu()),
                })
        update_count += 1
        return {key: float(np.mean([entry[key] for entry in metrics]))
                for key in metrics[0]}

    stagnant = 0
    window_accepted = window_improving = window_evaluations = 0
    window_rewards = []
    window_proposed_rewards = []
    window_training_rewards = []
    window_selected_proxy = []
    window_proxy_advantages = []
    window_logit_stds = []
    window_logit_ranges = []
    window_feature_stds = []
    window_uniform_action_matches = []
    window_uniform_l1 = []
    window_selected_true_gains = []
    window_uniform_true_gains = []
    window_oracle_true_gains = []
    last_delta = 0.0
    last_accepted = False
    recent_acceptance = target_acceptance
    policy_budget = max(0, n_iter - calibration)
    proposal_steps = 0
    supervised_pools = 0
    supervised_candidate_evaluations = 0
    chain_evaluations = calibration
    progress_interval = max(adapt_window, max(1, n_iter // 50))
    next_progress = progress_interval

    if policy_budget:
        n_perturb = max(2 if n > 1 else 1,
                        min(n, round(perturb_fraction * n)))
        pool = make_proposal_pool(n_perturb)
    else:
        pool = None

    while evaluations < n_iter:
        candidates, changed_sets, relocations, features = pool
        global_state = make_global_state(
            current_progress(), last_delta, last_accepted,
            recent_acceptance)
        global_tensor = torch.as_tensor(
            global_state, dtype=torch.float32,
            device=torch_device).unsqueeze(0)
        feature_tensor = torch.as_tensor(
            features, dtype=torch.float32,
            device=torch_device).unsqueeze(0)
        with torch.no_grad():
            if proposal_learner == "bandit":
                if disable_learning and bandit_control == "proxy_greedy":
                    normalized_logits = feature_tensor[:, :, 7]
                elif disable_learning and bandit_control == "checkpoint":
                    normalized_logits = bandit(global_tensor, feature_tensor)
                elif disable_learning:
                    normalized_logits = torch.zeros_like(feature_tensor[:, :, 7])
                else:
                    normalized_logits = bandit(global_tensor, feature_tensor)
                if disable_learning and bandit_control == "uniform":
                    epsilon = 1.0
                    probabilities = torch.full_like(
                        normalized_logits, 1.0 / candidate_count)
                elif disable_learning and \
                        bandit_control == "bottleneck_random":
                    if focused_candidate_count <= 0:
                        raise ValueError(
                            "bottleneck_random requires a positive focus fraction")
                    epsilon = 1.0
                    probabilities = torch.zeros_like(normalized_logits)
                    probabilities[:, :focused_candidate_count] = (
                        1.0 / focused_candidate_count)
                else:
                    if bandit_observations < bandit_warmup:
                        epsilon = 1.0
                    else:
                        decay_steps = max(
                            1, round(policy_budget *
                                     bandit_epsilon_decay_fraction))
                        # Use total observations, not replay length: replay is
                        # capped and must not freeze the exploration schedule.
                        learned_steps = max(
                            0, bandit_observations - bandit_warmup)
                        decay_progress = min(
                            1.0, learned_steps / decay_steps)
                        epsilon = (1.0 - decay_progress *
                                   (1.0 - bandit_epsilon_final))
                    if bandit_observations < bandit_warmup or \
                            bandit_exploration == "epsilon_all":
                        probabilities = torch.full_like(
                            normalized_logits, epsilon / candidate_count)
                        greedy = normalized_logits.argmax(
                            dim=-1, keepdim=True)
                        probabilities.scatter_add_(
                            1, greedy,
                            torch.full_like(
                                greedy, 1.0 - epsilon,
                                dtype=probabilities.dtype))
                    else:
                        global_mass = bandit_global_epsilon
                        top_mass = epsilon * (1.0 - global_mass)
                        greedy_mass = ((1.0 - epsilon) *
                                       (1.0 - global_mass))
                        probabilities = torch.full_like(
                            normalized_logits,
                            global_mass / candidate_count)
                        # topk(1) and argmax may break equal-score ties differently.
                        # Share the greedy winner so top-m=1 remains one-hot.
                        top_indices = (normalized_logits.argmax(dim=-1, keepdim=True)
                            if bandit_top_m == 1 else normalized_logits.topk(
                                bandit_top_m, dim=-1).indices)
                        probabilities.scatter_add_(
                            1, top_indices,
                            torch.full_like(
                                top_indices, top_mass / bandit_top_m,
                                dtype=probabilities.dtype))
                        greedy = normalized_logits.argmax(
                            dim=-1, keepdim=True)
                        probabilities.scatter_add_(
                            1, greedy,
                            torch.full_like(
                                greedy, greedy_mass,
                                dtype=probabilities.dtype))
                value = torch.zeros(1, device=torch_device)
            elif update_style == "reference_separate":
                distribution = actor.distribution(
                    global_tensor, feature_tensor)
                value = critic(global_tensor, feature_tensor)
                probabilities = distribution.probs
                normalized_logits = distribution.logits
            else:
                distribution, value = model.distribution_and_value(
                    global_tensor, feature_tensor)
                probabilities = distribution.probs
                normalized_logits = distribution.logits
            sample = torch.rand(
                (probabilities.shape[0], 1), device=torch_device,
                generator=action_generator)
            action = torch.sum(
                sample > probabilities.cumsum(dim=-1), dim=-1).clamp(
                    max=candidate_count - 1)
            log_prob = (torch.log(probabilities.gather(
                1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
                        if proposal_learner == "bandit" else
                        distribution.log_prob(action))
            uniform_action = torch.floor(
                sample.squeeze(-1) * candidate_count).long().clamp(
                    max=candidate_count - 1)
        selected = int(action.item())
        selected_small_move_count += int(selected < small_candidate_count)
        candidate = candidates[selected]
        changed = changed_sets[selected]
        relocation = relocations[selected]
        before_cost = current_cost
        previous_accepted, previous_improving = accepted, improving
        policy_index = evaluations - calibration
        audit_pool = (
            len(oracle_pool_rows) < oracle_pools and
            policy_index % oracle_stride == 0)
        supervise_pool = bool(
            bandit_pool_supervision_stride and
            proposal_steps % bandit_pool_supervision_stride == 0 and
            evaluations + candidate_count <= n_iter)
        oracle_costs = oracle_pending = None
        if audit_pool or supervise_pool:
            oracle_costs, oracle_pending = evaluate_oracle_pool(
                candidates, changed_sets)
            true_gains = before_cost - oracle_costs
            if audit_pool:
                oracle_candidate_evaluations += candidate_count
                uniform_gain = float(np.mean(true_gains))
                oracle_gain = float(np.max(true_gains))
                selected_gain = float(true_gains[selected])
                oracle_feature_rows.append(np.asarray(features, dtype=np.float64))
                oracle_global_rows.append(np.asarray(global_state, dtype=np.float64))
                oracle_gain_rows.append(true_gains)
                oracle_score_rows.append(normalized_logits[0].detach().cpu().numpy().copy())
                oracle_probability_rows.append(probabilities[0].detach().cpu().numpy().copy())
                oracle_selected_rows.append(selected)
                oracle_chain_rows.append(chain_count)
                oracle_pool_rows.append({
                    "before_cost": before_cost,
                    "uniform_gain": uniform_gain,
                    "oracle_gain": oracle_gain,
                    "selected_gain": selected_gain,
                    "candidate_neutral_fraction": float(np.mean(true_gains == 0)),
                    "has_improving_candidate": bool(np.any(true_gains > 0)),
                    "proxy_argmax_matches_oracle": bool(
                        int(np.argmax(features[:, 7])) ==
                        int(np.argmax(true_gains))),
                })
                window_selected_true_gains.append(selected_gain)
                window_uniform_true_gains.append(uniform_gain)
                window_oracle_true_gains.append(oracle_gain)
            if supervise_pool:
                supervised_pools += 1
                supervised_candidate_evaluations += candidate_count
            env.place(candidate)
            if evaluator is not None:
                (evaluator.pending,
                 evaluator.pending_details) = oracle_pending[selected]
            new_cost = float(oracle_costs[selected])
        else:
            new_cost = evaluate(candidate, changed)
        charged_evaluations = candidate_count if supervise_pool else 1
        evaluations += charged_evaluations
        chain_evaluations += charged_evaluations
        proposal_steps += 1
        delta = new_cost - before_cost
        if delta == 0:
            neutral_proposals += 1
        accept = (delta <= 0 or acceptance_rng.random() <
                  math.exp(-delta / max(temperature, np.finfo(float).eps)))
        finish_proposal(candidate, new_cost, relocation, accept)
        proposed_reward = float(np.clip(
            (before_cost - new_cost) / reward_scale, -10.0, 10.0))
        realized_reward = float(np.clip(
            (before_cost - current_cost) / reward_scale, -10.0, 10.0))
        training_reward = proposal_training_reward(
            reward_mode, proposed_reward, realized_reward,
            features[selected, 7])
        if proposal_learner == "bandit":
            if supervise_pool:
                pool_rewards = np.clip(
                    true_gains / reward_scale, -10.0, 10.0)
                if reward_mode == "positive_delta":
                    pool_rewards = np.maximum(pool_rewards, 0.0)
                elif reward_mode == "planted_proxy":
                    pool_rewards = features[:, 7]
                if bandit_ranking_weight:
                    ranking_replay.append((global_state.copy(), features.copy(), pool_rewards.copy()))
                    del ranking_replay[:-256]
                for row, reward in zip(features, pool_rewards):
                    bandit_replay.append((
                        global_state.copy(), row.copy(), float(reward)))
                added_observations = candidate_count
            else:
                bandit_replay.append((
                    global_state.copy(), features[selected].copy(),
                    training_reward))
                added_observations = 1
            bandit_observations += added_observations
            if len(bandit_replay) > bandit_replay_capacity:
                del bandit_replay[:len(bandit_replay) -
                                  bandit_replay_capacity]
            bandit_steps_since_update += added_observations
            bandit_metrics = update_bandit()
            if bandit_metrics is not None:
                latest_metrics = bandit_metrics
        window_rewards.append(realized_reward)
        window_proposed_rewards.append(proposed_reward)
        window_training_rewards.append(training_reward)
        window_selected_proxy.append(float(features[selected, 7]))
        window_proxy_advantages.append(float(
            features[selected, 7] - np.mean(features[:, 7])))
        window_logit_stds.append(float(normalized_logits.std(
            dim=-1, unbiased=False).mean().cpu()))
        window_logit_ranges.append(float((
            normalized_logits.max(dim=-1).values -
            normalized_logits.min(dim=-1).values).mean().cpu()))
        window_feature_stds.append(np.std(features, axis=0))
        window_uniform_action_matches.append(
            float(selected == int(uniform_action.item())))
        window_uniform_l1.append(float(
            (0.5 * torch.abs(
                probabilities - 1.0 / candidate_count).sum(dim=-1).mean()).cpu()))
        window_evaluations += 1
        window_accepted += accepted - previous_accepted
        window_improving += improving - previous_improving
        last_delta = delta
        last_accepted = accept

        end_chain = bool(
            restart_interval and chain_evaluations >= restart_interval)
        end_window = (window_evaluations >= adapt_window or
                      evaluations == n_iter or end_chain)
        write_record = False
        if annealing_schedule == "fixed":
            completed_policy_steps = evaluations - calibration
            completed_temperature_steps = min(
                fixed_temperature_steps,
                completed_policy_steps // fixed_iterations_per_temperature)
            temperature = max(
                initial_temperature * final_temperature_ratio,
                initial_temperature *
                (fixed_cooldown ** completed_temperature_steps))

        if end_window:
            recent_acceptance = window_accepted / window_evaluations
            if annealing_schedule == "adaptive":
                if recent_acceptance < target_acceptance * 0.5:
                    temperature *= 1.5
                elif recent_acceptance > min(0.95, target_acceptance * 1.5):
                    temperature *= 0.75
                else:
                    temperature *= 0.95
                if window_improving == 0:
                    stagnant += 1
                else:
                    stagnant = 0
                    perturb_fraction = max(
                        min_perturb_frac, perturb_fraction * 0.9)
                if stagnant >= stall_windows:
                    temperature = max(
                        temperature, initial_temperature * 0.5)
                    perturb_fraction = min(
                        max_perturb_frac,
                        max(min_perturb_frac, perturb_fraction * 1.5))
                    reheats += 1
                    stagnant = 0
                temperature = min(
                    max_temperature, max(min_temperature, temperature))
            write_record = True

        overall_done = evaluations == n_iter
        done = overall_done or end_chain
        next_pool = None
        next_value = 0.0
        if not done:
            next_n_perturb = max(2 if n > 1 else 1,
                                 min(n, round(perturb_fraction * n)))
            next_pool = make_proposal_pool(next_n_perturb)
            next_global = make_global_state(
                current_progress(), last_delta, last_accepted,
                recent_acceptance)
            with torch.no_grad():
                next_global_tensor = torch.as_tensor(
                    next_global, dtype=torch.float32,
                    device=torch_device).unsqueeze(0)
                next_feature_tensor = torch.as_tensor(
                    next_pool[3], dtype=torch.float32,
                    device=torch_device).unsqueeze(0)
                if proposal_learner == "bandit":
                    next_value_tensor = torch.zeros(
                        1, device=torch_device)
                elif update_style == "reference_separate":
                    next_value_tensor = critic(
                        next_global_tensor, next_feature_tensor)
                else:
                    _, next_value_tensor = model.distribution_and_value(
                        next_global_tensor, next_feature_tensor)
            next_value = float(next_value_tensor.item())

        if proposal_learner == "ppo":
            pending.append({
                "global_state": global_state,
                "candidate_features": features,
                "action": selected,
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "reward": training_reward,
                "next_value": next_value,
                "done": done,
            })
            if (len(pending) >= rollout_steps or overall_done or
                    (end_chain and update_at_chain_end)):
                latest_metrics = update_policy(pending)
                pending.clear()

        if write_record:
            record = {
                "evaluations": evaluations,
                "temperature": temperature,
                "acceptance_ratio": recent_acceptance,
                "perturb_fraction": perturb_fraction,
                "current_cost": current_cost,
                "best_cost": best_cost,
                "reheats": reheats,
                "mean_realized_reward": float(np.mean(window_rewards)),
                "mean_proposed_reward": float(
                    np.mean(window_proposed_rewards)),
                "mean_training_reward": float(
                    np.mean(window_training_rewards)),
                "mean_selected_proxy_gain": float(
                    np.mean(window_selected_proxy)),
                "mean_proxy_advantage_over_uniform": float(
                    np.mean(window_proxy_advantages)),
                "mean_logit_std": float(np.mean(window_logit_stds)),
                "mean_logit_range": float(np.mean(window_logit_ranges)),
                "mean_feature_std": np.mean(
                    np.asarray(window_feature_stds), axis=0).tolist(),
                "uniform_action_match_fraction": float(
                    np.mean(window_uniform_action_matches)),
                "mean_probability_l1_from_uniform": float(
                    np.mean(window_uniform_l1)),
                "mean_selected_true_gain": (
                    float(np.mean(window_selected_true_gains))
                    if window_selected_true_gains else None),
                "mean_uniform_true_gain": (
                    float(np.mean(window_uniform_true_gains))
                    if window_uniform_true_gains else None),
                "mean_oracle_true_gain": (
                    float(np.mean(window_oracle_true_gains))
                    if window_oracle_true_gains else None),
                "learning_enabled": not disable_learning,
                "exploration_epsilon": (
                    epsilon if proposal_learner == "bandit" else None),
                "update_count": update_count,
                "chain": chain_count,
            }
            for key in ("policy_loss", "value_loss", "entropy",
                        "approximate_kl", "clip_fraction", "gradient_norm",
                        "actor_gradient_norm", "critic_gradient_norm",
                        "policy_head_gradient_norm",
                        "value_head_gradient_norm",
                        "score_head_weight_norm", "bandit_loss", "ranking_loss",
                        "mean_prediction", "mean_target", "mean_replay_age", "sampled_recent_fraction",
                        "positive_replay_fraction"):
                record[key] = (latest_metrics.get(key)
                               if latest_metrics is not None else None)
            if diagnostics is not None:
                diagnostics.write(json.dumps(record, allow_nan=False) + "\n")
                diagnostics.flush()
            window_accepted = window_improving = window_evaluations = 0
            window_rewards.clear()
            window_proposed_rewards.clear()
            window_training_rewards.clear()
            window_selected_proxy.clear()
            window_proxy_advantages.clear()
            window_logit_stds.clear()
            window_logit_ranges.clear()
            window_feature_stds.clear()
            window_uniform_action_matches.clear()
            window_uniform_l1.clear()
            window_selected_true_gains.clear()
            window_uniform_true_gains.clear()
            window_oracle_true_gains.clear()
        if end_chain and evaluations < n_iter:
            placement = np.asarray(
                proposal_rng.sample(env.allowed_cores.tolist(), n),
                dtype=np.int32)
            env.place(placement)
            current_cost = float(env.evaluate())
            chain_initial_cost = current_cost
            chain_best_cost = current_cost
            initialization_evaluations += 1
            chain_count += 1
            if current_cost < best_cost:
                best_cost, best_placement = current_cost, placement.copy()
            free_pool = _FreeCorePool(env.allowed_cores, placement)
            evaluator = None
            if incremental:
                try:
                    evaluator = IncrementalPipelineEvaluator(env)
                except (TypeError, AttributeError):
                    evaluator = None
            temperature = initial_temperature
            perturb_fraction = min_perturb_frac
            stagnant = 0
            last_delta = 0.0
            last_accepted = False
            recent_acceptance = target_acceptance
            chain_evaluations = 0
            next_n_perturb = max(
                2 if n > 1 else 1,
                min(n, round(perturb_fraction * n)))
            next_pool = make_proposal_pool(next_n_perturb)
        if evaluations >= next_progress or overall_done:
            print(
                f"[PPO-ASA] proposals={evaluations}/{n_iter} "
                f"chains={chain_count} updates={update_count} "
                f"best_us={best_cost * 1e6:.6f}",
                flush=True)
            while next_progress <= evaluations:
                next_progress += progress_interval
        pool = next_pool

    oracle_summary = None
    if oracle_pool_rows:
        features_flat = np.concatenate(oracle_feature_rows, axis=0)
        gains_flat = np.concatenate(oracle_gain_rows, axis=0)
        feature_means_by_pool = np.asarray(
            [row.mean(axis=0) for row in oracle_feature_rows])
        gain_means_by_pool = np.asarray(
            [row.mean() for row in oracle_gain_rows])
        centered_features = np.concatenate([
            row - mean for row, mean in
            zip(oracle_feature_rows, feature_means_by_pool)], axis=0)
        centered_gains = np.concatenate([
            row - mean for row, mean in
            zip(oracle_gain_rows, gain_means_by_pool)], axis=0)
        uniform_gains = np.asarray(
            [row["uniform_gain"] for row in oracle_pool_rows])
        oracle_gains = np.asarray(
            [row["oracle_gain"] for row in oracle_pool_rows])
        selected_gains = np.asarray(
            [row["selected_gain"] for row in oracle_pool_rows])

        def paired_interval(values):
            values = np.asarray(values, dtype=np.float64)
            mean = float(np.mean(values))
            half_width = (1.96 * float(np.std(values, ddof=1)) /
                          math.sqrt(len(values))) if len(values) > 1 else 0.0
            return [mean - half_width, mean + half_width]

        correlations = {}
        for index, name in enumerate(oracle_feature_names):
            correlations[name] = {
                "pooled_pearson": safe_correlation(
                    features_flat[:, index], gains_flat),
                "pooled_spearman": safe_correlation(
                    features_flat[:, index], gains_flat, rank=True),
                "within_pool_centered_pearson": safe_correlation(
                    centered_features[:, index], centered_gains),
            }
        oracle_advantage = oracle_gains - uniform_gains
        selected_advantage = selected_gains - uniform_gains
        mean_before = float(np.mean([
            row["before_cost"] for row in oracle_pool_rows]))
        oracle_summary = {
            "oracle_pools": len(oracle_pool_rows),
            "candidate_count": candidate_count,
            "oracle_candidate_evaluations": oracle_candidate_evaluations,
            "additional_objective_evaluations": (
                oracle_candidate_evaluations - len(oracle_pool_rows)),
            "mean_before_cost_seconds": mean_before,
            "candidate_neutral_fraction": float(np.mean([
                row["candidate_neutral_fraction"]
                for row in oracle_pool_rows])),
            "pools_with_improving_candidate_fraction": float(np.mean([
                row["has_improving_candidate"] for row in oracle_pool_rows])),
            "proxy_argmax_matches_oracle_fraction": float(np.mean([
                row["proxy_argmax_matches_oracle"]
                for row in oracle_pool_rows])),
            "mean_uniform_gain_seconds": float(np.mean(uniform_gains)),
            "mean_selected_gain_seconds": float(np.mean(selected_gains)),
            "mean_oracle_gain_seconds": float(np.mean(oracle_gains)),
            "mean_oracle_advantage_over_uniform_seconds": float(
                np.mean(oracle_advantage)),
            "mean_oracle_advantage_over_uniform_percent_of_cost": (
                float(np.mean(oracle_advantage)) / mean_before * 100),
            "oracle_advantage_95_percent_interval_seconds": paired_interval(
                oracle_advantage),
            "mean_selected_advantage_over_uniform_seconds": float(
                np.mean(selected_advantage)),
            "selected_advantage_95_percent_interval_seconds": paired_interval(
                selected_advantage),
            "feature_true_gain_correlations": correlations,
        }
        parent = os.path.dirname(oracle_diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(oracle_diagnostics_path, "w", encoding="utf-8") as stream:
            json.dump(oracle_summary, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if oracle_dataset_path:
            parent = os.path.dirname(oracle_dataset_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez_compressed(
                oracle_dataset_path,
                global_states=np.asarray(oracle_global_rows, dtype=np.float32),
                candidate_features=np.asarray(
                    oracle_feature_rows, dtype=np.float32),
                true_gains=np.asarray(oracle_gain_rows, dtype=np.float64),
                policy_scores=np.asarray(oracle_score_rows, dtype=np.float32),
                selection_probabilities=np.asarray(oracle_probability_rows, dtype=np.float32),
                selected_indices=np.asarray(oracle_selected_rows, dtype=np.int64),
                chain_ids=np.asarray(oracle_chain_rows, dtype=np.int64),
                feature_names=np.asarray(oracle_feature_names),
                candidate_count=np.asarray(candidate_count, dtype=np.int64),
                seed=np.asarray(seed_value, dtype=np.int64),
            )

    if diagnostics is not None:
        diagnostics.close()
    env.place(best_placement)
    if save_checkpoint:
        parent = os.path.dirname(save_checkpoint)
        if parent:
            os.makedirs(parent, exist_ok=True)
        checkpoint = {
            "algorithm": (
                "bandit_sa_proposal_v1" if proposal_learner == "bandit" else
                "ppo_asa_reference_separate_v2"
                if update_style == "reference_separate" else
                "ppo_asa_proposal_v1"),
            "resumable": False,
            "evaluations": evaluations,
            "best_cost": best_cost,
            "best_placement": best_placement,
            "seed": seed_value,
            "candidate_count": candidate_count,
            "learning_enabled": not disable_learning,
            "update_count": update_count,
            "hidden_dim": hidden_dim,
            "focus_fraction": focus_fraction,
            "focus_stage_slack": focus_stage_slack,
            "pool_focus_fraction": effective_pool_focus,
            "pool_small_move_fraction": pool_small_move_fraction,
            "restart_interval": restart_interval,
            "update_at_chain_end": update_at_chain_end,
            "independent_chains": chain_count,
            "update_style": update_style,
            "proposal_learner": proposal_learner,
            "bandit_control": bandit_control,
            "bandit_feature_set": bandit_feature_set,
            "bandit_exploration": bandit_exploration,
            "bandit_top_m": bandit_top_m,
            "bandit_global_epsilon": bandit_global_epsilon,
            "bandit_shuffle_labels": bandit_shuffle_labels,
            "bandit_recent_fraction": bandit_recent_fraction,
            "bandit_recent_window": bandit_recent_window,
            "bandit_scoring": bandit_scoring,
            "bandit_residual_limit": bandit_residual_limit,
            "state_encoding": state_encoding,
            "bandit_ranking_weight": bandit_ranking_weight,
            "bandit_pool_supervision_stride": (
                bandit_pool_supervision_stride),
            "reward_mode": reward_mode,
            "annealing_schedule": annealing_schedule,
        }
        if proposal_learner == "bandit":
            checkpoint.update({
                "bandit": bandit.state_dict(),
                "bandit_optimizer": bandit_optimizer.state_dict(),
            })
        elif update_style == "reference_separate":
            checkpoint.update({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            })
        else:
            checkpoint.update({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
        torch.save(checkpoint, save_checkpoint)
    if metadata is not None:
        metadata.update({
            "method": (
                f"{proposal_learner}_guided_fixed_simulated_annealing"
                if annealing_schedule == "fixed" else
                f"{proposal_learner}_guided_adaptive_simulated_annealing"),
            "proposal_policy": (
                "epsilon_greedy_proxy_gain"
                if (proposal_learner == "bandit" and
                    bandit_control == "proxy_greedy")
                else "frozen_checkpoint_contextual_bandit"
                if (proposal_learner == "bandit" and
                    bandit_control == "checkpoint")
                else "focused_candidate_uniform"
                if (proposal_learner == "bandit" and
                    bandit_control == "bottleneck_random")
                else "uniform_candidate_selection"
                if (proposal_learner == "bandit" and disable_learning)
                else "epsilon_greedy_contextual_bandit"
                if proposal_learner == "bandit" else "candidate_set_ppo"),
            "proposal_focus": proposal_focus_label(
                effective_pool_focus, focus_stage_slack),
            "pool_focus_fraction": effective_pool_focus,
            "pool_small_move_fraction": pool_small_move_fraction,
            "calibration_focus_fraction": focus_fraction,
            "small_move_candidates_per_pool": small_candidate_count,
            "selected_small_move_proposals": selected_small_move_count,
            "acceptance_rule": "fixed_metropolis",
            "temperature_controller": (
                "paper_style_fixed_schedule"
                if annealing_schedule == "fixed"
                else "existing_adaptive_sa"),
            "annealing_schedule": annealing_schedule,
            "fixed_cooldown": (fixed_cooldown
                               if annealing_schedule == "fixed" else None),
            "fixed_temperature_steps": (
                fixed_temperature_steps
                if annealing_schedule == "fixed" else None),
            "fixed_iterations_per_temperature": (
                fixed_iterations_per_temperature
                if annealing_schedule == "fixed" else None),
            "ending_temperature": (
                initial_temperature * final_temperature_ratio
                if annealing_schedule == "fixed" else min_temperature),
            "learning_enabled": not disable_learning,
            "model_initialization": (
                "loaded_checkpoint" if loaded_checkpoint is not None
                else "proxy_prior_zero_residual"
                if proposal_learner == "bandit" and bandit_scoring == "proxy_residual"
                else "zero_score_uniform"),
            "bandit_scoring": bandit_scoring,
            "bandit_residual_limit": bandit_residual_limit,
            "state_encoding": state_encoding,
            "bandit_target_transform": (
                "tanh_selected_reward" if bandit_scoring == "proxy_residual"
                else "selected_reward"),
            "bandit_ranking_weight": bandit_ranking_weight,
            "bandit_pool_supervision_stride": (
                bandit_pool_supervision_stride),
            "supervised_pools": supervised_pools,
            "supervised_candidate_evaluations": (
                supervised_candidate_evaluations),
            "loaded_checkpoint_path": load_checkpoint,
            "initialization": "seeded_random",
            "initial_cost": initial_cost,
            "best_cost": best_cost,
            "candidate_evaluations": evaluations,
            "initial_placement_evaluations": initialization_evaluations,
            "optimization_objective_evaluations": (
                evaluations + initialization_evaluations),
            "total_objective_evaluations": (
                evaluations + initialization_evaluations +
                oracle_candidate_evaluations - len(oracle_pool_rows)),
            "independent_chains": chain_count,
            "restart_interval": restart_interval,
            "progress_normalization": (
                "within_restart_chain" if restart_interval else "within_run"),
            "update_at_chain_end": update_at_chain_end,
            "weight_decay": weight_decay,
            "reward_mode": reward_mode,
            "proposal_learner": proposal_learner,
            "bandit_control": (
                bandit_control if proposal_learner == "bandit" else None),
            "bandit_recent_fraction": bandit_recent_fraction,
            "bandit_recent_window": bandit_recent_window,
            "bandit_replay_capacity": (
                bandit_replay_capacity if proposal_learner == "bandit" else None),
            "bandit_warmup": (
                bandit_warmup if proposal_learner == "bandit" else None),
            "bandit_epsilon_final": (
                bandit_epsilon_final if proposal_learner == "bandit" else None),
            "bandit_epsilon_decay_fraction": (
                bandit_epsilon_decay_fraction
                if proposal_learner == "bandit" else None),
            "bandit_exploration": (
                bandit_exploration if proposal_learner == "bandit" else None),
            "bandit_top_m": (
                bandit_top_m if proposal_learner == "bandit" else None),
            "bandit_global_epsilon": (
                bandit_global_epsilon
                if proposal_learner == "bandit" else None),
            "bandit_feature_set": (
                bandit_feature_set if proposal_learner == "bandit" else None),
            "bandit_shuffle_labels": (
                bandit_shuffle_labels
                if proposal_learner == "bandit" else None),
            "bandit_observations": (
                bandit_observations if proposal_learner == "bandit" else None),
            "final_exploration_epsilon": (
                epsilon if proposal_learner == "bandit" else None),
            "update_style": update_style,
            "policy_value_parameter_sharing": (
                update_style != "reference_separate"),
            "reference_implementation": (
                "https://github.com/nathanqiu07/RL-Based-SA-Public"
                if update_style == "reference_separate" else None),
            "reference_implementation_commit": (
                "6e64bec12b9a7c6db601a4a4ae254b942e4aed79"
                if update_style == "reference_separate" else None),
            "proposal_candidates_per_step": candidate_count,
            "candidate_feature_names": oracle_feature_names,
            "generated_proposal_candidates": (
                calibration + proposal_steps * candidate_count),
            "unevaluated_proposal_candidates": (
                proposal_steps * candidate_count -
                supervised_candidate_evaluations -
                (proposal_steps - supervised_pools)),
            "calibration_evaluations": calibration,
            "policy_steps": proposal_steps,
            "accepted_moves": accepted,
            "improving_moves": improving,
            "neutral_proposals": neutral_proposals,
            "reheats": reheats,
            "initial_temperature": initial_temperature,
            "final_temperature": temperature,
            "final_perturb_fraction": perturb_fraction,
            "incremental_evaluation": evaluator is not None,
            "update_count": update_count,
            "random_seed": seed_value,
            "state_dimension": global_dim,
            "candidate_feature_dimension": candidate_dim,
            "diagnostics_path": diagnostics_path,
            "checkpoint_path": save_checkpoint,
            "checkpoint_resumable": False,
            "oracle_diagnostics_path": oracle_diagnostics_path,
            "oracle_dataset_path": oracle_dataset_path,
            "oracle_pools": len(oracle_pool_rows),
            "oracle_candidate_evaluations": oracle_candidate_evaluations,
            "oracle_additional_objective_evaluations": (
                oracle_candidate_evaluations - len(oracle_pool_rows)),
            "oracle_summary": oracle_summary,
        })
        if loaded_checkpoint is not None:
            metadata.update({
                "source_training_evaluations": loaded_checkpoint.get(
                    "evaluations"),
                "source_training_updates": loaded_checkpoint.get(
                    "update_count"),
                "source_training_best_cost": loaded_checkpoint.get(
                    "best_cost"),
            })
        if latest_metrics is not None:
            metadata["final_update_metrics"] = latest_metrics
    return best_cost


def run_operator_bandit_sa(env: MultiChipEnvironment, n_iter: int = 100000,
                           proxy_candidates: int = 8,
                           ucb_alpha: float = 0.5,
                           discount: float = 0.995,
                           minimum_exploration: float = 0.10,
                           phase_bins: int = 3,
                           full_information_stride: int = 0,
                           selection_policy: str = "linucb",
                           proxy_gate_slate: int = 1,
                           thompson_scale: float = 0.25,
                           reward_mode: str = "signed",
                           reward_window: int = 50,
                           macro_horizon: int = 0,
                           control: str = "learned",
                           disable_learning: bool = False,
                           seed: int = None, initial_acceptance: float = 0.8,
                           calibration_trials: int = 32,
                           final_temperature_ratio: float = 1e-6,
                           metadata: dict = None) -> float:
    """Online contextual-bandit selection of fixed-SA move operators.

    The bandit chooses among proxy-ranked, bottleneck-focused, random-swap,
    and free-core relocation proposals.  Optional proxy gating cheaply screens
    a slate before exactly one real-objective call.  SA retains the ordinary
    Metropolis rule.  LinUCB and Cholesky Linear Thompson Sampling are
    available with signed-delta or outcome-aware rewards.
    """
    if n_iter <= 0 or proxy_candidates < 2 or ucb_alpha < 0:
        raise ValueError("invalid operator-bandit SA configuration")
    if not 0 < discount <= 1 or not 0 <= minimum_exploration <= 1:
        raise ValueError("invalid operator-bandit discount/exploration")
    if phase_bins <= 0:
        raise ValueError("operator-bandit phase count must be positive")
    if full_information_stride < 0:
        raise ValueError("full-information stride cannot be negative")
    if selection_policy not in {"linucb", "lints"}:
        raise ValueError("unknown operator-SA selection policy")
    if proxy_gate_slate <= 0 or proxy_gate_slate > 4:
        raise ValueError("proxy-gate slate must be in [1,4]")
    if thompson_scale < 0 or reward_window <= 0:
        raise ValueError("invalid Thompson scale/reward window")
    if reward_mode not in {"signed", "outcome"}:
        raise ValueError("unknown operator-SA reward mode")
    if macro_horizon < 0:
        raise ValueError("operator-SA macro horizon cannot be negative")
    if macro_horizon and full_information_stride:
        raise ValueError("macro actions and full-information labels are exclusive")
    if control not in {"learned", "uniform", "proxy"}:
        raise ValueError("unknown operator-SA control")
    if disable_learning:
        control = "uniform"
    if not 0 < initial_acceptance < 1 or not 0 < final_temperature_ratio < 1:
        raise ValueError("invalid fixed-SA temperature configuration")
    seed_value = 0 if seed is None else int(seed)
    proposal_rng = random.Random(seed_value)
    selection_rng = random.Random(seed_value + 5_000_003)
    thompson_rng = np.random.default_rng(seed_value + 7_000_003)
    acceptance_rng = random.Random(seed_value + 1_000_003)
    n = env.num_tasks
    placement = np.asarray(
        proposal_rng.sample(env.allowed_cores.tolist(), n), dtype=np.int32)
    env.place(placement)
    current_cost = float(env.evaluate())
    initial_cost = best_cost = current_cost
    best_placement = placement.copy()
    free_pool = _FreeCorePool(env.allowed_cores, placement)
    try:
        evaluator = IncrementalPipelineEvaluator(env)
    except (TypeError, AttributeError):
        evaluator = None

    sources, destinations = np.nonzero(env.task_graph)
    sources = sources.astype(np.int64, copy=False)
    destinations = destinations.astype(np.int64, copy=False)
    volumes = env.task_graph[sources, destinations].astype(np.float64, copy=False)
    incident = [[] for _ in range(n)]
    for edge, (source, destination) in enumerate(
            zip(sources.tolist(), destinations.tolist())):
        incident[source].append(edge)
        incident[destination].append(edge)
    topo = env.topo

    def proxy_distance(left, right):
        left, right = np.asarray(left), np.asarray(right)
        left_chip = left // topo.cores_per_chip
        right_chip = right // topo.cores_per_chip
        left_local = left % topo.cores_per_chip
        right_local = right % topo.cores_per_chip
        row_delta = np.abs(left_local // topo.cols_per_chip -
                           right_local // topo.cols_per_chip)
        col_delta = np.abs(left_local % topo.cols_per_chip -
                           right_local % topo.cols_per_chip)
        chip_x_delta = np.abs(left_chip % topo.num_chips_x -
                              right_chip % topo.num_chips_x)
        chip_y_delta = np.abs(left_chip // topo.num_chips_x -
                              right_chip // topo.num_chips_x)
        if topo.topology == "torus":
            row_delta = np.minimum(row_delta, topo.rows_per_chip - row_delta)
            col_delta = np.minimum(col_delta, topo.cols_per_chip - col_delta)
            chip_x_delta = np.minimum(
                chip_x_delta, topo.num_chips_x - chip_x_delta)
            chip_y_delta = np.minimum(
                chip_y_delta, topo.num_chips_y - chip_y_delta)
        return ((row_delta + col_delta) * topo.on_chip_latency +
                (chip_x_delta + chip_y_delta) * topo.off_chip_latency)

    def proxy_gain(candidate, changed):
        edge_ids = sorted({edge for task in np.asarray(changed, dtype=int)
                           for edge in incident[task]})
        if not edge_ids:
            return 0.0
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        src, dst, weight = (sources[edge_ids], destinations[edge_ids],
                            volumes[edge_ids])
        old = float(np.sum(weight * proxy_distance(
            placement[src], placement[dst])))
        new = float(np.sum(weight * proxy_distance(
            candidate[src], candidate[dst])))
        return (old - new) / max(abs(old), np.finfo(float).eps)

    def bottleneck_tasks():
        if evaluator is None:
            return None
        return evaluator.stages[int(np.argmax(evaluator.stage_costs))]

    perturb = max(2 if n > 1 else 1, min(n, round(0.01 * n)))

    def random_neighbor(anchor=None, allow_relocation=True):
        return _placement_neighbor(
            placement, perturb, free_pool if allow_relocation else None,
            rng=proposal_rng, anchor_tasks=anchor)

    def relocation_neighbor():
        if not free_pool:
            return random_neighbor()
        anchors = bottleneck_tasks()
        task = (int(anchors[proposal_rng.randrange(len(anchors))])
                if anchors and proposal_rng.random() < 0.5
                else proposal_rng.randrange(n))
        candidate = placement.copy()
        used_core = free_pool.choice(proposal_rng)
        freed_core = int(candidate[task])
        candidate[task] = used_core
        return (candidate, np.asarray([task], dtype=np.int64),
                (used_core, freed_core))

    def proposal(operator):
        if operator == "proxy":
            choices = []
            anchors = bottleneck_tasks()
            for index in range(proxy_candidates):
                item = random_neighbor(
                    anchors if anchors and index < proxy_candidates // 2 else None)
                choices.append((*item, proxy_gain(item[0], item[1])))
            candidate, changed, relocation, _ = max(
                choices, key=lambda item: item[3])
            return candidate, changed, relocation
        if operator == "bottleneck":
            anchors = bottleneck_tasks()
            return random_neighbor(anchors if anchors else None)
        if operator == "swap":
            return random_neighbor(allow_relocation=False)
        return relocation_neighbor()

    def evaluate(candidate, changed):
        env.place(candidate)
        return (evaluator.candidate_cost(changed) if evaluator is not None
                else float(env.evaluate()))

    accepted = improving = evaluations = 0
    positive_deltas = []
    calibration = min(n_iter, max(1, calibration_trials))

    def finish(candidate, changed, relocation, new_cost, accept):
        nonlocal placement, current_cost, best_cost, best_placement
        nonlocal accepted, improving
        if accept:
            improving += int(new_cost < current_cost)
            accepted += 1
            placement, current_cost = candidate, new_cost
            if evaluator is not None:
                evaluator.accept()
            if relocation is not None:
                free_pool.accept_relocation(*relocation)
            if new_cost < best_cost:
                best_cost, best_placement = new_cost, candidate.copy()
        else:
            env.place(placement)
            if evaluator is not None:
                evaluator.reject()

    for _ in range(calibration):
        candidate, changed, relocation = random_neighbor()
        new_cost = evaluate(candidate, changed)
        evaluations += 1
        delta = new_cost - current_cost
        if delta > 0:
            positive_deltas.append(delta)
        finish(candidate, changed, relocation, new_cost, delta <= 0)

    scale = (float(np.median(positive_deltas)) if positive_deltas else
             max(abs(current_cost) * 1e-3, np.finfo(float).eps))
    initial_temperature = max(
        -scale / math.log(initial_acceptance), np.finfo(float).eps)
    operators = ("proxy", "bottleneck", "swap", "relocation")
    dimension = 8
    matrices = [{name: np.eye(dimension) for name in operators}
                for _ in range(phase_bins)]
    targets = [{name: np.zeros(dimension) for name in operators}
               for _ in range(phase_bins)]
    counts = {name: 0 for name in operators}
    phase_counts = [{name: 0 for name in operators}
                    for _ in range(phase_bins)]
    reward_sums = {name: 0.0 for name in operators}
    supervised_reward_sums = {name: 0.0 for name in operators}
    bandit_primary_counts = {name: 0 for name in operators}
    exploration_selections = proxy_fallback_selections = 0
    proxy_gate_overrides = proxy_gate_candidate_proposals = 0
    supervised_pools = supervised_operator_labels = 0
    positive_reward_deltas = []
    macro_action_counts = {name: 0 for name in operators}
    macro_phase_counts = [{name: 0 for name in operators}
                          for _ in range(phase_bins)]
    macro_reward_sums = {name: 0.0 for name in operators}
    macro_operator = macro_context = macro_phase = None
    macro_start_best = None
    macro_steps = macro_updates = 0
    decision_steps = 0
    recent_acceptance = 0.0
    last_reward = 0.0
    stagnant = 0
    remaining = max(1, n_iter - calibration)
    progress_interval = max(1, n_iter // 50)

    while evaluations < n_iter:
        decision_steps += 1
        progress = (evaluations - calibration) / remaining
        temperature = initial_temperature * (
            final_temperature_ratio ** progress)
        dominance = (float(evaluator.stage_costs.max()) /
                     max(float(evaluator.stage_costs.sum()),
                         np.finfo(float).eps)) if evaluator is not None else 0.0
        context = np.asarray([
            1.0, progress,
            math.log(max(temperature / initial_temperature,
                         np.finfo(float).eps)) /
            math.log(final_temperature_ratio),
            recent_acceptance, min(1.0, stagnant / 500.0),
            best_cost / max(current_cost, np.finfo(float).eps),
            last_reward, dominance,
        ], dtype=np.float64)
        phase = min(phase_bins - 1, int(progress * phase_bins))
        supervised_step = (
            full_information_stride > 0 and
            decision_steps % full_information_stride == 0 and
            evaluations + len(operators) <= n_iter)
        supervised_rewards = None
        models_discounted = False
        if supervised_step:
            candidate_rows = [proposal(name) for name in operators]
            costs, pending_rows = [], []
            for candidate_row, changed_row, _ in candidate_rows:
                env.place(candidate_row)
                if evaluator is not None:
                    costs.append(float(evaluator.candidate_cost(changed_row)))
                    pending_rows.append((
                        dict(evaluator.pending or {}),
                        dict(evaluator.pending_details or {})))
                    evaluator.reject()
                else:
                    costs.append(float(env.evaluate()))
                    pending_rows.append(None)
                env.place(placement)
            operator = "proxy"
            selected = 0
            candidate, changed, relocation = candidate_rows[selected]
            new_cost = costs[selected]
            env.place(candidate)
            if evaluator is not None:
                evaluator.pending = pending_rows[selected][0]
                evaluator.pending_details = pending_rows[selected][1]
            evaluations += len(operators)
            supervised_rewards = {
                name: float(np.tanh(-(cost - current_cost) / scale))
                for name, cost in zip(operators, costs)}
            supervised_pools += 1
            supervised_operator_labels += len(operators)
        else:
            ranked_operators = list(operators)
            if macro_horizon and control != "proxy":
                if macro_operator is None:
                    macro_context = context.copy()
                    macro_phase = phase
                    macro_start_best = best_cost
                    macro_steps = 0
                    unseen = [name for name in operators
                              if macro_phase_counts[phase][name] == 0]
                    if control == "uniform":
                        macro_operator = operators[
                            selection_rng.randrange(len(operators))]
                    elif unseen:
                        macro_operator = unseen[0]
                    elif selection_rng.random() < minimum_exploration:
                        macro_operator = operators[
                            selection_rng.randrange(len(operators))]
                        exploration_selections += 1
                    else:
                        identity = np.eye(dimension)
                        scores = {}
                        for name in operators:
                            cholesky = np.linalg.cholesky(
                                matrices[phase][name])
                            estimate_weights = np.linalg.solve(
                                cholesky.T, np.linalg.solve(
                                    cholesky, targets[phase][name]))
                            if selection_policy == "lints":
                                sample = (estimate_weights + thompson_scale *
                                          np.linalg.solve(
                                              cholesky.T,
                                              thompson_rng.standard_normal(
                                                  dimension)))
                                scores[name] = float(sample @ context)
                            else:
                                uncertainty = ucb_alpha * float(np.linalg.norm(
                                    np.linalg.solve(cholesky, context)))
                                scores[name] = float(
                                    estimate_weights @ context + uncertainty)
                        macro_operator = max(scores, key=scores.get)
                    macro_action_counts[macro_operator] += 1
                    macro_phase_counts[phase][macro_operator] += 1
                    bandit_primary_counts[macro_operator] += 1
                primary = macro_operator
                slate = (["proxy"] if primary == "proxy" else
                         [primary, "proxy"])
                candidate_rows = [proposal(name) for name in slate]
                proxy_scores = [proxy_gain(row[0], row[1])
                                for row in candidate_rows]
                selected = int(np.argmax(proxy_scores))
                operator = slate[selected]
                proxy_gate_overrides += int(operator != primary)
                proxy_gate_candidate_proposals += len(slate)
                candidate, changed, relocation = candidate_rows[selected]
            else:
                unseen = [name for name in operators
                          if phase_counts[phase][name] == 0]
                if control == "uniform":
                    operator = operators[
                        selection_rng.randrange(len(operators))]
                elif control == "proxy":
                    operator = "proxy"
                elif unseen and not full_information_stride:
                    operator = unseen[0]
                    ranked_operators = unseen + [name for name in operators
                                                 if name not in unseen]
                elif selection_rng.random() < minimum_exploration:
                    selection_rng.shuffle(ranked_operators)
                    operator = ranked_operators[0]
                    exploration_selections += 1
                else:
                    identity = np.eye(dimension)
                    scores, estimates, uncertainties = {}, {}, {}
                    for name in operators:
                        matrices[phase][name] = (
                            discount * matrices[phase][name] +
                            (1.0 - discount) * identity)
                        targets[phase][name] *= discount
                        models_discounted = True
                        cholesky = np.linalg.cholesky(matrices[phase][name])
                        estimate_weights = np.linalg.solve(
                            cholesky.T,
                            np.linalg.solve(cholesky, targets[phase][name]))
                        estimate = float(estimate_weights @ context)
                        uncertainty_vector = np.linalg.solve(
                            cholesky, context)
                        uncertainty = ucb_alpha * float(np.linalg.norm(
                            uncertainty_vector))
                        estimates[name] = estimate
                        uncertainties[name] = uncertainty
                        if selection_policy == "lints":
                            sample = (estimate_weights + thompson_scale *
                                      np.linalg.solve(
                                          cholesky.T,
                                          thompson_rng.standard_normal(
                                              dimension)))
                            scores[name] = float(sample @ context)
                        else:
                            scores[name] = estimate + uncertainty
                    ranked_operators = sorted(
                        operators, key=lambda name: scores[name], reverse=True)
                    operator = ranked_operators[0]
                    if (selection_policy == "linucb" and
                            operator != "proxy" and
                            estimates[operator] - uncertainties[operator] <=
                            estimates["proxy"] + uncertainties["proxy"]):
                        operator = "proxy"
                        ranked_operators.remove("proxy")
                        ranked_operators.insert(0, "proxy")
                        proxy_fallback_selections += 1
                bandit_primary_counts[operator] += 1
                if control == "learned" and proxy_gate_slate > 1:
                    slate = ranked_operators[:proxy_gate_slate]
                    if operator not in slate:
                        slate[-1] = operator
                    if slate[0] != operator:
                        slate.remove(operator)
                        slate.insert(0, operator)
                    candidate_rows = [proposal(name) for name in slate]
                    proxy_scores = [proxy_gain(row[0], row[1])
                                    for row in candidate_rows]
                    selected = int(np.argmax(proxy_scores))
                    chosen = slate[selected]
                    proxy_gate_overrides += int(chosen != operator)
                    proxy_gate_candidate_proposals += len(slate)
                    operator = chosen
                    candidate, changed, relocation = candidate_rows[selected]
                else:
                    candidate, changed, relocation = proposal(operator)
            new_cost = evaluate(candidate, changed)
            evaluations += 1
        old_cost = current_cost
        old_best = best_cost
        delta = new_cost - old_cost
        accept = (delta <= 0 or acceptance_rng.random() <
                  math.exp(-delta / max(temperature, np.finfo(float).eps)))
        if reward_mode == "outcome" and supervised_rewards is None:
            positive = max(0.0, -delta)
            if positive:
                positive_reward_deltas.append(positive)
                if len(positive_reward_deltas) > reward_window:
                    positive_reward_deltas.pop(0)
            base = (0.6 if new_cost < old_best else
                    0.3 if delta < 0 else
                    0.1 if accept else 0.0)
            magnitude = (positive / max(positive_reward_deltas)
                         if positive_reward_deltas and positive else 0.0)
            reward = float(base + (1.0 - base) * min(1.0, magnitude))
        else:
            reward = float(np.tanh(-delta / scale))
        finish(candidate, changed, relocation, new_cost, accept)
        if macro_horizon and control != "proxy":
            macro_steps += 1
            if macro_steps >= macro_horizon or evaluations == n_iter:
                macro_reward = float(np.tanh(
                    max(0.0, macro_start_best - best_cost) / scale))
                if control == "learned":
                    identity = np.eye(dimension)
                    for name in operators:
                        matrices[macro_phase][name] = (
                            discount * matrices[macro_phase][name] +
                            (1.0 - discount) * identity)
                        targets[macro_phase][name] *= discount
                    matrices[macro_phase][macro_operator] += np.outer(
                        macro_context, macro_context)
                    targets[macro_phase][macro_operator] += (
                        macro_reward * macro_context)
                    macro_updates += 1
                macro_reward_sums[macro_operator] += macro_reward
                last_reward = macro_reward
                macro_operator = macro_context = macro_phase = None
                macro_start_best = None
                macro_steps = 0
        if control == "learned" and supervised_rewards is not None:
            identity = np.eye(dimension)
            for name in operators:
                matrices[phase][name] = (
                    discount * matrices[phase][name] +
                    (1.0 - discount) * identity)
                targets[phase][name] *= discount
                matrices[phase][name] += np.outer(context, context)
                targets[phase][name] += supervised_rewards[name] * context
                supervised_reward_sums[name] += supervised_rewards[name]
        elif (control == "learned" and not full_information_stride and
              not macro_horizon):
            if selection_policy == "lints" and not models_discounted:
                identity = np.eye(dimension)
                for name in operators:
                    matrices[phase][name] = (
                        discount * matrices[phase][name] +
                        (1.0 - discount) * identity)
                    targets[phase][name] *= discount
            matrices[phase][operator] += np.outer(context, context)
            targets[phase][operator] += reward * context
        counts[operator] += 1
        phase_counts[phase][operator] += 1
        reward_sums[operator] += reward
        if not macro_horizon:
            last_reward = reward
        stagnant = 0 if new_cost < old_best else stagnant + 1
        recent_acceptance = 0.95 * recent_acceptance + 0.05 * float(accept)
        if evaluations % progress_interval == 0 or evaluations == n_iter:
            print(f"[OP-SA] proposals={evaluations}/{n_iter} "
                  f"best_us={best_cost * 1e6:.6f} operator={operator}",
                  flush=True)

    env.place(best_placement)
    if metadata is not None:
        metadata.update({
            "method": ("delayed_macro_operator_lints_sa_v5"
                       if macro_horizon else
                       "full_information_operator_gate_sa_v3"
                       if full_information_stride else
                       f"proxy_gated_discounted_{selection_policy}_sa_v4"
                       if proxy_gate_slate > 1 else
                       "phase_discounted_operator_bandit_fixed_sa_v2"),
            "learning_enabled": control == "learned",
            "control": control,
            "unsupervised_online_learning": True,
            "operators": list(operators),
            "operator_selection_counts": counts,
            "phase_operator_selection_counts": phase_counts,
            "operator_reward_sums": reward_sums,
            "macro_action_counts": macro_action_counts,
            "macro_phase_action_counts": macro_phase_counts,
            "macro_reward_sums": macro_reward_sums,
            "macro_horizon": macro_horizon,
            "macro_updates": macro_updates,
            "bandit_primary_selection_counts": bandit_primary_counts,
            "supervised_operator_reward_sums": supervised_reward_sums,
            "ucb_alpha": ucb_alpha,
            "selection_policy": selection_policy,
            "thompson_scale": thompson_scale,
            "discount": discount,
            "minimum_exploration": minimum_exploration,
            "phase_bins": phase_bins,
            "full_information_stride": full_information_stride,
            "supervised_pools": supervised_pools,
            "supervised_operator_labels": supervised_operator_labels,
            "decision_steps": decision_steps,
            "proxy_gate_slate": proxy_gate_slate,
            "proxy_gate_candidate_proposals": proxy_gate_candidate_proposals,
            "proxy_gate_overrides": proxy_gate_overrides,
            "exploration_selections": exploration_selections,
            "proxy_fallback_selections": proxy_fallback_selections,
            "proxy_fallback_rule": "nonproxy_lcb_exceeds_proxy_ucb",
            "proxy_candidates": proxy_candidates,
            "initial_cost": initial_cost,
            "best_cost": best_cost,
            "accepted_moves": accepted,
            "improving_moves": improving,
            "update_count": (
                macro_updates if control == "learned" and macro_horizon else
                supervised_operator_labels if control == "learned" and
                full_information_stride else
                sum(counts.values()) if control == "learned" else 0),
            "initial_placement_evaluations": 1,
            "total_objective_evaluations": n_iter + 1,
            "acceptance_rule": "fixed_metropolis",
            "reward": ("delayed_segment_best_cost_improvement"
                       if macro_horizon else
                       "alns_outcome_plus_window_normalized_improvement"
                       if reward_mode == "outcome" else
                       "signed_proposed_cost_reduction"),
            "reward_window": reward_window,
        })
    return best_cost


def run_proxy_guided_sa(env: MultiChipEnvironment, n_iter: int = 100000,
                        candidate_count: int = 16,
                        focus_fraction: float = 0.5,
                        focus_stage_slack: float = 0.05,
                        epsilon_final: float = 0.25,
                        seed: int = None, metadata: dict = None) -> float:
    """Training-free proxy-ranked fixed SA.

    Each step creates legal neighbors, ranks them by local communication
    byte-hop reduction, samples with a decaying epsilon schedule, and applies
    the ordinary fixed Metropolis rule to the exact configured objective.
    No labels, checkpoint, pretrained weights, or gradient updates are used.
    """
    result = run_ppo_asa(
        env, n_iter=n_iter, candidate_count=candidate_count,
        rollout_steps=256, device="cpu", learning_rate=3e-4,
        update_epochs=1, minibatch_size=1, hidden_dim=16,
        disable_learning=True, proposal_learner="bandit",
        bandit_control="proxy_greedy", bandit_epsilon_final=epsilon_final,
        bandit_epsilon_decay_fraction=1.0,
        bandit_exploration="epsilon_top_m",
        bandit_top_m=min(4, candidate_count),
        bandit_global_epsilon=0.05, bandit_feature_set="legacy9",
        reward_mode="proposed_delta", focus_fraction=focus_fraction,
        focus_stage_slack=focus_stage_slack, annealing_schedule="fixed",
        seed=seed, metadata=metadata)
    if metadata is not None:
        metadata.update({
            "method": "training_free_proxy_guided_fixed_sa",
            "unsupervised": True,
            "requires_training_data": False,
            "proxy_definition": "local_communication_byte_hop_reduction",
        })
    return result


def run_ddpg_guided(env: MultiChipEnvironment, n_episodes: int = 3000,
                    batch_size: int = 64, baseline_trials: int = 1000,
                    batch_z: int = 3, train_every: int = 1,
                    device: str = None, save_checkpoint: str = None,
                    load_checkpoint: str = None, checkpoint_every: int = 100,
                    diagnostics_path: str = None, run_metadata: dict = None,
                    agent_arch: str = "paper_cnn", reward_scale: float = 1.0,
                    exploration_decay_placements: int = None,
                    diagnostics_every: int = 30, top_k: int = 8,
                    demo_iterations: int = 2000, pretrain_updates: int = 200,
                    bc_decay_placements: int = 2000,
                    retain_deterministic_candidates: bool = True,
                    asa_options=None, random_start_placements: int = 100,
                    disable_learning: bool = False, seed: int = None) -> float:
    """Sample-efficient DDPG variant for the discrete placement interface.

    This experimental mode keeps a deterministic continuous actor but fixes
    the failure observed in the 3,000-placement Kaggle run: the environment no
    longer repairs almost every action behind the critic's back.  The actor
    proposes a proto-action, the critic ranks nearby legal actions, and replay
    stores the selected legal action.  ASA supplies one valid demonstration;
    complete-episode returns and prioritized replay make every sparse-reward
    transition trainable.  Twin critics, Huber loss, delayed actor updates and
    gradient clipping bound the critic feedback used by the actor.
    """
    if not HAS_TORCH:
        raise RuntimeError("guided DDPG requires PyTorch")
    if min(n_episodes, batch_z, train_every, diagnostics_every, top_k,
           demo_iterations, bc_decay_placements) <= 0 or pretrain_updates < 0:
        raise ValueError("guided DDPG budgets and intervals must be positive")
    if random_start_placements < 0:
        raise ValueError("guided random-start placements must be nonnegative")
    if exploration_decay_placements is None:
        exploration_decay_placements = max(1, int(0.8 * n_episodes))
    guided_asa_options = dict(asa_options or {})
    guided_asa_options.pop("metadata", None)
    asa_fingerprint_options = tuple(sorted(
        (key, value) for key, value in guided_asa_options.items()
        if key != "diagnostics_path"))

    signature = hashlib.sha256()
    signature.update(np.ascontiguousarray(env.task_graph).tobytes())
    signature.update(np.ascontiguousarray(env.compute_latency).tobytes())
    signature.update(np.ascontiguousarray(env.allowed_cores).tobytes())
    signature.update(repr((env.topo, env.topo.on_chip_latency,
                           env.topo.off_chip_latency, batch_z, env.timing_units,
                           agent_arch, reward_scale, exploration_decay_placements,
                           train_every, top_k, demo_iterations, pretrain_updates,
                           bc_decay_placements, random_start_placements,
                           disable_learning, seed, asa_fingerprint_options,
                           "ddpg-guided-v1")).encode())
    fingerprint = signature.hexdigest()
    checkpoint = None
    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Guided checkpoint is from a different configuration")
        baseline_latency = checkpoint["baseline_latency"]
        print(f"[GUIDED] Resuming from placement {checkpoint['episode']}; "
              f"baseline B={baseline_latency:.6g}")
    else:
        print(f"[GUIDED] Computing random-search baseline B ({baseline_trials} trials)...")
        baseline_latency = run_random(env, n_trials=baseline_trials)
        baseline_placement = env.placement.copy()
        print(f"[GUIDED] Baseline B = {baseline_latency:.6g}")

    batch_z = max(1, min(batch_z, env.num_tasks))
    action_dim = 2 * batch_z
    mapper = MultiChipCoreMapper(
        env, baseline_latency=baseline_latency, batch_z=batch_z,
        reward_mode="sparse", reward_scale=reward_scale, compact_state=True)
    state_dim = mapper.state_rows * mapper.state_cols + env.num_tasks
    agent = DDPGAgent(
        state_dim=state_dim, action_dim=action_dim, device=device,
        agent_arch=agent_arch, rows=mapper.state_rows, cols=mapper.state_cols,
        num_tasks=env.num_tasks, stable=True, lr_critic=3e-4)
    stream_sequence = np.random.SeedSequence(seed)
    action_sequence, replay_sequence = stream_sequence.spawn(2)
    action_rng = np.random.default_rng(action_sequence)
    replay_buffer = GuidedReplayBuffer(seed=replay_sequence)
    reward_normalizer = max(mapper._potential_normalizer, np.finfo(float).eps)
    print(f"[GUIDED] Compact state: {mapper.state_rows}x{mapper.state_cols}; "
          f"legal top-k={top_k}; twin critics; episode-return replay")

    def add_trajectory(transitions, terminal_reward, is_demo):
        normalized_terminal = float(terminal_reward) / reward_normalizer
        horizon = len(transitions)
        for index, (state, executed_action) in enumerate(transitions):
            return_target = (agent.gamma ** (horizon - index - 1)) * normalized_terminal
            replay_buffer.add(state, executed_action, return_target,
                              is_demo=is_demo,
                              priority=2.0 if is_demo else None)
        return normalized_terminal

    def exact_trajectory(physical_placement):
        grid_placement = mapper.topology_to_grid_placement(physical_placement)
        state = mapper.reset()
        transitions, done, grid, cost, reward = [], False, "", 0.0, 0.0
        while not done:
            n_this = min(batch_z, env.num_tasks - mapper._task_ptr)
            cores = grid_placement[mapper._task_ptr:mapper._task_ptr + n_this]
            encoded = []
            for core in cores:
                encoded.extend(mapper._grid_core_to_action_pair(int(core)))
            while len(encoded) < action_dim:
                encoded.extend((0.0, 0.0))
            transitions.append((state.copy(), np.asarray(encoded, dtype=np.float32)))
            reward, done, grid, cost = mapper.step_legal(cores)
            state = mapper._occ_map()
        return transitions, reward, grid, cost

    def policy_trajectory(explore, noise_scale, uniform_random=False):
        state = mapper.reset()
        transitions, proto_actions = [], []
        done, grid, final_cost, terminal_reward = False, "", 0.0, 0.0
        while not done:
            if uniform_random:
                n_this = min(batch_z, env.num_tasks - mapper._task_ptr)
                free = np.asarray(
                    [core for core in mapper._allowed_grid
                     if int(core) not in mapper._occupied], dtype=np.int32)
                selected_cores = action_rng.choice(free, n_this, replace=False)
                encoded = []
                for core in selected_cores:
                    encoded.extend(mapper._grid_core_to_action_pair(int(core)))
                while len(encoded) < action_dim:
                    encoded.extend((0.0, 0.0))
                proto = executed_action = np.asarray(encoded, dtype=np.float32)
            else:
                proto = agent.select_action(
                    state, noise_scale=noise_scale, explore=explore,
                    rng=action_rng if explore else None)
                candidates, core_batches = mapper.legal_action_candidates(
                    proto, top_k=top_k)
                scores = agent.score_actions(state, candidates)
                selected = int(np.argmax(scores))
                executed_action = candidates[selected]
                selected_cores = core_batches[selected]
            transitions.append((state.copy(), executed_action.copy()))
            proto_actions.append(proto.copy())
            terminal_reward, done, grid, final_cost = mapper.step_legal(selected_cores)
            state = mapper._occ_map()
        return (transitions, proto_actions, terminal_reward, grid, final_cost,
                mapper.get_placement().copy())

    # ASA -> demonstration trajectory. This reverses the old DDPG->ASA hybrid:
    # the heuristic now teaches the policy instead of only polishing its output.
    asa_metadata = {}
    print(f"[GUIDED] Generating ASA demonstration ({demo_iterations} candidates)...")
    guided_asa_options["metadata"] = asa_metadata
    asa_cost = run_adaptive_sa(
        env, n_iter=demo_iterations, **guided_asa_options)
    asa_placement = env.placement.copy()
    demo_cost, demo_placement, demo_source = asa_cost, asa_placement, "adaptive_sa"
    if checkpoint is not None and checkpoint["best_cost"] < demo_cost:
        demo_cost = checkpoint["best_cost"]
        demo_placement = np.asarray(checkpoint["best_placement"], dtype=np.int32).copy()
        demo_source = "checkpoint_best"
    elif checkpoint is None and baseline_latency < demo_cost:
        demo_cost, demo_placement = baseline_latency, baseline_placement.copy()
        demo_source = "random_baseline"
    demo_transitions, demo_reward, demo_grid, replayed_demo_cost = exact_trajectory(
        demo_placement)
    if not math.isclose(demo_cost, replayed_demo_cost, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(
            "ASA demonstration changed cost when replayed through legal actions: "
            f"{demo_cost:.12g} != {replayed_demo_cost:.12g}")
    add_trajectory(demo_transitions, demo_reward, is_demo=True)
    print(f"[GUIDED] Demonstration source={demo_source}, cost={demo_cost:.6g}; "
          f"{len(demo_transitions)} transitions added to replay")

    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent"])
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if agent.device.type == "cuda" and checkpoint.get("cuda_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_state"])
        if checkpoint.get("guided_action_rng_state") is not None:
            action_rng.bit_generator.state = checkpoint["guided_action_rng_state"]
        if checkpoint.get("guided_replay_rng_state") is not None:
            replay_buffer.rng.bit_generator.state = checkpoint["guided_replay_rng_state"]
        best_cost = checkpoint["best_cost"]
        best_placement = checkpoint["best_placement"]
        best_grid = checkpoint.get("best_grid")
        best_candidate_source = checkpoint.get("best_candidate_source", "guided_training")
        global_step = checkpoint.get("global_step", 0)
        start_episode = checkpoint["episode"]
        deterministic_candidate_evaluations = checkpoint.get(
            "deterministic_candidate_evaluations", 0)
    else:
        best_cost = min(baseline_latency, demo_cost)
        if demo_cost <= baseline_latency:
            best_placement, best_grid = demo_placement.copy(), demo_grid
            best_candidate_source = ("asa_demonstration"
                                     if demo_source == "adaptive_sa"
                                     else "random_baseline")
        else:
            best_placement, best_grid = baseline_placement.copy(), None
            best_candidate_source = "random_baseline"
        global_step = start_episode = deterministic_candidate_evaluations = 0
        if pretrain_updates and not disable_learning:
            print(f"[GUIDED] Pretraining on the demonstration ({pretrain_updates} updates)...")
            for _ in range(pretrain_updates):
                agent.train_guided(replay_buffer, batch_size=batch_size, bc_weight=5.0)
        elif disable_learning:
            print("[GUIDED] Learning disabled: running a matched untrained control")

    diagnostics_stream = None
    if diagnostics_path is not None:
        parent = os.path.dirname(diagnostics_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        diagnostics_stream = open(
            diagnostics_path, "a" if checkpoint is not None else "w",
            encoding="utf-8")

    def noise_at(episode):
        return max(0.01, 0.01 ** (episode / exploration_decay_placements))

    def save(ep):
        if save_checkpoint is None:
            return
        torch.save({
            "fingerprint": fingerprint,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "guided_action_rng_state": action_rng.bit_generator.state,
            "guided_replay_rng_state": replay_buffer.rng.bit_generator.state,
            "torch_state": torch.get_rng_state(),
            "cuda_state": (torch.cuda.get_rng_state_all()
                           if agent.device.type == "cuda" else None),
            "agent": agent.state_dict(), "best_cost": best_cost,
            "best_placement": best_placement, "best_grid": best_grid,
            "best_candidate_source": best_candidate_source,
            "global_step": global_step, "episode": ep,
            "baseline_latency": baseline_latency,
            "deterministic_candidate_evaluations": deterministic_candidate_evaluations,
        }, save_checkpoint)

    start_time = time.time()
    final_cost = demo_cost
    noise_scale = noise_at(start_episode)
    for ep in range(start_episode + 1, n_episodes + 1):
        noise_scale = noise_at(ep)
        uniform_random = ep <= random_start_placements
        (transitions, proto_actions, terminal_reward, grid, final_cost,
         placement) = policy_trajectory(
             explore=True, noise_scale=noise_scale,
             uniform_random=uniform_random)
        normalized_return = add_trajectory(transitions, terminal_reward, is_demo=False)
        global_step += len(transitions)
        bc_weight = (0.0 if disable_learning else
                     5.0 * max(0.0, 1.0 - ep / bc_decay_placements))
        losses = []
        updates = (0 if disable_learning else
                   max(1, math.ceil(len(transitions) / train_every)))
        for _ in range(updates):
            loss = agent.train_guided(replay_buffer, batch_size=batch_size,
                                      bc_weight=bc_weight)
            if loss is not None:
                losses.append(loss)

        if final_cost < best_cost:
            best_cost, best_grid = final_cost, grid
            best_placement = placement.copy()
            best_candidate_source = ("untrained_control_rollout"
                                     if disable_learning else "guided_training")

        write_diagnostics = (ep % diagnostics_every == 0 or ep == n_episodes)
        if write_diagnostics:
            (eval_transitions, eval_proto, _, eval_grid, deterministic_cost,
             deterministic_placement) = policy_trajectory(explore=False, noise_scale=0.0)
            deterministic_candidate_evaluations += 1
            retained = False
            if retain_deterministic_candidates and deterministic_cost < best_cost:
                best_cost, best_grid = deterministic_cost, eval_grid
                best_placement = deterministic_placement.copy()
                best_candidate_source = ("untrained_control_deterministic"
                                         if disable_learning else
                                         "guided_deterministic")
                retained = True
            env.place(placement)
            action_values = np.concatenate(proto_actions)
            finite_actor = [item["actor_loss"] for item in losses
                            if item["actor_loss"] is not None]
            finite_bc = [item["bc_loss"] for item in losses
                         if item["bc_loss"] is not None]
            record = {
                "episode": ep, "current_cost": final_cost,
                "best_cost": best_cost, "deterministic_cost": deterministic_cost,
                "rollout_source": "uniform_random" if uniform_random else "guided_actor",
                "learning_enabled": not disable_learning,
                "retained_deterministic_candidate": retained,
                "normalized_terminal_return": normalized_return,
                "noise_scale": noise_scale, "collision_repairs": mapper.collision_repairs,
                "unique_executed_cores": len(mapper.intended_cores),
                "deterministic_collision_repairs": 0,
                "deterministic_unique_executed_cores": env.num_tasks,
                "proto_action_mean": float(action_values.mean()),
                "proto_action_std": float(action_values.std()),
                "proto_action_saturated_fraction": float(
                    np.mean(np.abs(action_values) >= 0.999)),
                "actor_loss_mean": float(np.mean(finite_actor)) if finite_actor else None,
                "critic_loss_mean": float(np.mean(
                    [item["critic_loss"] for item in losses])) if losses else None,
                "bc_loss_mean": float(np.mean(finite_bc)) if finite_bc else None,
                "q_abs_max": float(max(item["q_abs_max"] for item in losses)) if losses else None,
                "critic_grad_norm_max": float(max(
                    item["critic_grad_norm"] for item in losses)) if losses else None,
                "actor_grad_norm_max": float(max(
                    item["actor_grad_norm"] for item in losses
                    if item["actor_grad_norm"] is not None)) if finite_actor else None,
                "gradient_updates": len(losses), "bc_weight": bc_weight,
                "replay_size": len(replay_buffer),
            }
            if diagnostics_stream is not None:
                diagnostics_stream.write(json.dumps(record, allow_nan=False) + "\n")
                diagnostics_stream.flush()

        if ep % 10 == 0 or ep == n_episodes:
            elapsed = time.time() - start_time
            per_ep = elapsed / max(1, ep - start_episode)
            eta = timedelta(seconds=int(per_ep * (n_episodes - ep)))
            print(f"# of guided placements: {ep:7d} | Current Cost: {final_cost:.6g} | "
                  f"Best Cost: {best_cost:.6g} | {per_ep:.2f}s/ep | ETA: {eta}")
        if save_checkpoint is not None and ep % checkpoint_every == 0:
            save(ep)
            print(f"[GUIDED] Checkpoint saved to {save_checkpoint} (placement {ep})")

    if n_episodes > start_episode:
        save(n_episodes)
    if diagnostics_stream is not None:
        diagnostics_stream.close()
    env.place(best_placement)
    if run_metadata is not None:
        run_metadata.update({
            "method": "ddpg_guided", "baseline_cost": baseline_latency,
            "asa_demonstration_cost": asa_cost,
            "demonstration_cost": demo_cost,
            "demonstration_source": demo_source,
            "asa_demonstration_evaluations": demo_iterations,
            "training_candidate_evaluations": n_episodes - start_episode,
            "deterministic_candidate_evaluations": deterministic_candidate_evaluations,
            "total_candidate_evaluations": (baseline_trials + demo_iterations +
                                            n_episodes - start_episode +
                                            deterministic_candidate_evaluations),
            "best_candidate_source": best_candidate_source,
            "best_cost": best_cost, "top_k": top_k,
            "pretrain_updates": 0 if disable_learning else pretrain_updates,
            "pretrain_updates_configured": pretrain_updates,
            "bc_decay_placements": bc_decay_placements,
            "random_start_placements": random_start_placements,
            "learning_enabled": not disable_learning,
            "random_seed": seed,
            "compact_state_shape": [mapper.state_rows, mapper.state_cols],
            "replay_target": "discounted_complete_episode_return",
            "critic": "twin_huber_clipped_gradient",
            "diagnostics_path": diagnostics_path, "asa": asa_metadata,
        })
    return best_cost


def run_ddpg_asa(env: MultiChipEnvironment, ddpg_placements: int,
                 asa_iterations: int, ddpg_options=None, asa_options=None,
                 metadata: dict = None) -> float:
    """Train DDPG, then refine its best placement with adaptive SA."""
    if ddpg_placements <= 0 or asa_iterations <= 0:
        raise ValueError("DDPG+ASA requires positive budgets for both phases")
    ddpg_metadata, asa_metadata = {}, {}
    options = dict(ddpg_options or {})
    options["run_metadata"] = ddpg_metadata
    ddpg_cost = run_ddpg(env, n_episodes=ddpg_placements, **options)
    ddpg_placement = env.placement.copy()
    asa_options = dict(asa_options or {})
    asa_options["metadata"] = asa_metadata
    final_cost = run_adaptive_sa(
        env, n_iter=asa_iterations, initial_placement=ddpg_placement,
        **asa_options)
    if metadata is not None:
        ddpg_total_evaluations = ddpg_metadata.get(
            "total_candidate_evaluations", ddpg_placements)
        metadata.update({
            "method": "ddpg_then_adaptive_simulated_annealing",
            "ddpg_best_cost": ddpg_cost,
            "final_best_cost": final_cost,
            "ddpg_complete_placement_evaluations": ddpg_placements,
            "ddpg_total_candidate_evaluations": ddpg_total_evaluations,
            "asa_candidate_evaluations": asa_iterations,
            "combined_candidate_evaluations": ddpg_total_evaluations + asa_iterations,
            "ddpg": ddpg_metadata,
            "asa": asa_metadata,
        })
    return final_cost

def run_sa(env: MultiChipEnvironment, n_iter: int = 100000,
           T_start: float = None, T_end: float = None,
           cooldown: float = 0.99, perturb_frac: float = 0.01,
           initial_acceptance: float = 0.8,
           final_temperature_ratio: float = 1e-6,
           calibration_trials: int = 32, metadata: dict = None,
           incremental: bool = True) -> float:
    """
    PAPER FIX (Algorithm 2, Sec 4.2): the paper's SA uses a FIXED cooldown
    factor of 0.99 in a nested loop (outer: cool temperature; inner:
    iterationmax trials per temperature), with a neighborhood function that
    "randomly changes the placement of 1% of logic cores" per trial, sized
    so T0/Tend are chosen to search ~1,000,000 total placements.

    The earlier version of this function used a single flat loop with a
    per-step geometric cooldown derived from n_iter, and a neighborhood of
    exactly ONE pairwise swap -- both structurally different from the
    paper. This version restructures to match: fixed `cooldown` factor,
    nested temperature loop, and `perturb_frac` (default 1%) of tasks
    reassigned per trial via a random permutation of their currently-held
    cores (keeps the placement a valid bijection while changing exactly
    that many task->core assignments).

    The paper says T0 and Tend are application-specific but does not publish
    their values.  By default, T0 is therefore measured from uphill deltas in
    the current objective units and Tend is a fixed ratio of T0.  Supplying
    both T_start and T_end retains the historical absolute-temperature mode.

    NOTE: `n_iter=100000` here is still far short of the paper's ~1,000,000
    -- raise it explicitly via the CLI if you want closer to paper-scale
    search; a printed warning fires below this threshold so it's never
    silently unclear which regime a given run used.
    """
    n = env.num_tasks
    if n_iter <= 0 or not (0 < cooldown < 1):
        raise ValueError("SA requires a positive budget and cooldown in (0,1)")
    if (T_start is None) != (T_end is None):
        raise ValueError("SA requires both T_start and T_end, or neither")
    calibrated_temperature = T_start is None
    if calibrated_temperature:
        if not (0 < initial_acceptance < 1):
            raise ValueError("SA initial_acceptance must be in (0,1)")
        if not (0 < final_temperature_ratio < 1):
            raise ValueError("SA final_temperature_ratio must be in (0,1)")
        if calibration_trials <= 0:
            raise ValueError("SA calibration_trials must be positive")
    elif not (0 < T_end < T_start):
        raise ValueError("SA requires T_start > T_end > 0")
    if n_iter < 1_000_000:
        print(f"[SA] WARNING: n_iter={n_iter} is far below the paper's ~1,000,000-placement "
              f"search budget (Sec 4.2). Results are not directly comparable to the paper's "
              f"reported SA baseline until run at closer to that scale.")

    allowed = env.allowed_cores.tolist()
    placement = np.array(random.sample(allowed, n), dtype=np.int32)
    env.place(placement)
    best_cost = env.evaluate()
    initial_cost = best_cost
    best_p = placement.copy()
    cur_cost = best_cost
    free_pool = _FreeCorePool(env.allowed_cores, placement)

    evaluator = None
    if incremental:
        try:
            evaluator = IncrementalPipelineEvaluator(env)
        except (TypeError, AttributeError):
            evaluator = None

    accepted = improving = total_done = 0
    positive_deltas = []
    progress_interval = max(1, n_iter // 50)
    next_progress = progress_interval

    def report_progress():
        nonlocal next_progress
        if total_done >= next_progress or total_done == n_iter:
            print(f"[SA] proposals={total_done}/{n_iter} "
                  f"best={best_cost:.9g}", flush=True)
            while next_progress <= total_done:
                next_progress += progress_interval

    def evaluate(candidate, changed):
        env.place(candidate)
        return (evaluator.candidate_cost(changed) if evaluator is not None
                else float(env.evaluate()))

    def finish(candidate, new_cost, relocation, accept):
        nonlocal placement, cur_cost, best_cost, best_p, accepted, improving
        if accept:
            if new_cost < cur_cost:
                improving += 1
            accepted += 1
            placement, cur_cost = candidate, new_cost
            if evaluator is not None:
                evaluator.accept()
            if relocation is not None:
                free_pool.accept_relocation(*relocation)
            if new_cost < best_cost:
                best_cost, best_p = new_cost, candidate.copy()
        else:
            env.place(placement)
            if evaluator is not None:
                evaluator.reject()

    n_perturb = max(1, min(n, max(2, round(perturb_frac * n))))
    if calibrated_temperature:
        calibration = min(n_iter, calibration_trials)
        for _ in range(calibration):
            candidate, changed, relocation = _placement_neighbor(
                placement, n_perturb, free_pool)
            new_cost = evaluate(candidate, changed)
            total_done += 1
            delta = new_cost - cur_cost
            if delta > 0:
                positive_deltas.append(delta)
            finish(candidate, new_cost, relocation, delta <= 0)
            report_progress()
        scale = (float(np.median(positive_deltas)) if positive_deltas else
                 max(abs(cur_cost) * 1e-3, np.finfo(float).eps))
        T_start = max(-scale / math.log(initial_acceptance),
                      np.finfo(float).eps)
        T_end = max(T_start * final_temperature_ratio,
                    np.finfo(float).tiny)

    num_temp_steps = max(1, math.ceil(math.log(T_end / T_start) / math.log(cooldown)))
    remaining = n_iter - total_done
    iters_per_temp = max(1, math.ceil(remaining / num_temp_steps))

    T = T_start
    while T > T_end and total_done < n_iter:
        for _ in range(iters_per_temp):
            # Paper's neighborhood: randomly perturb ~1% of logic cores.
            # Implemented as a random permutation of a random subset's
            # currently-assigned cores, which changes exactly that subset's
            # task->core assignments while keeping the placement a valid
            # bijection (no core assigned to two tasks).
            new_p, changed, relocation = _placement_neighbor(
                placement, n_perturb, free_pool)
            new_cost = evaluate(new_p, changed)
            delta = new_cost - cur_cost
            accept = (delta <= 0 or random.random() <
                      math.exp(-delta / max(T, np.finfo(float).eps)))
            finish(new_p, new_cost, relocation, accept)

            total_done += 1
            report_progress()
            if total_done >= n_iter:
                break
        T *= cooldown

    env.place(best_p)
    if metadata is not None:
        metadata.update({
            "method": "fixed_schedule_simulated_annealing",
            "temperature_mode": ("measured_objective_scale"
                                 if calibrated_temperature
                                 else "legacy_absolute"),
            "initial_cost": initial_cost,
            "best_cost": best_cost,
            "candidate_evaluations": total_done,
            "initial_placement_evaluations": 1,
            "total_objective_evaluations": total_done + 1,
            "accepted_moves": accepted,
            "improving_moves": improving,
            "initial_temperature": T_start,
            "ending_temperature": T_end,
            "final_temperature": T,
            "cooldown": cooldown,
            "perturb_fraction": perturb_frac,
            "calibration_trials": (min(n_iter, calibration_trials)
                                   if calibrated_temperature else 0),
            "incremental_evaluation": evaluator is not None,
        })
    return best_cost

def run_random(env: MultiChipEnvironment, n_trials: int = 1000) -> float:
    """Paper (Sec 4.2): Random Search samples 1,000,000 placements and keeps
    the best. `n_trials` defaults far below that for practicality (e.g. as
    DDPG's reward-baseline computation, which runs before every training
    session) -- raise explicitly via CLI for a paper-scale comparison."""
    if n_trials < 1_000_000:
        print(f"[RS] WARNING: n_trials={n_trials} is far below the paper's 1,000,000-sample "
              f"search budget (Sec 4.2). Results are not directly comparable to the paper's "
              f"reported RS baseline until run at closer to that scale.")
    n = env.num_tasks
    best_cost, best_p = float("inf"), None
    allowed = env.allowed_cores.tolist()
    progress_interval = max(1, n_trials // 50)
    next_progress = progress_interval
    for trial in range(1, n_trials + 1):
        p = np.array(random.sample(allowed, n), dtype=np.int32)
        env.place(p); cost = env.evaluate()
        if cost < best_cost: best_cost, best_p = cost, p.copy()
        if trial >= next_progress or trial == n_trials:
            print(f"[RS] samples={trial}/{n_trials} best={best_cost:.9g}",
                  flush=True)
            while next_progress <= trial:
                next_progress += progress_interval
    env.place(best_p)
    return best_cost


def measure_objective_sensitivity(env: MultiChipEnvironment, n_trials: int = 64) -> dict:
    """Sample valid placements without perturbing the caller's RNG stream."""
    if n_trials <= 1:
        raise ValueError("objective sensitivity requires at least two trials")
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    costs, best_cost, best_placement = [], float("inf"), None
    allowed = env.allowed_cores.tolist()
    try:
        for _ in range(n_trials):
            placement = np.asarray(random.sample(allowed, env.num_tasks),
                                   dtype=np.int32)
            env.place(placement)
            cost = float(env.evaluate())
            costs.append(cost)
            if cost < best_cost:
                best_cost, best_placement = cost, placement.copy()
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
    env.place(best_placement)
    values = np.asarray(costs, dtype=np.float64)
    breakdown = env.pipeline_breakdown()
    median = float(np.median(values))
    minimum, maximum = float(values.min()), float(values.max())
    relative_span = ((maximum - minimum) / median if median > 0 else 0.0)
    compute_floor = float(breakdown["compute_only_floor"])
    return {
        "trials": n_trials,
        "minimum": minimum,
        "median": median,
        "maximum": maximum,
        "relative_span": relative_span,
        "best_sample_breakdown": breakdown,
        "best_headroom_above_compute_floor": (
            (minimum - compute_floor) / minimum if minimum > 0 else 0.0),
    }


def run_sequential(env: MultiChipEnvironment) -> float:
    """Paper BS baseline: assign tasks by chip index, then core index."""
    if env.num_tasks > len(env.allowed_cores):
        raise ValueError("Sequential placement requires at least one physical core per task")
    placement = env.allowed_cores[:env.num_tasks].copy()
    env.place(placement)
    return env.evaluate()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-chip core placement")
    parser.add_argument("--algo", choices=["ddpg", "ddpg_guided", "ppo_masked",
                                            "ppo_asa", "proxy_sa", "operator_sa",
                                            "sa", "asa", "ddpg_asa",
                                            "random", "bs"],
                        default="ddpg",
                        help="Placement method; ddpg_guided uses ASA demonstrations, "
                             "legal top-k actions, episode-return replay and stable twin "
                             "critics; ppo_masked is an experimental exact legal-action "
                             "baseline; ppo_asa learns legal local proposals while keeping "
                             "ASA acceptance and temperature control; proxy_sa is a "
                             "training-free communication-guided fixed SA; ddpg remains the "
                             "paper-faithful baseline")
    parser.add_argument("--proxy_sa_candidates", type=int, default=16,
                        help="Legal neighbors ranked per proxy-guided SA step")
    parser.add_argument("--proxy_sa_focus_fraction", type=float, default=0.5,
                        help="Fraction of proxy-SA neighbors anchored near bottleneck stages")
    parser.add_argument("--proxy_sa_focus_stage_slack", type=float, default=0.05,
                        help="Relative stage-cost slack included in proxy-SA focus")
    parser.add_argument("--proxy_sa_epsilon_final", type=float, default=0.25,
                        help="Final exploration probability for proxy-guided SA")
    parser.add_argument("--operator_sa_proxy_candidates", type=int, default=8,
                        help="Cheap neighbors ranked by the proxy operator")
    parser.add_argument("--operator_sa_ucb_alpha", type=float, default=0.5,
                        help="LinUCB exploration strength for move-operator selection")
    parser.add_argument("--operator_sa_discount", type=float, default=0.995,
                        help="Recent-evidence discount for phase-specific operator models")
    parser.add_argument("--operator_sa_minimum_exploration", type=float, default=0.10,
                        help="Minimum uniform operator-exploration probability")
    parser.add_argument("--operator_sa_phase_bins", type=int, default=3,
                        help="Independent early/middle/late operator models")
    parser.add_argument("--operator_sa_full_information_stride", type=int,
                        default=0,
                        help="Every N decisions, label all four operators within budget")
    parser.add_argument("--operator_sa_selection_policy",
                        choices=["linucb", "lints"], default="linucb",
                        help="Contextual operator-selection rule")
    parser.add_argument("--operator_sa_proxy_gate_slate", type=int, default=1,
                        help="Cheaply proxy-screen this many bandit-ranked operators")
    parser.add_argument("--operator_sa_thompson_scale", type=float, default=0.25,
                        help="Posterior sampling scale for Linear Thompson Sampling")
    parser.add_argument("--operator_sa_reward",
                        choices=["signed", "outcome"], default="signed",
                        help="Selected-action operator credit")
    parser.add_argument("--operator_sa_reward_window", type=int, default=50,
                        help="Recent improvement window for outcome reward normalization")
    parser.add_argument("--operator_sa_macro_horizon", type=int, default=0,
                        help="Hold an operator strategy for N steps and learn from delayed best-cost gain")
    parser.add_argument("--operator_sa_disable_learning", action="store_true",
                        help="Select move operators uniformly as a matched control")
    parser.add_argument("--operator_sa_control",
                        choices=["learned", "uniform", "proxy"],
                        default="learned",
                        help="Learned selection or an exact uniform/proxy-only control")
    parser.add_argument("--chips_x", type=int, default=2)
    parser.add_argument("--chips_y", type=int, default=2)
    parser.add_argument("--rows", type=int, default=4, help="Rows per chip")
    parser.add_argument("--cols", type=int, default=4, help="Cols per chip")
    parser.add_argument("--on_lat", type=float, default=1.0, help="On-chip link latency")
    parser.add_argument("--off_lat", type=float, default=5.0, help="Off-chip link latency (hierarchical penalty)")
    parser.add_argument("--topology", type=str, default="mesh", choices=["mesh", "torus"],
                         help="Hardware routing topology. 'mesh' (default) matches the "
                              "paper's grid exactly. 'torus' adds wraparound edges on both "
                              "the on-chip mesh and inter-chip grid (paper Sec 4.4 mentions "
                              "torus as an alternative topology they also tested). Other "
                              "topology families (HNoC, dragonfly, arbitrary graphs) aren't "
                              "supported -- see multi_chip_topology.py docstring.")
    parser.add_argument("--iters", type=int, default=5000,
                        help="SA/ASA/random candidate evaluations; for ddpg_asa this is the ASA phase budget")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="DDPG training epochs; multiply by --placements_per_epoch "
                             "to obtain complete placements evaluated")
    parser.add_argument("--placements_per_epoch", type=int, default=1,
                        help="Complete placements generated in each declared epoch; "
                             "the paper states 30 (default 1 preserves historical runs)")
    parser.add_argument("--baseline_trials", type=int, default=1000,
                         help="Number of random-search trials used to compute the "
                              "baseline B for the sparse reward r_t = sqrt(B) - sqrt(L(P)) "
                              "(Algorithm 1, line 11). Larger = better/more stable "
                              "baseline estimate but slower startup.")
    parser.add_argument("--batch_z", type=int, default=3,
                         help="Number of logic cores placed per DDPG action, "
                              "per the paper's batched action [x1,y1,...,xz,yz] "
                              "(Sec 3.2). Clamped to num_tasks if larger.")
    parser.add_argument("--save_checkpoint", type=str, default=None,
                         help="Path to save an RL checkpoint (networks, optimizer state, "
                              "best result, noise schedule position, baseline B) every "
                              "--checkpoint_every episodes and once more at the end. "
                              "Without this, nothing persists between runs -- each run "
                              "starts from scratch.")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                         help="Path to resume training from a previously saved checkpoint. "
                              "If the file doesn't exist yet, training starts fresh and "
                              "will create it (combine with --save_checkpoint pointing to "
                              "the same path to make a run resumable from itself). Note: "
                              "DDPG replay is not persisted; masked PPO checkpoints are "
                              "saved only between complete on-policy rollout batches. "
                              "For PPO-ASA this loads a frozen proposal model and must be "
                              "combined with --ppo_asa_disable_learning.")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                         help="Save a checkpoint every N episodes (only used with "
                              "--save_checkpoint).")
    parser.add_argument("--ddpg_gamma", type=float, default=.98)
    parser.add_argument("--ddpg_disable_learning", action="store_true")
    parser.add_argument("--train_every", type=int, default=5,
                         help="Run one DDPG gradient update every N environment "
                              "steps instead of every step. Experience is still "
                              "recorded every step via the replay buffer -- this "
                              "only spaces out the (expensive) actor/critic "
                              "forward+backward passes. Set to 1 to train on "
                              "every step (original behavior, much slower at "
                              "large task counts).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                         help="RL device. The cpu branch defaults to CPU explicitly.")
    parser.add_argument("--cpu_threads", type=int, default=None,
                        help="PyTorch intra-op CPU threads (default: OMP_NUM_THREADS or all logical CPUs)")
    parser.add_argument("--cpu_interop_threads", type=int, default=1,
                        help="PyTorch inter-op CPU threads; keep at 1 for independent multiseed jobs")
    parser.add_argument("--agent_arch", choices=["mlp", "cnn", "paper_cnn"], default="mlp",
                        help="DDPG architecture: historical MLP, junior-derived augmented CNN, or Figure-9 paper CNN.")
    parser.add_argument("--reward_mode", choices=["sparse", "potential"], default="sparse",
                        help="RL reward: paper-style sparse terminal reward, or opt-in potential-based shaping with the same fixed-horizon discounted objective.")
    parser.add_argument("--reward_scale", type=float, default=None,
                        help="Multiply latency before sqrt reward. Default: 400e6 for "
                             "seconds-based timing (400-MHz cycle units), otherwise 1")
    parser.add_argument("--exploration_decay_placements", type=int, default=None,
                        help="Absolute complete-placement horizon at which OU noise reaches 0.01. "
                             "Set one fixed value for every stage of a resumed run")
    parser.add_argument("--retain_deterministic_candidates", action="store_true",
                        help="Allow deterministic diagnostic rollouts to update the saved best "
                             "placement; their extra objective evaluations are reported separately")
    parser.add_argument("--guided_top_k", type=int, default=8,
                        help="Legal action candidates ranked by the guided DDPG critics")
    parser.add_argument("--guided_demo_iterations", type=int, default=2000,
                        help="Adaptive-SA candidate budget used to create the guided demonstration")
    parser.add_argument("--guided_pretrain_updates", type=int, default=200,
                        help="Twin-critic/behavior-cloning updates before guided online training")
    parser.add_argument("--guided_bc_decay_placements", type=int, default=2000,
                        help="Placements over which the guided behavior-cloning weight decays to zero")
    parser.add_argument("--guided_random_start_placements", type=int, default=100,
                        help="Initial guided placements sampled uniformly from legal cores to "
                             "broaden replay coverage before relying on actor proposals")
    parser.add_argument("--guided_disable_learning", action="store_true",
                        help="Disable guided pretraining and online gradient updates for a "
                             "matched untrained-policy control")
    parser.add_argument("--ppo_learning_rate", type=float, default=3e-4)
    parser.add_argument("--ppo_rollout_placements", type=int, default=10,
                        help="Complete placements collected before each masked-PPO update")
    parser.add_argument("--ppo_update_epochs", type=int, default=4)
    parser.add_argument("--ppo_minibatch_size", type=int, default=512)
    parser.add_argument("--ppo_gamma", type=float, default=0.98)
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_clip_ratio", type=float, default=0.2)
    parser.add_argument("--ppo_entropy_coef", type=float, default=0.01)
    parser.add_argument("--ppo_value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_hidden_dim", type=int, default=256)
    parser.add_argument("--ppo_disable_learning", action="store_true",
                        help="Freeze masked PPO for a matched untrained-policy control")
    parser.add_argument("--ppo_asa_candidates", type=int, default=16,
                        help="Unevaluated legal neighbors ranked at each PPO-ASA step")
    parser.add_argument(
        "--ppo_asa_learner", choices=["ppo", "bandit"], default="ppo",
        help="Proposal learner: PPO or an immediate-gain contextual bandit")
    parser.add_argument("--ppo_asa_bandit_replay", type=int, default=50000,
                        help="Maximum selected-move observations retained by the bandit")
    parser.add_argument("--ppo_asa_bandit_recent_fraction", type=float, default=0.0)
    parser.add_argument("--ppo_asa_bandit_recent_window", type=int, default=10000)
    parser.add_argument("--ppo_asa_bandit_warmup", type=int, default=512,
                        help="Uniform proposals collected before bandit exploitation")
    parser.add_argument("--ppo_asa_bandit_epsilon_final", type=float,
                        default=0.50,
                        help="Final random-proposal probability for the bandit")
    parser.add_argument("--ppo_asa_bandit_epsilon_decay", type=float,
                        default=1.00,
                        help="Fraction of the run used to decay bandit exploration")
    parser.add_argument(
        "--ppo_asa_bandit_control",
        choices=["uniform", "proxy_greedy", "bottleneck_random", "checkpoint"],
        default="uniform",
        help=("Non-learned comparator used with --ppo_asa_disable_learning: "
              "uniform selection, epsilon-greedy proxy ranking, frozen loaded "
              "checkpoint scoring, or uniform selection within the focused pool"))
    parser.add_argument(
        "--ppo_asa_bandit_exploration",
        choices=["epsilon_all", "epsilon_top_m"], default="epsilon_all",
        help=("Bandit exploration: uniform across the whole pool or split "
              "between greedy, top-m, and a small global-uniform mass"))
    parser.add_argument("--ppo_asa_bandit_top_m", type=int, default=4,
                        help="Top-scored candidates used by epsilon_top_m")
    parser.add_argument(
        "--ppo_asa_bandit_global_epsilon", type=float, default=0.05,
        help="Global-uniform probability retained by epsilon_top_m")
    parser.add_argument(
        "--ppo_asa_bandit_feature_set",
        choices=["legacy9", "contention12", "topology16"], default="legacy9",
        help=("Candidate features: historical hop proxies or those features "
              "plus contention deltas and optional portable topology context"))
    parser.add_argument(
        "--ppo_asa_bandit_shuffle_labels", action="store_true",
        help="Shuffle bandit targets within each update as a learning control")
    parser.add_argument(
        "--ppo_asa_bandit_scoring", choices=["direct", "proxy_residual"],
        default="direct", help="Direct gain regression or bounded proxy correction")
    parser.add_argument(
        "--ppo_asa_bandit_residual_limit", type=float, default=0.25,
        help="Maximum absolute learned score correction in proxy_residual mode")
    parser.add_argument(
        "--ppo_asa_state_encoding", choices=["legacy_clipped", "relative_cost"],
        default="legacy_clipped", help="Historical clipped progress or cost ratios")
    parser.add_argument(
        "--ppo_asa_bandit_pool_supervision_stride", type=int, default=0,
        help="Evaluate and train on a whole candidate pool every N proposal steps")
    parser.add_argument("--ppo_asa_bandit_ranking_weight", type=float, default=0.0)
    parser.add_argument("--ppo_asa_rollout_steps", type=int, default=256,
                        help="Accepted/rejected proposal transitions per PPO-ASA update")
    parser.add_argument("--ppo_asa_weight_decay", type=float, default=0.0,
                        help="Adam weight decay for the PPO-ASA proposal model")
    parser.add_argument(
        "--ppo_asa_update_style",
        choices=["shared_gae", "reference_separate"], default="shared_gae",
        help="PPO update implementation: historical shared actor/critic or "
             "RL-Based-SA-compatible separate actor and critic")
    parser.add_argument("--ppo_asa_pool_small_move_fraction", type=float, default=0.0,
                        help="Fraction of candidates using two-task moves; calibration unchanged")
    parser.add_argument("--ppo_asa_pool_focus_fraction", type=float, default=None,
                        help="Override candidate-pool focus only; keep calibration focus unchanged")
    parser.add_argument(
        "--ppo_asa_reward_mode",
        choices=["realized", "proposed_delta", "positive_delta",
                 "planted_proxy"],
        default="realized",
        help="PPO-ASA training reward: post-acceptance realized improvement, "
             "signed proposed improvement, improvement-only positive delta, "
             "or a planted proxy signal")
    parser.add_argument(
        "--ppo_asa_annealing", choices=["adaptive", "fixed"],
        default="adaptive",
        help=("Annealing mechanics beneath the learned proposal policy. "
              "fixed reuses measured temperature, 0.99 cooldown, and the "
              "constant 1%% fixed-SA neighborhood."))
    parser.add_argument(
        "--ppo_asa_oracle_diagnostics", default=None,
        help="Write best-of-candidate headroom and feature correlations to JSON")
    parser.add_argument(
        "--ppo_asa_oracle_pools", type=int, default=0,
        help="Number of candidate pools in which to evaluate every candidate")
    parser.add_argument(
        "--ppo_asa_oracle_stride", type=int, default=1,
        help="Policy-step spacing between oracle-audited candidate pools")
    parser.add_argument(
        "--ppo_asa_oracle_dataset", default=None,
        help="Write oracle global states, candidate features, and exact gains "
             "to a compressed NPZ dataset")
    parser.add_argument(
        "--ppo_asa_finetune_loaded", action="store_true",
        help="Explicitly allow PPO updates from a loaded proposal checkpoint")
    parser.add_argument("--ppo_asa_disable_learning", action="store_true",
                        help="Keep learned-SA proposals exactly uniform for a matched control")
    parser.add_argument("--ppo_asa_focus_bottleneck", action="store_true",
                        help="Anchor every PPO-ASA candidate on the current bottleneck stage")
    parser.add_argument("--ppo_asa_focus_fraction", type=float, default=0.0,
                        help="Fraction of each PPO-ASA pool anchored on the bottleneck stage")
    parser.add_argument(
        "--ppo_asa_focus_stage_slack", type=float, default=0.0,
        help=("Also focus on stages within this fractional latency of the "
              "current bottleneck; 0 retains strict top-stage focus"))
    parser.add_argument("--ppo_asa_restart_interval", type=int, default=0,
                        help="Start an independent random placement chain after this many proposals; 0 disables restarts")
    parser.add_argument("--ppo_asa_update_at_chain_end", action="store_true",
                        help="Update PPO at every restart boundary instead of batching transitions across chains")
    parser.add_argument("--compute_ops", type=str, default=None,
                         help="Optional .npy file containing MAC operations per task; "
                              "converted with Table 1's 128 MACs/core at 400 MHz.")
    parser.add_argument("--timing_model", choices=["proxy", "full_frame", "paper_pipeline"],
                        default="proxy",
                        help="paper_pipeline adds reconstructed block scaling, XY routing "
                             "and shared-link contention to full-frame compute/traffic")
    parser.add_argument("--routing_model", choices=["legacy_distance", "paper_xy"],
                        default="legacy_distance",
                        help="paper_xy uses Figure-3 chip-periphery gateways and directed XY routes")
    parser.add_argument("--conv_blocks", type=int, default=4,
                        help="Blocks per frame for paper_pipeline CONV timing; 4 follows the "
                             "paper's illustrative Figure 7 because workload-specific values "
                             "are not published")
    parser.add_argument("--report", help="Write configuration, objective units, best placement and runtime as JSON")
    parser.add_argument("--diagnostics", help="Optional JSONL path for per-episode DDPG diagnostics: noisy and deterministic costs, reward, OU noise, collisions, action statistics, and losses")
    parser.add_argument("--diagnostics_every", type=int, default=100,
                        help="Run the costly deterministic DDPG diagnostic every N complete placements")
    parser.add_argument("--asa_diagnostics",
                        help="Optional JSONL path for per-window adaptive-SA diagnostics")
    parser.add_argument("--sa_initial_acceptance", type=float, default=0.8,
                        help="Target uphill acceptance used to measure fixed SA's initial temperature")
    parser.add_argument("--sa_final_temperature_ratio", type=float, default=1e-6,
                        help="Fixed SA ending temperature as a fraction of its measured initial temperature")
    parser.add_argument("--sa_calibration_trials", type=int, default=32,
                        help="Budgeted proposals used to measure fixed SA's objective scale")
    parser.add_argument("--sa_legacy_absolute_temperature", action="store_true",
                        help="Reproduce the historical unit-mismatched fixed SA temperatures 100 to 0.1")
    parser.add_argument("--sa_full_evaluation", action="store_true",
                        help="Disable exact affected-stage incremental evaluation for fixed SA")
    parser.add_argument("--asa_initial_acceptance", type=float, default=0.8,
                        help="Target uphill acceptance used to calibrate ASA's initial temperature")
    parser.add_argument("--asa_target_acceptance", type=float, default=0.30,
                        help="Acceptance feedback target for adaptive temperature control")
    parser.add_argument("--asa_adapt_window", type=int, default=100,
                        help="Candidate evaluations per ASA adaptation window")
    parser.add_argument("--asa_min_perturb_frac", type=float, default=0.005)
    parser.add_argument("--asa_max_perturb_frac", type=float, default=0.05)
    parser.add_argument("--asa_stall_windows", type=int, default=3,
                        help="Non-improving windows before ASA reheats and expands its neighborhood")
    parser.add_argument("--asa_calibration_trials", type=int, default=32,
                        help="Budgeted proposals used to estimate an objective-scaled initial temperature")
    parser.add_argument("--asa_full_evaluation", action="store_true",
                        help="Disable exact affected-stage incremental evaluation (debug/reference mode)")
    parser.add_argument("--sensitivity_trials", type=int, default=64,
                        help="Random placements used to measure objective sensitivity before paper-mode optimization")
    parser.add_argument("--min_relative_objective_span", type=float, default=0.001,
                        help="Minimum sampled (max-min)/median required before a long paper-mode optimization")
    parser.add_argument("--allow_flat_objective", action="store_true",
                        help="Continue despite a failed placement-sensitivity gate")
    parser.add_argument("--vva_ops_per_cycle", type=float, default=1.0,
                         help="Assumed VVA additions/cycle, not specified by paper (default 1)")
    parser.add_argument(
        "--partition_vva_ops_per_cycle", type=float, default=None,
        help=("VVA rate used only when reconstructing unpublished paper-target "
              "M/N grids. Defaults to --vva_ops_per_cycle. Set it explicitly "
              "to hold the logical graph fixed during execution-rate sensitivity tests."))
    parser.add_argument("--on_bandwidth_gbs", type=float, default=64.0)
    parser.add_argument("--off_bandwidth_gbs", type=float, default=100.0)
    parser.add_argument("--mac_utilization", type=float, default=1.0,
                         help="Assumed arithmetic utilization for full_frame timing (0,1].")
    
    # NEW ARGUMENT: Flag to use the real CNN workload
    parser.add_argument("--use_cnn", action="store_true", help="Use real CNN workload instead of random")
    parser.add_argument("--model", type=str, default="simple",
                         help="simple or a torchvision model name. FX tracing is required; "
                              "grouped convolutions and concatenation are unsupported. "
                              "Residual branch traffic is traced; addition uses the "
                              "documented destination-task assumption.")
    parser.add_argument("--custom_model", type=str, default=None,
                         help="Path to a Python file defining build_model() -> model or "
                              "(model, dummy_input), for a user-supplied architecture instead "
                              "of a built-in or torchvision one. See extract_model_task_graph's "
                              "docstring for the exact contract.")
    parser.add_argument("--channels_per_partition", type=int, default=8,
                         help="Max output channels per VMM/VVA partition group when "
                              "--use_cnn is set (paper Sec 3.1.1, Fig 5-6). Smaller = "
                              "more, finer-grained logic cores (closer to the paper's "
                              "per-model core counts in Fig 6). Set to 0 for the old "
                              "one-task-per-layer behavior. Real models (vgg16/resnet50) "
                              "have far more channels than 'simple' -- start with a "
                              "LARGER value (e.g. 32-64) or you'll generate more logic "
                              "cores than any reasonable grid can hold.")
    parser.add_argument("--partition_mode", choices=["uniform", "paper_targets"],
                        default="uniform",
                        help="uniform uses --channels_per_partition; paper_targets "
                             "matches Figure 6 aggregate CONV/FC core counts with "
                             "documented reconstructed per-layer grids")
    parser.add_argument("--partition_balance_weight", type=float, default=1.0,
                        help="Weight given to per-core VMM/VVA cycle balance when "
                             "reconstructing unpublished paper-target layer grids")
    parser.add_argument("--logic_task_order",
                        choices=["output_group_interleaved", "all_vmm_then_vva"],
                        default="output_group_interleaved",
                        help="Logical-core numbering for sequential BS. The paper publishes "
                             "physical ordering but not VMM/VVA indexing; the default keeps "
                             "the historical output-group interleaving.")
    parser.add_argument("--workload_region", choices=["all", "conv", "fc"], default="all",
                        help="Optimize the whole extracted graph or, with paper_targets, "
                             "a separately masked CONV/FC region as in Section 3.1.2")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for random/numpy/torch RNGs. Not fixed by default -- "
                              "the paper itself (Sec 4.5, Fig 20) runs 5 different seeds "
                              "and averages results rather than using one fixed seed, so "
                              "pass this explicitly per-run when you want either a single "
                              "reproducible run or a multi-seed comparison.")
    args = parser.parse_args()
    if (args.ddpg_gamma != .98 or args.ddpg_disable_learning) and args.algo != "ddpg":
        parser.error("DDPG gamma/frozen overrides apply only to --algo ddpg")
    if args.use_cnn and not HAS_TORCH:
        parser.error("--use_cnn requires PyTorch; refusing a synthetic-workload fallback")
    if args.device == "cuda" and (not HAS_TORCH or not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable; install CUDA PyTorch on the GPU host")
    if args.compute_ops:
        parser.error("--compute_ops is disabled: an untyped MAC vector cannot specify VVA work or communication units. Use --timing_model full_frame --use_cnn")
    if args.timing_model != "proxy" and not args.use_cnn:
        parser.error("full-frame timing requires an extracted workload with operation and byte metadata")
    if args.workload_region != "all" and args.partition_mode != "paper_targets":
        parser.error("--workload_region conv/fc requires --partition_mode paper_targets")
    if args.epochs <= 0 or args.placements_per_epoch <= 0 or args.iters <= 0:
        parser.error("--epochs, --placements_per_epoch and --iters must be positive")
    if args.diagnostics_every <= 0 or args.cpu_interop_threads <= 0 or (args.cpu_threads is not None and args.cpu_threads <= 0):
        parser.error("diagnostic and CPU thread counts must be positive")
    if args.asa_adapt_window <= 0 or args.asa_stall_windows <= 0 or args.asa_calibration_trials <= 0:
        parser.error("ASA windows and calibration counts must be positive")
    if args.sa_calibration_trials <= 0:
        parser.error("SA calibration count must be positive")
    if not 0 < args.sa_initial_acceptance < 1:
        parser.error("SA initial acceptance must be in (0,1)")
    if not 0 < args.sa_final_temperature_ratio < 1:
        parser.error("SA final temperature ratio must be in (0,1)")
    if not (0 < args.asa_initial_acceptance < 1 and 0 < args.asa_target_acceptance < 1):
        parser.error("ASA acceptance targets must be in (0,1)")
    if not (0 < args.asa_min_perturb_frac <= args.asa_max_perturb_frac <= 1):
        parser.error("ASA perturbation fractions must satisfy 0 < min <= max <= 1")
    if args.sensitivity_trials < 0 or args.sensitivity_trials == 1:
        parser.error("--sensitivity_trials must be 0 or at least 2")
    if (not math.isfinite(args.min_relative_objective_span) or
            args.min_relative_objective_span < 0):
        parser.error("--min_relative_objective_span must be finite and nonnegative")
    if args.reward_scale is not None and (not math.isfinite(args.reward_scale) or
                                          args.reward_scale <= 0):
        parser.error("--reward_scale must be finite and positive")
    if (args.exploration_decay_placements is not None and
            args.exploration_decay_placements <= 0):
        parser.error("--exploration_decay_placements must be positive")
    if (args.guided_top_k <= 0 or args.guided_demo_iterations <= 0 or
            args.guided_pretrain_updates < 0 or args.guided_bc_decay_placements <= 0):
        parser.error("guided DDPG budgets must be positive (pretrain updates may be zero)")
    if args.guided_random_start_placements < 0:
        parser.error("--guided_random_start_placements must be nonnegative")
    if min(args.ppo_rollout_placements, args.ppo_update_epochs,
           args.ppo_minibatch_size, args.ppo_hidden_dim) <= 0:
        parser.error("masked PPO sizes and update budgets must be positive")
    if args.ppo_asa_candidates < 2 or args.ppo_asa_rollout_steps <= 0:
        parser.error("PPO-ASA requires at least two candidates and a positive rollout")
    if args.proxy_sa_candidates < 2:
        parser.error("--proxy_sa_candidates must be at least 2")
    if not 0 <= args.proxy_sa_focus_fraction <= 1:
        parser.error("--proxy_sa_focus_fraction must be in [0,1]")
    if not 0 <= args.proxy_sa_focus_stage_slack < 1:
        parser.error("--proxy_sa_focus_stage_slack must be in [0,1)")
    if not 0 <= args.proxy_sa_epsilon_final <= 1:
        parser.error("--proxy_sa_epsilon_final must be in [0,1]")
    if args.operator_sa_proxy_candidates < 2:
        parser.error("--operator_sa_proxy_candidates must be at least 2")
    if args.operator_sa_ucb_alpha < 0:
        parser.error("--operator_sa_ucb_alpha must be nonnegative")
    if not 0 < args.operator_sa_discount <= 1:
        parser.error("--operator_sa_discount must be in (0,1]")
    if not 0 <= args.operator_sa_minimum_exploration <= 1:
        parser.error("--operator_sa_minimum_exploration must be in [0,1]")
    if args.operator_sa_phase_bins <= 0:
        parser.error("--operator_sa_phase_bins must be positive")
    if args.operator_sa_full_information_stride < 0:
        parser.error("--operator_sa_full_information_stride cannot be negative")
    if not 1 <= args.operator_sa_proxy_gate_slate <= 4:
        parser.error("--operator_sa_proxy_gate_slate must be in [1,4]")
    if args.operator_sa_thompson_scale < 0:
        parser.error("--operator_sa_thompson_scale must be nonnegative")
    if args.operator_sa_reward_window <= 0:
        parser.error("--operator_sa_reward_window must be positive")
    if args.operator_sa_macro_horizon < 0:
        parser.error("--operator_sa_macro_horizon cannot be negative")
    if (args.operator_sa_macro_horizon and
            args.operator_sa_full_information_stride):
        parser.error("macro actions and full-information labels are exclusive")
    if (args.operator_sa_full_information_stride and
            args.operator_sa_reward != "signed"):
        parser.error("full-information operator labels require signed reward")
    if not 0.0 <= args.ppo_asa_focus_fraction <= 1.0:
        parser.error("--ppo_asa_focus_fraction must be in [0, 1]")
    if not 0.0 <= args.ppo_asa_focus_stage_slack < 1.0:
        parser.error("--ppo_asa_focus_stage_slack must be in [0, 1)")
    if not 1 <= args.ppo_asa_bandit_top_m <= args.ppo_asa_candidates:
        parser.error(
            "--ppo_asa_bandit_top_m must be in [1, ppo_asa_candidates]")
    if not 0.0 <= args.ppo_asa_bandit_global_epsilon <= 1.0:
        parser.error("--ppo_asa_bandit_global_epsilon must be in [0, 1]")
    if args.ppo_asa_restart_interval < 0:
        parser.error("--ppo_asa_restart_interval cannot be negative")
    if (not math.isfinite(args.ppo_learning_rate) or args.ppo_learning_rate <= 0 or
            not valid_ppo_gamma(args.algo, args.ppo_gamma) or
            not 0 <= args.ppo_gae_lambda <= 1 or
            not 0 < args.ppo_clip_ratio < 1 or args.ppo_entropy_coef < 0 or
            args.ppo_value_coef < 0 or args.ppo_max_grad_norm <= 0):
        parser.error("invalid PPO hyperparameters")
    if (args.algo == "ppo_asa" and args.load_checkpoint and
            not args.ppo_asa_disable_learning and
            not args.ppo_asa_finetune_loaded):
        parser.error(
            "PPO-ASA checkpoint learning requires explicit "
            "--ppo_asa_finetune_loaded; use --ppo_asa_disable_learning for "
            "frozen evaluation")
    if args.conv_blocks <= 0:
        parser.error("--conv_blocks must be positive")
    if args.timing_model == "paper_pipeline":
        if (args.partition_mode != "paper_targets" or
                args.workload_region not in ("conv", "fc") or
                args.routing_model != "paper_xy"):
            parser.error("paper_pipeline requires --partition_mode paper_targets, "
                         "--workload_region conv|fc and --routing_model paper_xy")
    if (not 0 < args.mac_utilization <= 1 or
            not math.isfinite(args.vva_ops_per_cycle) or args.vva_ops_per_cycle <= 0):
        parser.error("utilization must be in (0,1] and VVA rate finite and positive")
    if (args.partition_vva_ops_per_cycle is not None and
            (not math.isfinite(args.partition_vva_ops_per_cycle) or
             args.partition_vva_ops_per_cycle <= 0)):
        parser.error("--partition_vva_ops_per_cycle must be finite and positive")
    if not math.isfinite(args.partition_balance_weight) or args.partition_balance_weight < 0:
        parser.error("--partition_balance_weight must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.on_bandwidth_gbs, args.off_bandwidth_gbs)):
        parser.error("bandwidths must be finite and positive")

    if args.reward_scale is None:
        args.reward_scale = 400e6 if args.timing_model != "proxy" else 1.0
    if args.partition_vva_ops_per_cycle is None:
        args.partition_vva_ops_per_cycle = args.vva_ops_per_cycle

    if args.device == "cpu":
        configured = configure_cpu_threads(args.cpu_threads, args.cpu_interop_threads)
        if configured is not None:
            print(f">> CPU mode: {configured} PyTorch intra-op thread(s), "
                  f"{args.cpu_interop_threads} inter-op thread(s)")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if HAS_TORCH:
            torch.manual_seed(args.seed)
        print(f">> Seeded RNGs with --seed {args.seed} (random baseline and "
              f"RL/search sampling are reproducible for this run)\n")

    if args.model != "simple" and args.custom_model is None and not HAS_TORCHVISION:
        print(f"ERROR: --model {args.model} requires torchvision, which isn't installed.")
        print("Install it with: pip install torchvision --break-system-packages")
        sys.exit(1)

    # 1. Determine which workload to use
    real_task_graph = None
    num_tasks = None

    if args.use_cnn and HAS_TORCH:
        print(f">> Extracting {args.model} Workload via torch.fx graph tracing...")
        extracted = extract_model_task_graph(
            model_name=args.model, channels_per_partition=args.channels_per_partition,
            custom_model_path=args.custom_model, return_work=args.timing_model != "proxy",
            partition_mode=args.partition_mode,
            partition_balance_weight=args.partition_balance_weight,
            vva_ops_per_cycle=args.partition_vva_ops_per_cycle,
        )
        real_task_graph, num_tasks, task_labels = extracted[:3]
        operations = extracted[3] if args.timing_model != "proxy" else None
        kinds = extracted[4] if args.timing_model != "proxy" else None
        if args.workload_region != "all":
            wanted = "conv" if args.workload_region == "conv" else "linear"
            selected = np.array([index for index, label in enumerate(task_labels)
                                 if f"_{wanted}_" in label], dtype=np.int64)
            real_task_graph = real_task_graph[np.ix_(selected, selected)]
            task_labels = [task_labels[index] for index in selected]
            if operations is not None:
                operations = operations[selected]
                kinds = [kinds[index] for index in selected]
            num_tasks = len(selected)
        (real_task_graph, task_labels, operations, kinds) = reorder_extracted_tasks(
            real_task_graph, task_labels, operations, kinds, args.logic_task_order)
        if args.logic_task_order != "output_group_interleaved":
            print(f">> Logical-core order: {args.logic_task_order} "
                  "(baseline-sensitivity reconstruction assumption)")
        if args.partition_mode == "paper_targets":
            target = PAPER_LOGIC_CORE_TARGETS[args.model.lower()]
            print(f">> Paper-target partitioning: CONV={target['conv']}, FC={target['linear']} "
                  f"-> {num_tasks} logic cores (VMM+VVA)\n")
        else:
            print(f">> Channel partitioning: channels_per_partition={args.channels_per_partition} "
                  f"-> {num_tasks} logic cores (VMM+VVA)\n" if args.channels_per_partition > 0 else
                  f">> Legacy mode (channels_per_partition=0): {num_tasks} whole-layer tasks\n")

    # Pre-flight check: partitioning can produce far more logic cores than a
    # small default grid has room for. Fail clearly here rather than deep
    # inside placement code with a confusing index error.
    total_cores_requested = args.chips_x * args.chips_y * args.rows * args.cols
    if num_tasks is not None and num_tasks > total_cores_requested:
        print(f"ERROR: {num_tasks} logic cores requested but the grid only has "
              f"{total_cores_requested} cores ({args.chips_x}x{args.chips_y} chips x "
              f"{args.rows}x{args.cols} cores/chip).")
        print("Fix by either:")
        print(f"  1. Increasing grid size, e.g. --chips_x 4 --chips_y 4 --rows 16 --cols 16 "
              f"(paper's 4096-core config)")
        print(f"  2. Increasing --channels_per_partition (fewer, coarser logic cores)")
        sys.exit(1)

    # 2. Build the Environment
    compute_latency = None
    if args.timing_model != "proxy":
        if args.timing_model == "paper_pipeline" and args.workload_region == "conv":
            operations = operations / args.conv_blocks
            real_task_graph = real_task_graph / args.conv_blocks
        compute_latency = compute_seconds(
            operations, kinds, utilization=args.mac_utilization,
            vva_ops_per_cycle=args.vva_ops_per_cycle
        )
        real_task_graph = edge_bytes(real_task_graph, kinds)
        args.on_lat = 1 / (args.on_bandwidth_gbs * 1e9)
        args.off_lat = 1 / (args.off_bandwidth_gbs * 1e9)
        if args.timing_model == "paper_pipeline":
            print(f">> Paper-pipeline reconstruction: {args.conv_blocks if args.workload_region == 'conv' else 1} "
                  "block(s), XY routing and shared-link contention; GRS gateway/block count are assumptions")
        else:
            print(">> Full-frame approximation in seconds: arithmetic + byte-hop serialization; no contention or streaming schedule")
    else:
        print(">> Compute latency disabled (communication-only objective)")

    allowed_cores = None
    if args.workload_region != "all":
        targets = PAPER_LOGIC_CORE_TARGETS[args.model.lower()]
        cores_per_chip = args.rows * args.cols
        conv_chips = math.ceil(targets["conv"] / cores_per_chip)
        region_chips = (conv_chips if args.workload_region == "conv"
                        else math.ceil(targets["linear"] / cores_per_chip))
        region_width = min(args.chips_x, region_chips)
        region_height = math.ceil(region_chips / region_width)
        start_row = 0 if args.workload_region == "conv" else math.ceil(conv_chips / args.chips_x)
        if start_row + region_height > args.chips_y:
            parser.error("paper region reconstruction does not fit the configured chip array")
        chip_ids = []
        for row in range(start_row, start_row + region_height):
            for col in range(region_width):
                if len(chip_ids) < region_chips:
                    chip_ids.append(row * args.chips_x + col)
        allowed_cores = np.concatenate([
            np.arange(chip * cores_per_chip, (chip + 1) * cores_per_chip,
                      dtype=np.int32) for chip in chip_ids
        ])
        print(f">> Masked {args.workload_region.upper()} region: chips {chip_ids}, "
              f"{len(allowed_cores)} physical cores "
              f"for {num_tasks} logic cores (minimal whole-chip reconstruction)")

    env = MultiChipEnvironment(
        num_chips_x=args.chips_x, num_chips_y=args.chips_y,
        rows_per_chip=args.rows, cols_per_chip=args.cols,
        on_chip_latency=args.on_lat, off_chip_latency=args.off_lat,
        topology=args.topology,
        routing_model=args.routing_model,
        task_graph=real_task_graph,  # Passes the CNN graph here (or None for random)
        num_tasks=num_tasks,
        compute_latency=compute_latency,
        timing_units="seconds" if args.timing_model != "proxy" else "proxy",
        allowed_cores=allowed_cores,
        pipeline_model="xy_contention" if args.timing_model == "paper_pipeline" else "task_sum",
    )
    
    print(f"System : {env.topo}")
    print(f"Tasks  : {env.num_tasks}")
    print(f"Algo   : {args.algo}")
    if args.algo in ("ddpg", "ddpg_guided", "ddpg_asa", "ppo_masked"):
        print(f"Reward : sqrt({args.reward_scale:g} * latency)")
    print("-" * 50)

    objective_sensitivity = None
    if args.timing_model == "paper_pipeline" and args.sensitivity_trials:
        objective_sensitivity = measure_objective_sensitivity(
            env, n_trials=args.sensitivity_trials)
        sample = objective_sensitivity
        split = sample["best_sample_breakdown"]
        print(
            f">> Objective sensitivity ({sample['trials']} random placements): "
            f"min={sample['minimum']:.9g}, median={sample['median']:.9g}, "
            f"max={sample['maximum']:.9g}, relative span={sample['relative_span']:.3%}")
        print(
            f">> Best sampled bottleneck: compute={split['compute']:.9g}, "
            f"communication={split['communication']:.9g}, "
            f"compute fraction={split['compute_fraction']:.3%}, "
            f"headroom above compute floor="
            f"{sample['best_headroom_above_compute_floor']:.3%}")
        if (args.algo != "bs" and
                sample["relative_span"] < args.min_relative_objective_span and
                not args.allow_flat_objective):
            parser.error(
                "paper-mode objective is effectively placement-insensitive: "
                f"sampled relative span {sample['relative_span']:.6g} is below "
                f"{args.min_relative_objective_span:.6g}. Fix/validate timing "
                "assumptions or pass --allow_flat_objective for a deliberate diagnostic run")

    # 3. Run the algorithms
    run_started = time.perf_counter()
    complete_placements = args.epochs * args.placements_per_epoch
    algorithm_metadata = None
    ddpg_options = {
        "batch_z": args.batch_z,
        "baseline_trials": args.baseline_trials,
        "train_every": args.train_every,
        "device": args.device,
        "save_checkpoint": args.save_checkpoint,
        "load_checkpoint": args.load_checkpoint,
        "checkpoint_every": args.checkpoint_every,
        "diagnostics_path": args.diagnostics,
        "agent_arch": args.agent_arch,
        "reward_mode": args.reward_mode,
        "reward_scale": args.reward_scale,
        "exploration_decay_placements": args.exploration_decay_placements,
        "retain_deterministic_candidates": args.retain_deterministic_candidates,
        "diagnostics_every": args.diagnostics_every,
    }
    asa_options = {
        "initial_acceptance": args.asa_initial_acceptance,
        "target_acceptance": args.asa_target_acceptance,
        "adapt_window": args.asa_adapt_window,
        "min_perturb_frac": args.asa_min_perturb_frac,
        "max_perturb_frac": args.asa_max_perturb_frac,
        "stall_windows": args.asa_stall_windows,
        "calibration_trials": args.asa_calibration_trials,
        "diagnostics_path": args.asa_diagnostics,
        "incremental": not args.asa_full_evaluation,
    }
    if args.algo == "ddpg":
        ddpg_metadata = {}
        print(f">> DDPG budget: {args.epochs} epoch(s) x "
              f"{args.placements_per_epoch} placement(s) = "
              f"{complete_placements} complete placements")
        cost = run_ddpg(env, n_episodes=complete_placements,
                        run_metadata=ddpg_metadata, ddpg_gamma=args.ddpg_gamma,
                        ddpg_disable_learning=args.ddpg_disable_learning, **ddpg_options)
        ddpg_metadata.update({"declared_epochs": args.epochs,
                              "placements_per_epoch": args.placements_per_epoch,
                              "complete_placement_evaluations": complete_placements})
        algorithm_metadata = ddpg_metadata
    elif args.algo == "ddpg_guided":
        algorithm_metadata = {}
        print(f">> Guided DDPG budget: {complete_placements} online placements + "
              f"{args.guided_demo_iterations} ASA demonstration candidates")
        cost = run_ddpg_guided(
            env, n_episodes=complete_placements, batch_z=args.batch_z,
            baseline_trials=args.baseline_trials, train_every=args.train_every,
            device=args.device, save_checkpoint=args.save_checkpoint,
            load_checkpoint=args.load_checkpoint,
            checkpoint_every=args.checkpoint_every,
            diagnostics_path=args.diagnostics,
            run_metadata=algorithm_metadata, agent_arch=args.agent_arch,
            reward_scale=args.reward_scale,
            exploration_decay_placements=args.exploration_decay_placements,
            diagnostics_every=args.diagnostics_every,
            top_k=args.guided_top_k,
            demo_iterations=args.guided_demo_iterations,
            pretrain_updates=args.guided_pretrain_updates,
            bc_decay_placements=args.guided_bc_decay_placements,
            retain_deterministic_candidates=True,
            asa_options=asa_options,
            random_start_placements=args.guided_random_start_placements,
            disable_learning=args.guided_disable_learning,
            seed=args.seed)
        algorithm_metadata.update({"declared_epochs": args.epochs,
                                   "placements_per_epoch": args.placements_per_epoch,
                                   "complete_placement_evaluations": complete_placements})
    elif args.algo == "ppo_masked":
        algorithm_metadata = {}
        print(f">> Masked PPO budget: {args.epochs} epoch(s) x "
              f"{args.placements_per_epoch} placement(s) = "
              f"{complete_placements} complete placements")
        cost = run_masked_ppo(
            env, n_episodes=complete_placements,
            baseline_trials=args.baseline_trials, device=args.device,
            learning_rate=args.ppo_learning_rate,
            rollout_episodes=args.ppo_rollout_placements,
            update_epochs=args.ppo_update_epochs,
            minibatch_size=args.ppo_minibatch_size,
            gamma=args.ppo_gamma, gae_lambda=args.ppo_gae_lambda,
            clip_ratio=args.ppo_clip_ratio,
            entropy_coefficient=args.ppo_entropy_coef,
            value_coefficient=args.ppo_value_coef,
            max_grad_norm=args.ppo_max_grad_norm,
            hidden_dim=args.ppo_hidden_dim, reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
            diagnostics_every=args.diagnostics_every,
            diagnostics_path=args.diagnostics,
            save_checkpoint=args.save_checkpoint,
            load_checkpoint=args.load_checkpoint,
            checkpoint_every=args.checkpoint_every,
            disable_learning=args.ppo_disable_learning,
            seed=args.seed, run_metadata=algorithm_metadata)
        algorithm_metadata.update({"declared_epochs": args.epochs,
                                   "placements_per_epoch": args.placements_per_epoch,
                                   "complete_placement_evaluations": complete_placements})
    elif args.algo == "ppo_asa":
        algorithm_metadata = {}
        print(f">> PPO-ASA budget: one initialization + {args.iters} "
              f"evaluated proposals; {args.ppo_asa_candidates} legal candidates/step")
        cost = run_ppo_asa(
            env, n_iter=args.iters,
            candidate_count=args.ppo_asa_candidates,
            rollout_steps=args.ppo_asa_rollout_steps,
            device=args.device, learning_rate=args.ppo_learning_rate,
            weight_decay=args.ppo_asa_weight_decay,
            update_epochs=args.ppo_update_epochs,
            minibatch_size=args.ppo_minibatch_size,
            gamma=args.ppo_gamma, gae_lambda=args.ppo_gae_lambda,
            clip_ratio=args.ppo_clip_ratio,
            entropy_coefficient=args.ppo_entropy_coef,
            value_coefficient=args.ppo_value_coef,
            max_grad_norm=args.ppo_max_grad_norm,
            hidden_dim=args.ppo_hidden_dim,
            disable_learning=args.ppo_asa_disable_learning,
            proposal_learner=args.ppo_asa_learner,
            bandit_control=args.ppo_asa_bandit_control,
            bandit_replay_capacity=args.ppo_asa_bandit_replay,
            bandit_warmup=args.ppo_asa_bandit_warmup,
            bandit_recent_fraction=args.ppo_asa_bandit_recent_fraction,
            bandit_recent_window=args.ppo_asa_bandit_recent_window,
            bandit_epsilon_final=args.ppo_asa_bandit_epsilon_final,
            bandit_epsilon_decay_fraction=args.ppo_asa_bandit_epsilon_decay,
            bandit_exploration=args.ppo_asa_bandit_exploration,
            bandit_top_m=args.ppo_asa_bandit_top_m,
            bandit_global_epsilon=args.ppo_asa_bandit_global_epsilon,
            bandit_feature_set=args.ppo_asa_bandit_feature_set,
            bandit_shuffle_labels=args.ppo_asa_bandit_shuffle_labels,
            bandit_scoring=args.ppo_asa_bandit_scoring,
            bandit_residual_limit=args.ppo_asa_bandit_residual_limit,
            state_encoding=args.ppo_asa_state_encoding,
            bandit_ranking_weight=args.ppo_asa_bandit_ranking_weight,
            bandit_pool_supervision_stride=(
                args.ppo_asa_bandit_pool_supervision_stride),
            update_style=args.ppo_asa_update_style,
            reward_mode=args.ppo_asa_reward_mode,
            focus_bottleneck=args.ppo_asa_focus_bottleneck,
            focus_fraction=args.ppo_asa_focus_fraction,
            focus_stage_slack=args.ppo_asa_focus_stage_slack,
            pool_focus_fraction=args.ppo_asa_pool_focus_fraction,
            pool_small_move_fraction=args.ppo_asa_pool_small_move_fraction,
            restart_interval=args.ppo_asa_restart_interval,
            update_at_chain_end=args.ppo_asa_update_at_chain_end,
            seed=args.seed,
            initial_acceptance=args.asa_initial_acceptance,
            target_acceptance=args.asa_target_acceptance,
            adapt_window=args.asa_adapt_window,
            min_perturb_frac=args.asa_min_perturb_frac,
            max_perturb_frac=args.asa_max_perturb_frac,
            stall_windows=args.asa_stall_windows,
            calibration_trials=args.asa_calibration_trials,
            annealing_schedule=args.ppo_asa_annealing,
            fixed_cooldown=0.99,
            fixed_perturb_fraction=0.01,
            final_temperature_ratio=args.sa_final_temperature_ratio,
            diagnostics_path=(args.asa_diagnostics or args.diagnostics),
            metadata=algorithm_metadata,
            incremental=not args.asa_full_evaluation,
            save_checkpoint=args.save_checkpoint,
            load_checkpoint=args.load_checkpoint,
            checkpoint_every=args.checkpoint_every,
            oracle_diagnostics_path=args.ppo_asa_oracle_diagnostics,
            oracle_pools=args.ppo_asa_oracle_pools,
            oracle_stride=args.ppo_asa_oracle_stride,
            oracle_dataset_path=args.ppo_asa_oracle_dataset,
            allow_finetune_loaded=args.ppo_asa_finetune_loaded)
    elif args.algo == "proxy_sa":
        algorithm_metadata = {}
        print(f">> Proxy-SA budget: one initialization + {args.iters} "
              f"evaluated proposals; {args.proxy_sa_candidates} legal candidates/step")
        cost = run_proxy_guided_sa(
            env, n_iter=args.iters,
            candidate_count=args.proxy_sa_candidates,
            focus_fraction=args.proxy_sa_focus_fraction,
            focus_stage_slack=args.proxy_sa_focus_stage_slack,
            epsilon_final=args.proxy_sa_epsilon_final,
            seed=args.seed, metadata=algorithm_metadata)
    elif args.algo == "operator_sa":
        algorithm_metadata = {}
        print(f">> Operator-bandit SA budget: one initialization + {args.iters} "
              "evaluated proposals")
        cost = run_operator_bandit_sa(
            env, n_iter=args.iters,
            proxy_candidates=args.operator_sa_proxy_candidates,
            ucb_alpha=args.operator_sa_ucb_alpha,
            discount=args.operator_sa_discount,
            minimum_exploration=args.operator_sa_minimum_exploration,
            phase_bins=args.operator_sa_phase_bins,
            full_information_stride=args.operator_sa_full_information_stride,
            selection_policy=args.operator_sa_selection_policy,
            proxy_gate_slate=args.operator_sa_proxy_gate_slate,
            thompson_scale=args.operator_sa_thompson_scale,
            reward_mode=args.operator_sa_reward,
            reward_window=args.operator_sa_reward_window,
            macro_horizon=args.operator_sa_macro_horizon,
            control=args.operator_sa_control,
            disable_learning=args.operator_sa_disable_learning,
            seed=args.seed, metadata=algorithm_metadata)
    elif args.algo == "ddpg_asa":
        algorithm_metadata = {}
        print(f">> Hybrid budget: {complete_placements} DDPG placements + "
              f"{args.iters} adaptive-SA candidates")
        cost = run_ddpg_asa(
            env, ddpg_placements=complete_placements,
            asa_iterations=args.iters, ddpg_options=ddpg_options,
            asa_options=asa_options, metadata=algorithm_metadata)
    elif args.algo == "asa":
        algorithm_metadata = {}
        cost = run_adaptive_sa(env, n_iter=args.iters,
                               metadata=algorithm_metadata, **asa_options)
    elif args.algo == "sa":
        algorithm_metadata = {}
        cost = run_sa(
            env, n_iter=args.iters,
            T_start=(100.0 if args.sa_legacy_absolute_temperature else None),
            T_end=(0.1 if args.sa_legacy_absolute_temperature else None),
            initial_acceptance=args.sa_initial_acceptance,
            final_temperature_ratio=args.sa_final_temperature_ratio,
            calibration_trials=args.sa_calibration_trials,
            metadata=algorithm_metadata,
            incremental=not args.sa_full_evaluation)
    elif args.algo == "bs":
        cost = run_sequential(env)
    else:
        cost = run_random(env, n_trials=args.iters)

    bd = env.chip_breakdown()
    routing_diagnostics = env.routing_diagnostics()
    print(f"\nFinal placement cost : {cost:.6g}")
    print(f"  On-chip  comm cost : {bd['on_chip_cost']:.6g}")
    print(f"  Off-chip comm cost : {bd['off_chip_cost']:.6g}")
    print(f"  Chip placement     : {env.placement[:env.num_tasks]}")
    if args.report:
        report_parent = os.path.dirname(args.report)
        if report_parent:
            os.makedirs(report_parent, exist_ok=True)
        limitations = []
        if args.algo in ("ddpg", "ddpg_guided", "ddpg_asa"):
            limitations.append("replay buffer is not persisted in DDPG checkpoints")
        if args.agent_arch != "paper_cnn" and args.algo in ("ddpg", "ddpg_guided", "ddpg_asa"):
            limitations.append("selected DDPG agent is not the Figure-9 paper CNN")
        if args.algo == "ddpg_guided":
            limitations.append("ddpg_guided is a sample-efficiency extension, not the paper's original DDPG")
        if args.algo == "ppo_masked":
            limitations.append("ppo_masked is an experimental discrete policy, not the paper's original DDPG")
        if args.algo == "ppo_asa":
            limitations.append("ppo_asa is an experimental learned-neighbor extension, not the paper's original DDPG")
        if args.algo == "proxy_sa":
            limitations.append(
                "proxy_sa is a training-free communication-proxy-guided fixed-SA "
                "extension, not the paper's original SA or DDPG")
        if args.algo == "operator_sa":
            limitations.append(
                "operator_sa is an online contextual-bandit move-selector "
                "extension, not the paper's original DDPG")
        if args.algo == "sa":
            limitations.append(
                "the paper does not publish application-specific T0/Tend; "
                "fixed SA measures T0 in objective units and freezes the schedule"
                if not args.sa_legacy_absolute_temperature else
                "legacy fixed-SA absolute temperatures are retained only for historical reproduction")
        if args.partition_mode != "paper_targets":
            limitations.append("logic-core allocation does not match Figure-6 aggregate counts")
        else:
            limitations.append("per-layer M/N grids are reconstructed because Figure 6 publishes only aggregate counts")
        if args.logic_task_order != "output_group_interleaved":
            limitations.append("logical VMM/VVA index order is reconstructed for BS sensitivity")
        if args.workload_region == "all":
            limitations.append("CONV and FC placements are not optimized in separate masked regions")
        else:
            limitations.append("masked region uses a minimal rectangular whole-chip reconstruction")
        if args.timing_model != "paper_pipeline":
            limitations.append("timing omits the reconstructed block schedule and shared-link contention")
        else:
            limitations.extend([
                "workload-specific CONV block count is unpublished and configurable",
                "GRS is approximated with a lower-left chip-periphery gateway and XY routing",
                "64-KB input/activation-buffer stalls and compute/communication overlap remain unmodeled",
            ])
        if args.model.lower() == "resnet50":
            limitations.append("residual add is assigned to the destination transformation/VVA path")
        evaluated_placements = (complete_placements if args.algo == "ddpg"
                                else algorithm_metadata["total_candidate_evaluations"]
                                if args.algo in ("ddpg_guided", "ppo_masked")
                                else algorithm_metadata["total_objective_evaluations"]
                                if args.algo in ("ppo_asa", "proxy_sa", "operator_sa", "sa", "asa")
                                else complete_placements + args.iters
                                if args.algo == "ddpg_asa"
                                else 1 if args.algo == "bs" else args.iters)
        with open(args.report, "w") as stream:
            json.dump({"config": vars(args), "algorithm": args.algo,
                       "tasks": env.num_tasks,
                       "allowed_physical_cores": len(env.allowed_cores),
                       "objective_units": env.timing_units, "best_cost": cost,
                       "complete_placement_evaluations": evaluated_placements,
                       "reward_normalizer_evaluations": args.baseline_trials
                       if args.algo in ("ddpg", "ddpg_guided", "ddpg_asa", "ppo_masked") else 0,
                       "placement_chip_major": env.placement.tolist(),
                       "routing_diagnostics": routing_diagnostics,
                       "objective_sensitivity": objective_sensitivity,
                       "seconds_elapsed": time.perf_counter() - run_started,
                       "python_version": sys.version,
                       "git": git_provenance(),
                       "torch_version": torch.__version__ if HAS_TORCH else None,
                       "torch_cuda_build": torch.version.cuda if HAS_TORCH else None,
                       "cuda_available": HAS_TORCH and torch.cuda.is_available(),
                       "cuda_device": torch.cuda.get_device_name(0)
                       if HAS_TORCH and torch.cuda.is_available() else None,
                       "algorithm_metadata": algorithm_metadata,
                       "ddpg_metadata": algorithm_metadata if args.algo in ("ddpg", "ddpg_guided") else
                       algorithm_metadata.get("ddpg") if args.algo == "ddpg_asa" else None,
                       "limitations": limitations},
                      stream, indent=2)

if __name__ == "__main__":
    main()
