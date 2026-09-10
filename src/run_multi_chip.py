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
from datetime import timedelta
import numpy as np

# Make sure the original src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_chip_topology import MultiChipTopology
from multi_chip_environment import MultiChipEnvironment
from compute_model import channel_ranges, tile_work, compute_seconds, edge_bytes

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
    # PERF FIX: on CPU-only machines (no CUDA), PyTorch sometimes defaults
    # to a single thread depending on the environment, silently leaving
    # most cores idle. Explicitly use all available cores for the CPU
    # matmul-heavy actor/critic forward/backward passes.
    if not torch.cuda.is_available():
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1))))
except ImportError:
    HAS_TORCH = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


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
        if any(k in model_name for k in _CONCAT_MERGE_FAMILIES + _ADD_MERGE_FAMILIES):
            warning = ("Branch dependencies are traced, but merge arithmetic is not modeled. "
                       "Full-frame timing rejects residual/concat graphs; proxy scores are approximate.")
        return model, dummy_input, warning

    def extract_model_task_graph(model_name: str = "simple", channels_per_partition: int = 8,
                                  custom_model_path: str = None, return_work: bool = False):
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
        are routed through pooling/flatten using consumer shapes. Residual adds
        are bypassed for proxy graph inspection, not modeled as arithmetic tasks.
        Full-frame timing rejects residual/concat graphs; grouped convolutions
        and channel-changing merges are rejected. This is a restricted extractor,
        not an exact simulator for every torchvision architecture.

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
            if any(n.op == "call_function" and n.target in (operator.add, torch.add, torch.cat)
                   or n.op == "call_method" and n.target in ("add", "add_")
                   for n in traced.graph.nodes):
                raise ValueError("Full-frame timing does not yet model residual add/concat work; use proxy mode for graph inspection")

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

        print(f">> Per-layer VMM/VVA breakdown ({model_name}, channels_per_partition={channels_per_partition}, "
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

    class DDPGAgent:
        def __init__(self, state_dim, action_dim=2, lr_actor=2e-4, lr_critic=1e-3,
                     gamma=0.98, tau=0.005, device=None):
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

            self.action_dim = action_dim
            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.actor_target = Actor(state_dim, action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

            self.critic = Critic(state_dim, action_dim).to(self.device)
            self.critic_target = Critic(state_dim, action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

            self.gamma = gamma
            self.tau = tau
            self.ou_state = np.zeros(action_dim, dtype=np.float32)

        def select_action(self, state, noise_scale=0.1):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state).squeeze(0).cpu().numpy()
            self.actor.train()
            # Add exploration noise
            # Ornstein-Uhlenbeck exploration used by the paper (Sec. 3.2).
            self.ou_state += 0.15 * (0.0 - self.ou_state) + 0.2 * np.random.randn(self.action_dim)
            action += noise_scale * self.ou_state
            return np.clip(action, -1.0, 1.0)

        def train(self, replay_buffer, batch_size=64):
            if len(replay_buffer) < batch_size:
                return

            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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

            # Actor Update
            actor_loss = -self.critic(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft Update Targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        def state_dict(self) -> dict:
            """Everything needed to exactly resume this agent's networks/optimizers."""
            return {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            }

        def load_state_dict(self, sd: dict) -> None:
            self.actor.load_state_dict(sd["actor"])
            self.actor_target.load_state_dict(sd["actor_target"])
            self.critic.load_state_dict(sd["critic"])
            self.critic_target.load_state_dict(sd["critic_target"])
            self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
            self.critic_optimizer.load_state_dict(sd["critic_optimizer"])


# ---------------------------------------------------------------------------
# MultiChipCoreMapper (Translates continuous actions to discrete grid)
# ---------------------------------------------------------------------------

class MultiChipCoreMapper:
    def __init__(self, env: MultiChipEnvironment, baseline_latency: float = None, batch_z: int = 3):
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
        self._task_ptr  = 0
        self.baseline_latency = baseline_latency
        if baseline_latency is None:
            print("[WARN] MultiChipCoreMapper created without baseline_latency -- "
                  "reward will use unnormalized -sqrt(L(P)), not the paper's "
                  "sqrt(B) - sqrt(L(P)) (Algorithm 1, line 11).")

    def reset(self):
        self._placement[:] = -1
        self._occupied.clear()
        self._task_ptr = 0
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
        m = np.zeros(self.total_rows * self.total_cols, dtype=np.float32)
        valid = self._placement >= 0
        if np.any(valid):
            m[self._placement[valid]] = (np.nonzero(valid)[0] + 1) / self.num_tasks

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

        if intended_core not in self._occupied:
            core_id = intended_core
        else:
            total = self.total_rows * self.total_cols
            occupied_mask = np.zeros(total, dtype=bool)
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
            target_x = ((ax + 1.0) / 2.0) * (self.total_cols - 1)
            target_y = ((ay + 1.0) / 2.0) * (self.total_rows - 1)
            self._place_one(target_x, target_y)
            self._task_ptr += 1

        done = (self._task_ptr >= self.num_tasks)

        # PAPER IMPLEMENTATION: sparse reward, r_t = 0 for every non-terminal
        # step (Algorithm 1, line 18); only a completed placement gets a
        # reward, r_t = sqrt(B) - sqrt(L(P)) (Algorithm 1, line 11), where B
        # is the fixed random-search baseline and L(P) is the pipeline
        # bottleneck latency (env.evaluate(), Eq. 4).
        if not done:
            return 0.0, done, "", 0.0

        self.env.place(self.get_placement())
        final_cost = self.env.evaluate()
        grid = self._render()

        if self.baseline_latency is not None:
            step_reward = math.sqrt(max(self.baseline_latency, 0.0)) - math.sqrt(max(final_cost, 0.0))
        else:
            step_reward = -math.sqrt(max(final_cost, 0.0))

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
# DDPG RL Training Loop (Based on Paper Pseudocode)
# ---------------------------------------------------------------------------

def run_ddpg(env: MultiChipEnvironment, n_episodes: int = 500, batch_size: int = 64,
             baseline_trials: int = 1000, batch_z: int = 3, train_every: int = 5,
             device: str = None, save_checkpoint: str = None, load_checkpoint: str = None,
             checkpoint_every: int = 100) -> float:
    if not HAS_TORCH:
        raise RuntimeError("DDPG requires PyTorch; refusing a random-search fallback")

    signature = hashlib.sha256()
    signature.update(np.ascontiguousarray(env.task_graph).tobytes())
    signature.update(np.ascontiguousarray(env.compute_latency).tobytes())
    signature.update(repr((env.topo, env.topo.on_chip_latency, env.topo.off_chip_latency,
                           batch_z, env.timing_units, "chip-major-v2-ou")).encode())
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

    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    print(f"[DDPG] Using device: {agent.device}"
          + (f" ({torch.get_num_threads()} CPU threads)" if agent.device.type == "cpu" else ""))
    replay_buffer = ReplayBuffer()
    mapper = MultiChipCoreMapper(env, baseline_latency=baseline_latency, batch_z=batch_z)

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
        start_episode = checkpoint["episode"]
    else:
        best_cost = baseline_latency
        best_placement = baseline_placement.copy()
        best_grid = None
        noise_scale = 1.0
        global_step = 0
        start_episode = 0

    # Decay so noise_scale reaches ~0.01 by 80% of training, regardless of
    # n_episodes -- a fixed 0.995 barely decays (~8% remaining) over a
    # 500-1000 episode run, which was masking whether the policy had
    # actually converged versus still being exploration-noise-dominated.
    target_episode = max(1, int(0.8 * n_episodes))
    noise_decay = 0.01 ** (1.0 / target_episode)

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
            "episode": ep,
            "baseline_latency": baseline_latency,
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

        while not done:
            action = agent.select_action(state, noise_scale=noise_scale)
            reward, done, grid, final_cost = mapper.step(action)
            next_state = mapper._occ_map()

            replay_buffer.add(state, action, reward, next_state, done)
            global_step += 1
            if global_step % train_every == 0:
                agent.train(replay_buffer, batch_size)
            state = next_state

        noise_scale = max(0.01, noise_scale * noise_decay)

        if final_cost < best_cost:
            best_cost = final_cost          
            best_grid   = grid
            best_placement = mapper.get_placement()

        if ep % 10 == 0 or ep == n_episodes:
            elapsed = time.time() - start_time
            per_ep = elapsed / max(1, ep - start_episode)
            eta_sec = per_ep * (n_episodes - ep)
            eta_str = str(timedelta(seconds=int(eta_sec)))
            print(f"# of epochs: {ep:4d} | Current Cost: {final_cost:.6g} | "
                  f"Best Cost: {best_cost:.6g} | {per_ep:.2f}s/ep | ETA: {eta_str}")

        if save_checkpoint is not None and ep % checkpoint_every == 0:
            _save_checkpoint(ep)
            print(f"[DDPG] Checkpoint saved to {save_checkpoint} (episode {ep})")

    if save_checkpoint is not None and n_episodes > start_episode:
        _save_checkpoint(n_episodes)
        print(f"[DDPG] Final checkpoint saved to {save_checkpoint}")

    if best_grid:
        print("--- Current Best Layout ---")
        print(best_grid)
        print("---------------------------")

    if best_placement is not None:
        env.place(best_placement)
        
    return best_cost


# ---------------------------------------------------------------------------
# Simulated Annealing & Random Baseline
# ---------------------------------------------------------------------------

def run_sa(env: MultiChipEnvironment, n_iter: int = 100000,
           T_start: float = 100.0, T_end: float = 0.1, cooldown: float = 0.99,
           perturb_frac: float = 0.01) -> float:
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

    NOTE: `n_iter=100000` here is still far short of the paper's ~1,000,000
    -- raise it explicitly via the CLI if you want closer to paper-scale
    search; a printed warning fires below this threshold so it's never
    silently unclear which regime a given run used.
    """
    n, k = env.num_tasks, env.total_cores
    if n_iter <= 0 or not (0 < T_end < T_start) or not (0 < cooldown < 1):
        raise ValueError("SA requires positive budget, T_start > T_end > 0, and cooldown in (0,1)")
    if n_iter < 1_000_000:
        print(f"[SA] WARNING: n_iter={n_iter} is far below the paper's ~1,000,000-placement "
              f"search budget (Sec 4.2). Results are not directly comparable to the paper's "
              f"reported SA baseline until run at closer to that scale.")

    placement = np.array(random.sample(range(k), min(n, k)), dtype=np.int32)
    env.place(placement)
    best_cost = env.evaluate()
    best_p = placement.copy()
    cur_cost = best_cost

    num_temp_steps = max(1, math.ceil(math.log(T_end / T_start) / math.log(cooldown)))
    iters_per_temp = max(1, math.ceil(n_iter / num_temp_steps))
    n_perturb = max(1, min(n, max(2, round(perturb_frac * n))))

    T = T_start
    total_done = 0
    while T > T_end and total_done < n_iter:
        for _ in range(iters_per_temp):
            # Paper's neighborhood: randomly perturb ~1% of logic cores.
            # Implemented as a random permutation of a random subset's
            # currently-assigned cores, which changes exactly that subset's
            # task->core assignments while keeping the placement a valid
            # bijection (no core assigned to two tasks).
            idx = random.sample(range(n), min(n_perturb, n))
            cores_subset = list(placement[idx])
            perm = cores_subset.copy()
            random.shuffle(perm)
            # BUG FIX: shuffling a 1-element subset is always a no-op, and
            # even for 2+ elements shuffle can land back on the identity --
            # either way this would silently make the "neighbor" identical
            # to the current placement, wasting that trial. Force an actual
            # change by swapping the first two entries whenever the
            # perturbation didn't change anything and there's more than one
            # task in the subset.
            if perm == cores_subset and len(perm) >= 2:
                perm[0], perm[1] = perm[1], perm[0]
            new_p = placement.copy()
            new_p[idx] = perm
            # Swaps alone cannot explore unused physical cores. Include a
            # relocation proposal when spare cores exist (explicit interpretation).
            free = np.setdiff1d(np.arange(k), placement)
            if len(free) and (n == 1 or random.random() < 0.5):
                new_p[idx[0]] = random.choice(free)

            env.place(new_p)
            new_cost = env.evaluate()

            if new_cost < cur_cost or random.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-9)):
                placement = new_p
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost, best_p = new_cost, new_p.copy()

            total_done += 1
            if total_done >= n_iter:
                break
        T *= cooldown

    env.place(best_p)
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
    n, k = env.num_tasks, env.total_cores
    best_cost, best_p = float("inf"), None
    for _ in range(n_trials):
        p = np.array(random.sample(range(k), min(n, k)), dtype=np.int32)
        env.place(p); cost = env.evaluate()
        if cost < best_cost: best_cost, best_p = cost, p.copy()
    env.place(best_p)
    return best_cost


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-chip core placement")
    parser.add_argument("--algo", choices=["ddpg", "sa", "random"], default="ddpg")
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
    parser.add_argument("--iters", type=int, default=5000, help="SA/random iterations")
    parser.add_argument("--epochs", type=int, default=1000, help="DDPG training epochs")
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
                         help="Path to save a DDPG checkpoint (networks, optimizer state, "
                              "best result, noise schedule position, baseline B) every "
                              "--checkpoint_every episodes and once more at the end. "
                              "Without this, nothing persists between runs -- each run "
                              "starts from scratch.")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                         help="Path to resume training from a previously saved checkpoint. "
                              "If the file doesn't exist yet, training starts fresh and "
                              "will create it (combine with --save_checkpoint pointing to "
                              "the same path to make a run resumable from itself). Note: "
                              "the replay buffer is NOT persisted -- resumed training keeps "
                              "the trained networks and progress, but refills experience "
                              "from an empty buffer.")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                         help="Save a checkpoint every N episodes (only used with "
                              "--save_checkpoint).")
    parser.add_argument("--train_every", type=int, default=5,
                         help="Run one DDPG gradient update every N environment "
                              "steps instead of every step. Experience is still "
                              "recorded every step via the replay buffer -- this "
                              "only spaces out the (expensive) actor/critic "
                              "forward+backward passes. Set to 1 to train on "
                              "every step (original behavior, much slower at "
                              "large task counts).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Force a specific device for DDPG. Default: auto-detect "
                              "CUDA if available, else CPU.")
    parser.add_argument("--compute_ops", type=str, default=None,
                         help="Optional .npy file containing MAC operations per task; "
                              "converted with Table 1's 128 MACs/core at 400 MHz.")
    parser.add_argument("--timing_model", choices=["proxy", "full_frame"], default="proxy",
                         help="full_frame: compute + byte-hop serialization in seconds; approximate, no streaming/contention")
    parser.add_argument("--report", help="Write configuration, objective units, best placement and runtime as JSON")
    parser.add_argument("--vva_ops_per_cycle", type=float, default=1.0,
                         help="Assumed VVA additions/cycle, not specified by paper (default 1)")
    parser.add_argument("--on_bandwidth_gbs", type=float, default=64.0)
    parser.add_argument("--off_bandwidth_gbs", type=float, default=100.0)
    parser.add_argument("--mac_utilization", type=float, default=1.0,
                         help="Assumed arithmetic utilization for full_frame timing (0,1].")
    
    # NEW ARGUMENT: Flag to use the real CNN workload
    parser.add_argument("--use_cnn", action="store_true", help="Use real CNN workload instead of random")
    parser.add_argument("--model", type=str, default="simple",
                         help="simple or a torchvision model name. FX tracing is required; "
                              "grouped convolutions and some merges are unsupported. "
                              "Residual/concat timing is rejected in full_frame mode.")
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
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for random/numpy/torch RNGs. Not fixed by default -- "
                              "the paper itself (Sec 4.5, Fig 20) runs 5 different seeds "
                              "and averages results rather than using one fixed seed, so "
                              "pass this explicitly per-run when you want either a single "
                              "reproducible run or a multi-seed comparison.")
    args = parser.parse_args()
    if args.use_cnn and not HAS_TORCH:
        parser.error("--use_cnn requires PyTorch; refusing a synthetic-workload fallback")
    if args.device == "cuda" and (not HAS_TORCH or not torch.cuda.is_available()):
        parser.error("CUDA requested but unavailable; install CUDA PyTorch on the GPU host")
    if args.compute_ops:
        parser.error("--compute_ops is disabled: an untyped MAC vector cannot specify VVA work or communication units. Use --timing_model full_frame --use_cnn")
    if args.timing_model == "full_frame" and not args.use_cnn:
        parser.error("full_frame timing requires an extracted workload with operation and byte metadata")
    if not 0 < args.mac_utilization <= 1 or not math.isfinite(args.vva_ops_per_cycle) or args.vva_ops_per_cycle <= 0:
        parser.error("utilization must be in (0,1] and VVA rate finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.on_bandwidth_gbs, args.off_bandwidth_gbs)):
        parser.error("bandwidths must be finite and positive")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if HAS_TORCH:
            torch.manual_seed(args.seed)
        print(f">> Seeded RNGs with --seed {args.seed} (random search baseline B "
              f"and DDPG exploration noise are now reproducible for this run)\n")

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
            custom_model_path=args.custom_model, return_work=args.timing_model == "full_frame"
        )
        real_task_graph, num_tasks, task_labels = extracted[:3]
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
    if args.timing_model == "full_frame":
        operations, kinds = extracted[3:]
        compute_latency = compute_seconds(
            operations, kinds, utilization=args.mac_utilization,
            vva_ops_per_cycle=args.vva_ops_per_cycle
        )
        real_task_graph = edge_bytes(real_task_graph, kinds)
        args.on_lat = 1 / (args.on_bandwidth_gbs * 1e9)
        args.off_lat = 1 / (args.off_bandwidth_gbs * 1e9)
        print(">> Full-frame approximation in seconds: arithmetic + byte-hop serialization; no contention or streaming schedule")
    else:
        print(">> Compute latency disabled (communication-only objective)")

    env = MultiChipEnvironment(
        num_chips_x=args.chips_x, num_chips_y=args.chips_y,
        rows_per_chip=args.rows, cols_per_chip=args.cols,
        on_chip_latency=args.on_lat, off_chip_latency=args.off_lat,
        topology=args.topology,
        task_graph=real_task_graph,  # Passes the CNN graph here (or None for random)
        num_tasks=num_tasks,
        compute_latency=compute_latency,
        timing_units="seconds" if args.timing_model == "full_frame" else "proxy"
    )
    
    print(f"System : {env.topo}")
    print(f"Tasks  : {env.num_tasks}")
    print(f"Algo   : {args.algo}")
    print("-" * 50)

    # 3. Run the algorithms
    run_started = time.perf_counter()
    if args.algo == "ddpg":
        cost = run_ddpg(env, n_episodes=args.epochs, batch_z=args.batch_z,
                         baseline_trials=args.baseline_trials,
                         train_every=args.train_every, device=args.device,
                         save_checkpoint=args.save_checkpoint,
                         load_checkpoint=args.load_checkpoint,
                         checkpoint_every=args.checkpoint_every)
    elif args.algo == "sa":
        cost = run_sa(env, n_iter=args.iters)
    else:
        cost = run_random(env, n_trials=args.iters)

    bd = env.chip_breakdown()
    print(f"\nFinal placement cost : {cost:.6g}")
    print(f"  On-chip  comm cost : {bd['on_chip_cost']:.6g}")
    print(f"  Off-chip comm cost : {bd['off_chip_cost']:.6g}")
    print(f"  Chip placement     : {env.placement[:env.num_tasks]}")
    if args.report:
        with open(args.report, "w") as stream:
            json.dump({"config": vars(args), "tasks": env.num_tasks,
                       "objective_units": env.timing_units, "best_cost": cost,
                       "placement_chip_major": env.placement.tolist(),
                       "seconds_elapsed": time.perf_counter() - run_started,
                       "torch_version": torch.__version__ if HAS_TORCH else None,
                       "cuda_available": HAS_TORCH and torch.cuda.is_available(),
                       "limitations": ["MLP policy, not paper CNN", "no shared-link contention",
                           "no block-streaming schedule", "uniform channel partitioning",
                           "full_frame excludes residual/concat timing", "replay not persisted"]},
                      stream, indent=2)

if __name__ == "__main__":
    main()
