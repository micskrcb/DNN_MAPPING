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
from datetime import timedelta
import numpy as np

# Make sure the original src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_chip_topology import MultiChipTopology
from multi_chip_environment import MultiChipEnvironment

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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

    def extract_cnn_task_graph(channels_per_partition: int = 8):
        """
        PAPER FIX (Sec 3.1.1, Fig. 5-6): partition each CONV/FC layer's
        weights along the input channel C and output channel K into a grid
        of VMM (vector-matrix-multiply) cores, each feeding a VVA (vector-
        vector-accumulation) core that reduces the partial sums for its
        output-channel group. This replaces the old "one task per whole
        layer" model, which produced only ~15 tasks for this demo CNN --
        nowhere near the paper's reported ~900-1900 logic cores for real
        networks (Fig. 6).

        Activation (ReLU) and pooling are NOT separate tasks: per the
        paper's core architecture (Sec 2.2), these run inside each core's
        transformation unit, so only Conv2d/Linear layers are partitioned.

        Args:
            channels_per_partition: max output channels per partition group
                (smaller -> more, finer-grained logic cores, closer to the
                paper's per-model core counts in Fig. 6). Set to 0 to
                restore the old unpartitioned one-task-per-layer behavior
                (useful for quick smoke tests).

        Returns:
            task_graph: directed volume matrix, task_graph[i, j] = elements
                sent from logic core i to logic core j.
            num_tasks: total logic core count.
            labels: human-readable label per task index, for debugging.
        """
        model = SimpleCNN()
        dummy_input = torch.randn(1, 3, 32, 32)

        # Capture shape info only for Conv2d/Linear -- the only layers
        # actually partitioned into logic cores (see docstring above).
        captured = []  # list of (kind, C_in, C_out, H, W)

        def make_hook(kind):
            def hook_fn(module, inp, out):
                if kind == "conv":
                    _, C_out, H, W = out.shape
                    C_in = module.in_channels
                else:  # linear
                    C_out = module.out_features
                    C_in = module.in_features
                    H = W = 1
                captured.append((kind, C_in, C_out, H, W))
            return hook_fn

        hooks = []
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(make_hook("conv")))
            elif isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(make_hook("linear")))

        model(dummy_input)
        for h in hooks:
            h.remove()

        if channels_per_partition <= 0:
            # Legacy behavior: one task per Conv2d/Linear layer (activation/
            # pooling layers are no longer separately counted, unlike the
            # very first version of this extractor).
            num_tasks = len(captured)
            task_graph = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            labels = []
            for i, (kind, C_in, C_out, H, W) in enumerate(captured):
                labels.append(f"L{i}_{kind}")
                if i < num_tasks - 1:
                    task_graph[i, i + 1] = C_out * H * W
            return task_graph, num_tasks, labels

        # --- Channel-partitioned extraction (paper Sec 3.1.1) ---
        layer_meta = []   # (vmm_ids[M][N], vva_ids[M], M, N, partial_vol)
        labels = []
        edges = []        # (src_id, dst_id, volume)
        next_id = 0
        prev_M = 1         # first layer's few input channels aren't split

        for L, (kind, C_in, C_out, H, W) in enumerate(captured):
            M = max(1, math.ceil(C_out / channels_per_partition))
            N = prev_M

            vmm_ids = [[None] * N for _ in range(M)]
            vva_ids = [None] * M
            out_per_group = math.ceil(C_out / M)
            partial_vol = out_per_group * H * W  # VMM/VVA output size for this group

            for m in range(M):
                for n in range(N):
                    vmm_ids[m][n] = next_id
                    labels.append(f"L{L}_{kind}_VMM_m{m}_n{n}")
                    next_id += 1
                vva_ids[m] = next_id
                labels.append(f"L{L}_{kind}_VVA_m{m}")
                next_id += 1
                for n in range(N):
                    edges.append((vmm_ids[m][n], vva_ids[m], partial_vol))

            layer_meta.append((vmm_ids, vva_ids, M, N, partial_vol))
            prev_M = M

        # VVA(L, m) broadcasts its accumulated output-channel-group slice to
        # every VMM(L+1, m', n'=m) that consumes it as an input-channel group.
        for L in range(len(layer_meta) - 1):
            _, vva_ids, M, _, partial_vol = layer_meta[L]
            vmm_next, _, M_next, _, _ = layer_meta[L + 1]
            for m in range(M):
                for m_next in range(M_next):
                    edges.append((vva_ids[m], vmm_next[m_next][m], partial_vol))

        num_tasks = next_id
        task_graph = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        for src, dst, vol in edges:
            task_graph[src, dst] += vol

        return task_graph, num_tasks, labels


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
        def __init__(self, state_dim, action_dim=2, lr_actor=1e-4, lr_critic=1e-3,
                     gamma=0.99, tau=0.005, device=None):
            # PERF FIX: this agent previously never checked for a GPU, even
            # if one was available -- for a partitioned CNN workload the
            # state vector is total_cores + num_tasks (e.g. 4096+906=5002
            # dims), so the first Linear layer alone has ~1.3M parameters.
            # On a CUDA GPU this is dramatically faster per training step.
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

        def select_action(self, state, noise_scale=0.1):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state).squeeze(0).cpu().numpy()
            self.actor.train()
            # Add exploration noise
            action += noise_scale * np.random.randn(self.action_dim)
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
        m = np.zeros(self.total_rows * self.total_cols, dtype=np.float32)
        for t, c in enumerate(self._placement):
            if c >= 0:
                m[c] = (t + 1) / self.num_tasks  # +1 so task 0 != "empty" (0.0)

        # PAPER FIX: expose BOTH directions of communication volume,
        # aggregated over the WHOLE upcoming batch of up to batch_z tasks
        # (not just a single "current task"), since one action now assigns
        # all of them at once.
        remaining = self.num_tasks - self._task_ptr
        n_batch = min(self.batch_z, remaining) if remaining > 0 else 0
        task_comm = np.zeros(self.num_tasks, dtype=np.float32)
        for k in range(n_batch):
            idx = self._task_ptr + k
            task_comm += self.env.task_graph[idx] + self.env.task_graph[:, idx]

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
        intended position, ties broken by first-found (core-index order)."""
        intended_x = int(math.floor(target_x))
        intended_y = int(math.floor(target_y))
        intended_x = min(max(intended_x, 0), self.total_cols - 1)
        intended_y = min(max(intended_y, 0), self.total_rows - 1)
        intended_core = intended_y * self.total_cols + intended_x

        if intended_core not in self._occupied:
            core_id = intended_core
        else:
            best_core = -1
            best_dist = None
            for c in range(self.total_rows * self.total_cols):
                if c in self._occupied:
                    continue
                cx = c % self.total_cols
                cy = c // self.total_cols
                dist = abs(cx - intended_x) + abs(cy - intended_y)  # Manhattan
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_core = c
            core_id = best_core

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

        self.env.place(self._placement)
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
        return self._placement.copy()


# ---------------------------------------------------------------------------
# DDPG RL Training Loop (Based on Paper Pseudocode)
# ---------------------------------------------------------------------------

def run_ddpg(env: MultiChipEnvironment, n_episodes: int = 500, batch_size: int = 64,
             baseline_trials: int = 1000, batch_z: int = 3, train_every: int = 5,
             device: str = None) -> float:
    if not HAS_TORCH:
        print("[DDPG] torch not installed — falling back to random search")
        return run_random(env)

    # Paper Algorithm 1: B is the latency of the best placement found by
    # random search, computed once up front and held fixed as the reward
    # normalizer. run_random() leaves env.placement mutated -- harmless,
    # since every training episode starts with mapper.reset() -> env.reset().
    print(f"[DDPG] Computing random-search baseline B ({baseline_trials} trials)...")
    baseline_latency = run_random(env, n_trials=baseline_trials)
    print(f"[DDPG] Baseline B = {baseline_latency:.4f}")

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
    print(f"[DDPG] Using device: {agent.device}")
    replay_buffer = ReplayBuffer()
    mapper = MultiChipCoreMapper(env, baseline_latency=baseline_latency, batch_z=batch_z)

    best_cost = float('inf')
    best_placement = None
    best_grid = None

    noise_scale = 1.0
    # Decay so noise_scale reaches ~0.01 by 80% of training, regardless of
    # n_episodes -- a fixed 0.995 barely decays (~8% remaining) over a
    # 500-1000 episode run, which was masking whether the policy had
    # actually converged versus still being exploration-noise-dominated.
    target_episode = max(1, int(0.8 * n_episodes))
    noise_decay = 0.01 ** (1.0 / target_episode)

    # PERF FIX: agent.train() previously ran on EVERY environment step
    # (num_tasks/batch_z steps per episode -- e.g. 302 for a 906-task
    # workload at batch_z=3), each a full actor+critic forward/backward
    # pass. train_every spaces these out; replay_buffer.add() still runs
    # every step so no experience is lost, just the gradient-update
    # frequency is reduced.
    global_step = 0
    start_time = time.time()

    for ep in range(1, n_episodes + 1):
        state = mapper.reset()
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
            per_ep = elapsed / ep
            eta_sec = per_ep * (n_episodes - ep)
            eta_str = str(timedelta(seconds=int(eta_sec)))
            print(f"# of epochs: {ep:4d} | Current Cost: {final_cost:.2f} | "
                  f"Best Cost: {best_cost:.2f} | {per_ep:.2f}s/ep | ETA: {eta_str}")

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

def run_sa(env: MultiChipEnvironment, n_iter: int = 5000,
           T_start: float = 100.0, T_end: float = 0.1) -> float:
    n, k = env.num_tasks, env.total_cores
    placement = np.array(random.sample(range(k), min(n, k)), dtype=np.int32)
    env.place(placement)
    best_cost, best_p = env.evaluate(), placement.copy()

    T = T_start
    alpha = (T_end / T_start) ** (1.0 / n_iter)

    for step in range(n_iter):
        i, j = random.sample(range(n), 2)
        new_p = placement.copy(); new_p[i], new_p[j] = new_p[j], new_p[i]
        env.place(new_p); new_cost = env.evaluate()

        if new_cost < best_cost or random.random() < math.exp(-(new_cost - best_cost) / max(T, 1e-9)):
            placement = new_p
            if new_cost < best_cost:
                best_cost, best_p = new_cost, new_p.copy()
        T *= alpha

    env.place(best_p)
    return best_cost

def run_random(env: MultiChipEnvironment, n_trials: int = 1000) -> float:
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
    parser.add_argument("--iters", type=int, default=5000, help="SA/random iterations")
    parser.add_argument("--epochs", type=int, default=1000, help="DDPG training epochs")
    parser.add_argument("--batch_z", type=int, default=3,
                         help="Number of logic cores placed per DDPG action, "
                              "per the paper's batched action [x1,y1,...,xz,yz] "
                              "(Sec 3.2). Clamped to num_tasks if larger.")
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
    
    # NEW ARGUMENT: Flag to use the real CNN workload
    parser.add_argument("--use_cnn", action="store_true", help="Use real CNN workload instead of random")
    parser.add_argument("--channels_per_partition", type=int, default=8,
                         help="Max output channels per VMM/VVA partition group when "
                              "--use_cnn is set (paper Sec 3.1.1, Fig 5-6). Smaller = "
                              "more, finer-grained logic cores (closer to the paper's "
                              "per-model core counts in Fig 6). Set to 0 for the old "
                              "one-task-per-layer behavior.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed for random/numpy/torch RNGs. Not fixed by default -- "
                              "the paper itself (Sec 4.5, Fig 20) runs 5 different seeds "
                              "and averages results rather than using one fixed seed, so "
                              "pass this explicitly per-run when you want either a single "
                              "reproducible run or a multi-seed comparison.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if HAS_TORCH:
            torch.manual_seed(args.seed)
        print(f">> Seeded RNGs with --seed {args.seed} (random search baseline B "
              f"and DDPG exploration noise are now reproducible for this run)\n")

    # 1. Determine which workload to use
    real_task_graph = None
    num_tasks = None

    if args.use_cnn and HAS_TORCH:
        print(">> Extracting Real CNN Workload via PyTorch Hooks...")
        real_task_graph, num_tasks, task_labels = extract_cnn_task_graph(
            channels_per_partition=args.channels_per_partition
        )
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
              f"(paper's 1024-core config)")
        print(f"  2. Increasing --channels_per_partition (fewer, coarser logic cores)")
        sys.exit(1)

    # 2. Build the Environment
    env = MultiChipEnvironment(
        num_chips_x=args.chips_x, num_chips_y=args.chips_y,
        rows_per_chip=args.rows, cols_per_chip=args.cols,
        on_chip_latency=args.on_lat, off_chip_latency=args.off_lat,
        task_graph=real_task_graph,  # Passes the CNN graph here (or None for random)
        num_tasks=num_tasks          # Passes the CNN task count here
    )
    
    print(f"System : {env.topo}")
    print(f"Tasks  : {env.num_tasks}")
    print(f"Algo   : {args.algo}")
    print("-" * 50)

    # 3. Run the algorithms
    if args.algo == "ddpg":
        cost = run_ddpg(env, n_episodes=args.epochs, batch_z=args.batch_z,
                         baseline_trials=args.baseline_trials,
                         train_every=args.train_every, device=args.device)
    elif args.algo == "sa":
        cost = run_sa(env, n_iter=args.iters)
    else:
        cost = run_random(env, n_trials=args.iters)

    bd = env.chip_breakdown()
    print(f"\nFinal placement cost : {cost:.4f}")
    print(f"  On-chip  comm cost : {bd['on_chip_cost']:.4f}")
    print(f"  Off-chip comm cost : {bd['off_chip_cost']:.4f}")
    print(f"  Chip placement     : {env.placement[:env.num_tasks]}")

if __name__ == "__main__":
    main()