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

    def extract_cnn_task_graph():
        """Passes dummy data through the CNN to build the communication matrix."""
        model = SimpleCNN()
        dummy_input = torch.randn(1, 3, 32, 32) # Standard 32x32 image
        layer_output_sizes = []

        def hook_fn(module, input, output):
            layer_output_sizes.append(output.numel())

        hooks = []
        for layer in model.features:
            hooks.append(layer.register_forward_hook(hook_fn))

        model(dummy_input)

        for h in hooks:
            h.remove()

        num_tasks = len(layer_output_sizes)
        task_graph = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        # Sequential communication: Layer i sends data only to Layer i+1
        for i in range(num_tasks - 1):
            task_graph[i, i + 1] = layer_output_sizes[i]

        return task_graph, num_tasks


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
        def __init__(self, state_dim, action_dim=2, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
            self.actor = Actor(state_dim, action_dim)
            self.actor_target = Actor(state_dim, action_dim)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

            self.critic = Critic(state_dim, action_dim)
            self.critic_target = Critic(state_dim, action_dim)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

            self.gamma = gamma
            self.tau = tau

        def select_action(self, state, noise_scale=0.1):
            state = torch.FloatTensor(state).unsqueeze(0)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state).squeeze(0).numpy()
            self.actor.train()
            # Add exploration noise
            action += noise_scale * np.random.randn(2)
            return np.clip(action, -1.0, 1.0)

        def train(self, replay_buffer, batch_size=64):
            if len(replay_buffer) < batch_size:
                return

            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

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
    def __init__(self, env: MultiChipEnvironment, baseline_latency: float = None):
        """
        Args:
            baseline_latency: B in the paper's reward r_t = sqrt(B) - sqrt(L(P))
                (Algorithm 1, line 11) -- the best latency found by random
                search, computed once before training and held fixed. If
                None, the reward falls back to -sqrt(L(P)) (unnormalized)
                and a one-time warning is printed, since this is NOT what
                the paper specifies.
        """
        self.env = env
        topo = env.topo
        self.total_cols = topo.cols_per_chip * topo.num_chips_x
        self.total_rows = topo.rows_per_chip * topo.num_chips_y
        self.num_tasks  = env.num_tasks
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
        # State: Occupancy Map + Current Task Comm Volumes
        m = np.zeros(self.total_rows * self.total_cols, dtype=np.float32)
        for c in self._occupied:
            m[c] = 1.0
            
        task_comm = np.zeros(self.num_tasks, dtype=np.float32)
        if self._task_ptr < self.num_tasks:
            task_comm = self.env.task_graph[self._task_ptr].copy()
            max_vol = task_comm.max()
            if max_vol > 0:
                task_comm /= max_vol
                
        return np.concatenate([m, task_comm])

    def step(self, action):
        ax, ay = action[0], action[1]
        target_x = ((ax + 1.0) / 2.0) * (self.total_cols - 1)
        target_y = ((ay + 1.0) / 2.0) * (self.total_rows - 1)

        # PAPER IMPLEMENTATION (Sec 3.2, p.11-12): floor the continuous actor
        # output to get the intended integer grid position. If that core is
        # free, place there directly -- no search needed. Only on an actual
        # "contradiction" (core already occupied) do we search, and then by
        # MINIMUM MANHATTAN DISTANCE to the ORIGINAL intended position (not
        # Euclidean, not re-scanned against the current action every step).
        # Ties broken by first-found in core-index order, per the paper text.
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
             baseline_trials: int = 1000) -> float:
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

    agent = DDPGAgent(state_dim=state_dim, action_dim=2)
    replay_buffer = ReplayBuffer()
    mapper = MultiChipCoreMapper(env, baseline_latency=baseline_latency)

    best_cost = float('inf')
    best_placement = None
    best_grid = None

    noise_scale = 1.0    
    noise_decay = 0.995  

    for ep in range(1, n_episodes + 1):
        state = mapper.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, noise_scale=noise_scale)
            reward, done, grid, final_cost = mapper.step(action)
            next_state = mapper._occ_map()
            
            replay_buffer.add(state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size)
            state = next_state

        noise_scale = max(0.01, noise_scale * noise_decay)

        if final_cost < best_cost:
            best_cost = final_cost          
            best_grid   = grid
            best_placement = mapper.get_placement()

        if ep % 10 == 0 or ep == n_episodes:
             print(f"# of epochs: {ep:4d} | Current Cost: {final_cost:.2f} | Best Cost: {best_cost:.2f}")

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
    
    # NEW ARGUMENT: Flag to use the real CNN workload
    parser.add_argument("--use_cnn", action="store_true", help="Use real CNN workload instead of random")
    args = parser.parse_args()

    # 1. Determine which workload to use
    real_task_graph = None
    num_tasks = None
    
    if args.use_cnn and HAS_TORCH:
        print(">> Extracting Real CNN Workload via PyTorch Hooks...")
        real_task_graph, num_tasks = extract_cnn_task_graph()
        print(f">> Successfully extracted {num_tasks} layers/tasks from CNN.\n")

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
        cost = run_ddpg(env, n_episodes=args.epochs)
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