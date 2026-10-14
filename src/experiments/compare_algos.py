"""
compare_algos.py
----------------
Compares the performance of Simulated Annealing (SA) vs Proximal Policy 
Optimization (PPO) for multi-chip core placement on the exact same task graph.
"""

import time
import numpy as np
from multi_chip_environment import MultiChipEnvironment
from run_multi_chip import run_sa, run_ppo

def main():
    # 1. Define system parameters (2x2 chips, 4x4 cores per chip = 64 cores)
    chips_x, chips_y = 2, 2
    rows, cols = 4, 4
    num_tasks = 64 
    
    # 2. Generate a shared task graph for a FAIR comparison
    print("Initializing shared task graph...")
    rng = np.random.default_rng(42)
    task_graph = rng.integers(0, 10, size=(num_tasks, num_tasks)).astype(np.float32)
    task_graph = (task_graph + task_graph.T) / 2
    np.fill_diagonal(task_graph, 0)
    
    # 3. Create two identical environments using the same task graph
    env_sa = MultiChipEnvironment(
        num_chips_x=chips_x, num_chips_y=chips_y,
        rows_per_chip=rows, cols_per_chip=cols,
        task_graph=task_graph.copy(), num_tasks=num_tasks
    )
    
    env_ppo = MultiChipEnvironment(
        num_chips_x=chips_x, num_chips_y=chips_y,
        rows_per_chip=rows, cols_per_chip=cols,
        task_graph=task_graph.copy(), num_tasks=num_tasks
    )
    
    print(f"System: {chips_x * chips_y} chips, {rows * cols} cores/chip (Total: {env_sa.total_cores} cores)")
    print("-" * 50)
    
    # 4. Run Simulated Annealing (SA)
    print("Running Simulated Annealing (SA)...")
    start_time = time.time()
    # You can increase n_iter for better SA results, default in your script is 5000
    sa_cost = run_sa(env_sa, n_iter=2000) 
    sa_time = time.time() - start_time
    sa_bd = env_sa.chip_breakdown()
    
    print("-" * 50)
    
    # 5. Run Proximal Policy Optimization (PPO)
    print("Running PPO (Reinforcement Learning)...")
    start_time = time.time()
    # Epochs set to 500 to match your default PPO settings
    ppo_cost = run_ppo(env_ppo, epochs=500)
    ppo_time = time.time() - start_time
    ppo_bd = env_ppo.chip_breakdown()
    
    print("=" * 50)
    print("🏆 COMPARISON RESULTS 🏆")
    print("=" * 50)
    print(f"Simulated Annealing (SA):")
    print(f"  Final Total Cost : {sa_cost:.2f}")
    print(f"  On-chip Cost     : {sa_bd['on_chip_cost']:.2f}")
    print(f"  Off-chip Cost    : {sa_bd['off_chip_cost']:.2f}")
    print(f"  Time taken       : {sa_time:.2f} seconds")
    print()
    print(f"PPO (Reinforcement Learning):")
    print(f"  Final Total Cost : {ppo_cost:.2f}")
    print(f"  On-chip Cost     : {ppo_bd['on_chip_cost']:.2f}")
    print(f"  Off-chip Cost    : {ppo_bd['off_chip_cost']:.2f}")
    print(f"  Time taken       : {ppo_time:.2f} seconds")
    print("=" * 50)

    # Determine winner (lower cost is better)
    if ppo_cost < sa_cost:
        print(f"Winner: PPO beat SA by {sa_cost - ppo_cost:.2f} cost difference!")
    elif sa_cost < ppo_cost:
        print(f"Winner: SA beat PPO by {ppo_cost - sa_cost:.2f} cost difference!")
    else:
        print("Tie! Both algorithms found placements with the exact same cost.")

if __name__ == "__main__":
    main()