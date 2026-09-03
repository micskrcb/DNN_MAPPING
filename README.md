# Policy Gradient-based Core Placement Optimization for Multi-chip Many-core systems


![Overview](./Overview.png)

## Supported algoritms
* Exact Mapping (zigzag, neighbor)
* Random Search
* Simulated Annealing
* Reinforcement Learning

## AS-IS:
* Mapping algorithm for single-chip system is available.
* Community detection code is uploaded, but it is currently not compatible this code.
* Execution for multi-chip system is not yet uploaded.

# MODIFIED (VERSION:HARSHIT)
Here is the updated, comprehensive README. This includes the specific instructions for a non-GPU (CPU-only) installation and a deep dive into the code architecture.

---

## RL-Chip-Mapper: Core Placement with PPO

This project optimizes the physical mapping of logic cores onto a Network-on-Chip (NoC) grid. It uses **Proximal Policy Optimization (PPO)** to minimize the total communication energy/latency by placing high-traffic core pairs close together.

---

## 1. Initial Setup (CPU / No-CUDA)

To ensure computational reproducibility and isolation of dependencies, the system should be initialized within a virtualized environment using Python 3.10.

If you do not have an NVIDIA GPU or simply want to run this on your laptop's processor, follow these steps to ensure `torch` installs the CPU-only version.

### Create the Environment
```bash
# 1. Navigate to the project root
cd Core_Placement_with_Reinforcement_Learning

# 2. Create a virtual environment
python3.10 -m venv chip_env

# 3. Activate the environment
source chip_env/bin/activate

# 4. Install CPU-specific requirements
pip install --upgrade pip
# This command ensures you get the lightweight CPU version of Torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas
```

---

## 2. Execution & Error Handling

Run the training using:
```bash
python3 main.py --config config/example.py
```

### Common Modern Python Errors (The Fixes)
Since this code was written for older libraries, you must apply these manual fixes:

1.  **NumPy Attributes:** In `parsedata.py` (Line 47), change `dtype=np.int` to `dtype=int`.
2.  **Tuple Unpacking:** In `agent/agents.py`, ensure the step call is `reward, current_grid = self.env.step(action)`.
3.  **Config Setup:** Ensure `use_cuda` is set to `False` in your `config/example.py` file to avoid searching for a GPU.

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `AttributeError: module 'numpy' has no attribute 'int'` | NumPy 1.20+ removed `np.int`. | Change `np.int` to `int` in `parsedata.py`. |
| `AttributeError: module 'numpy' has no attribute 'float'` | NumPy 1.20+ removed `np.float`. | Change `np.float` to `float` in affected files. |
| `TypeError: bad operand type for unary -: 'tuple'` | The `step` function returns a tuple (Reward, Grid). | Unpack the reward: `reward, grid = self.env.step(action)`. |

---

---

## 3. Project Architecture (Detailed)

### **The Workflow Theory**
The system operates on an **Agent-Environment Loop**. The Agent (the AI) suggests a layout; the Environment (the Chip) calculates how much "battery" or "time" that layout would waste; the Agent then adjusts its "brain" to try and get a better score next time.



---

### **Quick File Overview**
* `main.py`: The "Manager" that starts everything.
* `parsedata.py`: The "Data Entry" that reads your chip requirements.
* `src/agent/`: The "Brain" (PPO logic and Neural Networks).
* `src/env/`: The "Physics" (The chip grid and reward math).
* `src/utils.py`: Helper functions for logging and timing.





### Structural Decomposition of Source Code

The project is organized into modular directories that separate the learning agents from the physical environment constraints.

### **Core Directory: `src/`**

| Sub-directory / File | Academic Classification | Functional Description |
| :--- | :--- | :--- |
| **`agent/`** | **Policy Optimization Engine** | Contains the neural network architectures and learning algorithms. |
| &nbsp;&nbsp;`actor.py` | Stochastic Policy Network | Defines the deep neural network responsible for coordinate prediction. |
| &nbsp;&nbsp;`agents.py` | PPO Controller | Implements the Proximal Policy Optimization logic, including clipped objective functions. |
| &nbsp;&nbsp;`model.py` | Network Backbone | Defines the base architectural layers and weight initialization parameters. |
| **`env/`** | **Environmental Modeling** | Defines the physical constraints and cost functions of the NoC. |
| &nbsp;&nbsp;`core_mapper.py` | Spatial Constraint Handler | Manages the 2D grid and implements heuristic-based overlap resolution. |
| &nbsp;&nbsp;`reward.py` | Objective Function | Calculates the total communication cost based on L1 norm distances. |
| **`runner/`** | **Execution Management** | contains the high-level logic for different optimization strategies. |
| &nbsp;&nbsp;`reinforcement_learning.py` | RL Training Pipeline | Orchestrates the interaction between the PPO agent and the NoC environment. |
| &nbsp;&nbsp;`simulated_annealing.py` | Metaheuristic Baseline | Provides a Simulated Annealing baseline for comparative performance analysis. |
| **`config/`** | **Hyperparameter Definition** | Contains JSON/Python scripts for grid dimensions and learning rates. |

---

## 4. Theoretical Framework

### **Problem Formulation**
The core placement problem is modeled as a discrete optimization task where the agent must determine a mapping function $M: C \rightarrow P$, where $C$ is the set of cores and $P$ is the set of physical coordinates on a 2D mesh.

### **Proximal Policy Optimization (PPO)**
The agent utilizes a Clipped Surrogate Objective to prevent excessively large policy updates. The probability ratio $r_t(\theta)$ is constrained within the interval $[1 - \epsilon, 1 + \epsilon]$, ensuring stable convergence in complex topological search spaces.



### **Heuristic Conflict Resolution**
When the policy network proposes a non-unique coordinate (collision), the `core_mapper` invokes a radial search algorithm (`detect_circle`) to identify the nearest vacant node, maintaining a valid 1-to-1 mapping throughout the stochastic exploration phase.

---

## 5. Modifications Made
We updated the original "static" code to be more interactive and persistent:

1.  **Environment Unpacking:** Modified `CoreMapper.step` to return the `placement_map` so we can visualize the physical core layout.
2.  **Early Stopping:** Changed the infinite `while True` loop to `while epoch <= 500 and best_reward > 768`. This prevents the CPU from running forever once the optimal solution is found.
3.  **NumPy Compatibility:** Patched type-hinting errors caused by the evolution of the NumPy library.
4.  **Print core val:**  Print the best core after the simulation is complete.

Below is a comprehensive technical log of the modifications performed on the source code to ensure compatibility with modern Python environments and to enhance the functionality of the Reinforcement Learning agent.

---

### **Detailed Modification Log**

| File Path | Original State | Modified State | Rationale |
| :--- | :--- | :--- | :--- |
| `src/parsedata.py` | `dtype=np.int` | `dtype=int` | **NumPy Compatibility:** NumPy 1.20+ removed the `np.int` alias. Using the built-in `int` prevents `AttributeError`. |
| `src/env/core_mapper.py` | `return -self.calculate_reward(...)` | `return -self.calculate_reward(...), placement_map` | **Observability:** Added the local 2D array to the return statement to allow the agent to visualize the final core layout. |
| `src/agent/agents.py` | `while True:` | `while epoch <= 500 and best_reward > 768:` | **Convergence Control:** Replaced the infinite loop with an early-stopping condition based on a target communication cost. |
| `src/agent/agents.py` | `reward = self.env.step(action)` | `reward, current_grid = self.env.step(action)` | **Tuple Unpacking:** Updated to handle the additional grid data returned by the modified `core_mapper`. |
| `src/agent/agents.py` | `best_reward = 1e12` (Inside Loop) | `best_reward = 1e12` (Outside Loop) | **Persistence:** Moved the initialization outside the while-loop so the best score is remembered across training epochs. |
| `src/agent/agents.py` | No Weight Persistence | `torch.save(self.policy.state_dict(), path)` | **Model Serialization:** Added logic to save the neural network weights (`.pth`) to disk when a new "Best Reward" is achieved. |
| `src/agent/agents.py` | No Load Logic | `self.policy.load_state_dict(...)` | **Transfer Learning:** Integrated logic to check for existing weights upon initialization to resume training instead of starting from scratch. |
| `src/config/example.py` | `"use_cuda": True` | `"use_cuda": False` | **Hardware Adaptation:** Configured for CPU-only execution to accommodate systems without dedicated NVIDIA hardware. |

---

### **Specific Change Summaries by Module**

#### **1. Data Parsing (`parsedata.py`)**
The original code relied on deprecated NumPy data types. By shifting to Python's native `int` and `float`, we ensured the script runs on current versions (Python 3.10+) without requiring an outdated environment.

#### **2. Environment Interaction (`core_mapper.py`)**
In the original version, the physical mapping of cores was a "black box"—you could see the score, but not the grid. We exposed the `placement_map` variable. This matrix represents the spatial configuration of Core IDs on the NoC mesh, which is critical for verifying the agent's logic.

#### **3. Agent Training Logic (`agents.py`)**
The most significant logic changes occurred here. We transitioned the agent from a "disposable" training script to a persistent one. 
* **State Unpacking:** Fixed a `TypeError` where the agent attempted to perform math on a tuple rather than a scalar reward.
* **Optimization:** The agent now tracks the `best_grid` across the entire session, ensuring the final output is the absolute best configuration found during exploration.

#### **4. Utility and Support (`utils.py` & `main.py`)**
Minor adjustments were made to the execution wrapper to handle the modified return values from the training runner, ensuring that the timing and logging decorators continued to function correctly despite changes to the underlying algorithm.



### **Verification of Changes**
To verify these changes, observe the terminal output during execution. You should now see a 2D matrix (the `placement_map`) printed alongside the epoch count, and a `chip_placement_brain.pth` file should appear in your directory, confirming that weight persistence is active.

---