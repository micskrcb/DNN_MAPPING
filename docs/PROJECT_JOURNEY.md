# Project Journey: Core Placement Optimization for Multi-Chip Many-Core DNN Accelerators

**Document Purpose**: Internal, comprehensive, forensic technical reconstruction of the project from inception to its present state.  
**Repository**: `micskrcb/DNN_MAPPING`  
**Active Development Branch**: `cleaned-up` (tracking `origin/cpu`, commit `1e9dee285d25d1ea7a59a996c7366cfd222cb5e9`)  
**Historical Branch Head**: `main` (commit `41f9d024fd8d118c41682e144a22366248a4953c`)  
**Primary Target Literature**: Nan Wu, Farhad Deng, Yuan Xie, et al., *Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning*, ACM Transactions on Design Automation of Electronic Systems (TODAES), Vol. 26, No. 4, Article 29, 2020. DOI: [10.1145/3418498](https://doi.org/10.1145/3418498).

---

## Internal Evidence Synthesis Table

The following table summarizes all primary internal documents analyzed during this forensic reconstruction.

| Document Name | Approximate Stage | Algorithms Discussed | Code / Experiment Changes | Reported Results | Conclusions Made at Stage | Claims Later Corrected or Superseded | Contradictions with Other Documents | Missing or Uncertain Info |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`docs/archive/prev_version_readmes/README-for_orignal+initialchanges.md`](archive/prev_version_readmes/README-for_orignal+initialchanges.md) | Stage 1 (Fork & CPU Port, Summer 2026) | Single-chip PPO, Exact Mapping (Zigzag, Neighbor), Random Search, SA | Ported `WOOSHIK-M` PPO codebase to Python 3.10; fixed `np.int`, `np.float`, tuple unpacking in `CoreMapper.step`; CPU installation. | Single-chip toy LUT run finished (`best_reward > 768`). | PPO works for core placement on a single chip; environment functions. | Single-chip netlist mapping was assumed directly applicable to multi-chip DNN accelerators. | Claims multi-chip core placement, but code only executed single-chip netlist L1 distance. | Exact paper hardware, multi-chip NoC, and DNN workload models were absent. |
| [`docs/archive/prev_version_readmes/old_section_41-50_of_readme_by_gpt.md`](archive/prev_version_readmes/old_section_41-50_of_readme_by_gpt.md) | Stage 2 (Early Multi-Chip & SimpleCNN Draft) | DDPG, RS, SA | Implemented `multi_chip_topology.py`, `multi_chip_environment.py`, channel partitioning on `SimpleCNN` (906 logic cores). | First multi-chip run on SimpleCNN took 10 min for 40 episodes before scaling bottlenecks. | DDPG framework operational on multi-chip; needs optimization and paper targets. | Believed the objective was simply reducing global communication delay across all pairs. | Assumed Figure 10's 50.5% DDPG reduction was immediately reproducible on full workloads. | Exact paper batch size $z$ and layer partition grids were unknown. |
| [`docs/archive/prev_version_readmes/README_before_GPTplus.md`](archive/prev_version_readmes/README_before_GPTplus.md) | Stage 3 (Exhaustive Manual, early Sept 2026) | DDPG, RS, SA, Batched Actions ($z=3$) | Added cached sparse adjacency list ($19.4\times$ speedup), `--train_every`, GPU auto-detection, live ETA, `extract_model_task_graph` for torchvision. | SimpleCNN 10k ep: Best DDPG 340,992 vs Baseline $B=444,416$ (23.3% gain). AlexNet ran partially. | Optimization resolved execution bottlenecks; DDPG demonstrates learning convergence. | Claimed 23.3% gain was proof of learning; later proved to be random exploration in an unconstrained space. | Claimed DDPG was converging, but noted Current Cost never tightened near Best Cost. | Hardware bandwidths and exact VVA latency were uncalibrated. |
| [`docs/archive/prev_version_readmes/claude_compressed_readme_6thsep.md`](archive/prev_version_readmes/claude_compressed_readme_6thsep.md) | Stage 4 (First AlexNet Full Run, Sept 6, 2026) | DDPG, RS, SA | Evaluated AlexNet (1,445 logic cores) on 4,096 cores; batched actions ($z=3$); Manhattan collision repair. | AlexNet ~870 ep: Best DDPG 6,749,184 vs Baseline $B=12,697,984$ (46.8% reduction). | Headline result: DDPG achieved 46.8% latency reduction, approaching the paper's 50.5%. | **Completely superseded**: The objective function used unscaled proxy latency; current cost oscillated wildly; deterministic policy had collapsed. | Conflicted with subsequent deterministic evaluation audits showing zero learning. | Unvectorized `_occ_map` loop and CPU thermal drift observed. |
| Commit [`41f9d02`](file:///home/harshit/.gemini/antigravity/scratch/DNN_MAPPING) & `temp.txt` | Stage 4 Archive (Sept 7, 2026) | DDPG | Added AlexNet checkpoint (`alexnet_ckpt.pt`, 46 MB) and committed raw execution dump (`temp.txt`). | Raw terminal log showed episodes oscillating between 8M and 14M while Best stayed at 6.74M. | Checkpoint captured a working policy. | Checkpoint inspection later revealed the deterministic policy placed tasks onto only 3 unique cores with 181 collisions. | Overlooked that best cost was discovered by exploration noise, not actor weights. | Deterministic evaluation was not executed at the time of commit. |
| [`docs/RECONCILIATION.md`](RECONCILIATION.md) | Stage 5 (Model Reconciliation, Sept 30, 2026) | Paper DDPG, BS, RS, SA, ASA, DDPG+ASA | FX extraction; exact Figure 6 counts (183 CONV, 932 FC); physical units (seconds, cycles, bytes); XY gateway routing; Kahn DAG staging. | Pure AlexNet-CONV random span 16.34%; FC span 4.05%. Old 2,080-ep CONV run had flat 0.001% span due to $M=1, N=61$ VVA bottleneck. | Old run logs were invalidated by flat objectives. Reconstructed simulator is placement-sensitive. | Flat objective caused earlier apparent lack of sensitivity; fixed partitioning restored headroom. | Demonstrates that earlier "46.8% improvement" was evaluated on a flawed cost metric. | Router startup and activation buffer stalls remain unmodeled. |
| `runs/local-cpu-5seed-2026-09-30/STUDY.md` | Stage 6 (Untrained Control Study, Sept 30, 2026) | DDPG, Untrained Control, RS, SA, ASA, BS | Bounded 5-seed study on AlexNet-CONV (12 training placements, 6 retained deterministic). | Trained DDPG 38.24 µs vs Untrained Control 38.47 µs (0.59% benefit; 95% CI [-0.34, +0.79] µs includes 0). | DDPG gains over RS/SA do not establish learning; untrained initial policy + collision repair already yields 38.4 µs. | Superseded early claims that DDPG beats baselines due to learned policy gradients. | Contradicted the belief that DDPG training was responsible for good placements. | Deep budget scaling on CPU was too slow without cloud GPUs. |
| `runs/NORMALIZATION_EXPERIMENTS.md` | Stage 6 (BatchNorm Diagnostic, Sept 30, 2026) | DDPG (Actor BN, Critic BN, Target Sync) | Evaluated target network BatchNorm evaluation mode, buffer sync, and Polyak soft updates. | Seed-4 100-placement run ended at 48.08 µs deterministic with 181 collisions and 3 intended cores. | BatchNorm in Bellman target calculations caused minibatch-dependent targets; repairs stabilized training but did not solve collapse. | Closed continuous DDPG normalization ablations as insufficient to fix fundamental discrete policy collapse. | Shows that continuous actor-critic cannot overcome discrete coordinate saturation. | Replay buffer capacity and soft update rates remained uncalibrated. |
| [`docs/RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md) & [`docs/archive/DDPG_REFERENCE_AUDIT.md`](archive/DDPG_REFERENCE_AUDIT.md) | Stage 7 (DDPG Failure Audit, Oct 8, 2026) | DDPG, Guided DDPG, Wolpertinger, Categorical Masked PPO | Audited Kaggle 3,000-placement CONV run; independent literature survey; implemented Guided DDPG with ASA demonstrations. | 3,000-ep DDPG: trained deterministic 37.64 µs vs untrained 37.45 µs; critic loss reached $1.48 \times 10^{12}$; 176/183 collisions. Guided DDPG 5-seed: trained 39.09 µs vs control 38.73 µs (-0.35 µs loss). | Continuous DDPG suffers from policy collapse and critic divergence in discrete placement spaces. Guided DDPG definitively rejected. | Officially closed the DDPG research direction after exhaustive 5-seed paired empirical ablation. | Proved that the 2020 paper's DDPG cannot learn without undisclosed stabilization mechanisms. | Exact proprietary simulator code used by Wu et al. remains missing from public literature. |
| `runs/kaggle-ppo-asa-v1250-2026-10-11/STUDY.md` & [`docs/archive/PAPER_SCALE_DECISION.md`](archive/PAPER_SCALE_DECISION.md) | Stage 8 (Sufficient-Scale PPO-ASA, Oct 11, 2026) | PPO-guided ASA (Shared Encoder), Frozen Control, ASA | Scaled PPO-ASA to 1,250,000 proposals, 1,250 PPO updates, 10-seed frozen holdout on AlexNet-FC. | Trained PPO-ASA 23.5026 µs vs Uniform Control 23.5072 µs vs ASA 23.4894 µs. Won 3/10 seeds vs control, 4/10 vs ASA. Policy entropy remained uniform. | Shared-encoder candidate-scoring PPO fails to learn; additional training epochs are futile. | Closed the candidate-scoring PPO-ASA architecture as a documented negative result. | Refuted the hypothesis that PPO-ASA only failed earlier due to insufficient training transitions. | Disjoint actor-critic separation was identified as a potential remaining confounder. |
| [`docs/archive/CLAUDE_AUDIT_RESPONSE.md`](archive/CLAUDE_AUDIT_RESPONSE.md) & [`docs/PPO_REFERENCE_ADAPTER.md`](PPO_REFERENCE_ADAPTER.md) | Stage 9 (Decoupled PPO Diagnostic, Oct 11, 2026) | Reference-Split PPO-ASA, Planted Signal, Oracle Audit | Decoupled actor and critic networks; added E0 (pool oracle headroom), E1 (planted proxy reward), E2 (real-reward pilot). | E1 planted signal passed (policy learned); E2 real-reward pilot stayed uniform entropy (300k eval). Oracle headroom exists (0.66% gain), but proxy matched oracle in only 4.95% of pools. | PPO optimization mechanics work, but real-reward candidate features lack sufficient correlation with true objective gain. | Showed that policy failure was due to feature informativeness and credit assignment, not optimizer bugs. | Highlights that proxy distance metrics poorly represent bottleneck pipeline latency. | Alternative graph neural network representations were not yet implemented. |
| `runs/kaggle-paper-baseline-calibration-2026-10-12` | Stage 10 (Baseline Calibration, Oct 12, 2026) | BS, RS, Fixed SA, ASA (1M proposals) | Calibrated fixed SA initial temperature $T_0$ from uphill deltas; ran 1,000,000 proposals on AlexNet-FC. | BS: 24.92 µs; RS: 23.94 µs (0.961 of BS); Fixed SA: 22.70 µs (0.911 of BS); ASA: 23.09 µs (0.927 of BS). Compute floor: 22.40 µs (0.899 of BS). | Reconstructed simulator cannot quantitatively match Figure 10's ratios (RS 0.59, SA 0.50) because VVA rate 1 op/cycle creates an insurmountable 89.9% compute bottleneck. | Evaluator calibration failure: rejected quantitative reproduction under rate-1 VVA assumption. | Proved that optimizer comparison against paper numbers was invalid due to unpublished hardware parameters. | Exact VVA execution throughput and buffer sizes were omitted in the TODAES 2020 paper. |
| `runs/kaggle-evaluator-vva-sensitivity-2026-10-12` & `runs/kaggle-logic-order-ablation-2026-10-12` | Stage 11 (VVA Sweep & Logic Order, Oct 12-13, 2026) | RS, Fixed SA, ASA across VVA rates (1 to 128 ops/cycle); Logic Index Ordering | Decoupled partition VVA rate from execution VVA rate (`--partition_vva_ops_per_cycle 1 --vva_ops_per_cycle 16`); ablated task index ordering. | At rate 16: Fixed SA/BS reached 0.5232 (close to paper 0.50). However, logical ordering ablated BS from 3.92 µs to 9.92 µs, moving SA/BS from 0.61 to 0.24. | BS-normalized ratios are dominated by arbitrary logical numbering conventions. Normalization against BS is uncalibrated without absolute latencies. | Paper-normalized latency reduction percentages cannot serve as a reliable scientific yardstick. | Contradicts all earlier assumptions that matching Figure 10 percentages proved reproduction. | The authors' exact indexing convention cannot be deduced from the paper text. |
| `runs/kaggle-rl-fixed-sa-gate-2026-10-13` (Short, Extended, Adaptive) | Stage 12 (Fixed-SA + RL Integration, Oct 13, 2026) | Fixed SA + PPO Proposals, Adaptive SA + PPO Proposals | Integrated learned proposals inside Fixed SA ($\gamma=0, \lambda=0$ contextual bandit reward); evaluated signed vs positive-only delta. | Signed delta: -0.07% loss. Positive delta: +0.63% win at 20k evals. Extended (100k, 3-seed): +0.105% mean win over control, beat ASA by 7.35%, lost to plain SA by 1.55%. Adaptive short: -0.36% loss. | Fixed SA provides stable optimization mechanics; PPO proposal learning provides weak positive signal but adds complexity; Adaptive controller degrades performance. | Closed RL+Adaptive SA as negative; retained RL+Fixed SA as primary development path. | Distinguishes between proposal selection and temperature schedule stability. | Why positive-only delta outperformed signed delta remained an open question. |
| `runs/kaggle-bandit-vgg16-fc-*` & [`docs/PROJECT_STATE.md`](PROJECT_STATE.md) | Stage 13 (Contextual-Bandit Validation, Oct 14, 2026) | Contextual Bandit Fixed SA, Focused Uniform Control, Fixed SA, ASA, RS | Online MLP regressor predicting signed objective delta with $\epsilon$-greedy exploration ($1.0 \rightarrow 0.5$), 50% bottleneck-focused pools. | AlexNet-FC held-out (seeds 3-5): won 3/3 pairs (+1.965% mean, 95% CI [1.01, 2.92]%). VGG16-FC held-out (seeds 3-5): won 3/3 pairs (+1.637% mean, 95% CI [0.34, 2.94]%). Combined VGG16 6-seed: won 6/6 (+1.946%, sign test $p=0.031$). | Contextual-bandit proposal ranker demonstrates positive matched-control evidence on tested workloads; further held-out and longer-budget confirmation is required. | Replaces PPO with a lightweight, stable 1-step contextual bandit formulation. | Resolves earlier stability issues by discarding multi-step Bellman value estimation. | Attribution between learned scoring and hand-built `proxy_gain` feature requires further testing. |
| [`docs/IMP_README.md`](IMP_README.md) & [`README.md`](README.md) | Stage 14 (Present Cleaned-Up State, Oct 14, 2026) | Contextual Bandit Fixed SA, BS, RS, SA, ASA | Comprehensive repository reorganization, backward-compatibility shims, test runners, clean branch hygiene. | All 21 non-torch tests pass cleanly; CLI help and runners verified operational. | Cleaned codebase maintains full research history, preserves all negative evidence, and isolates active RL+SA method. | All obsolete DDPG and PPO paths marked as historical or experimental ablations. | Unifies documentation across historical iterations. | External peer-reviewed publication requires broader multi-workload benchmark suite. |

---

## 1. Original Goal

The primary objective of this project was an honest, independent reproduction of the research presented by Nan Wu, Farhad Deng, Yuan Xie, et al. in their seminal paper:

> **"Core Placement Optimization for Multi-chip Many-core Neural Network Systems with Reinforcement Learning"**  
> *ACM Transactions on Design Automation of Electronic Systems (TODAES)*, 2020. DOI: [10.1145/3418498](https://doi.org/10.1145/3418498).

### Intended Simulator and Hardware Target
The paper addresses the core placement problem for multi-chip many-core accelerators executing deep neural network (DNN) inference. The target architecture specified in Table 1 of the paper consists of:
- **System Scale**: A multi-chip module (MCM) comprising a $4 \times 4$ array of chips (16 chips total).
- **Core Density**: Each chip contains a $16 \times 16$ 2D mesh of processing cores (256 cores per chip), yielding **4,096 total physical cores**.
- **Core Microarchitecture**: Processing cores operating at 400 MHz, equipped with vector-matrix multiplication (VMM) units (128 MACs/cycle), vector-vector accumulation (VVA) units, and a 64 KB dedicated local weight SRAM buffer.
- **Interconnect Topology**: A hierarchical Network-on-Chip (NoC). On-chip communication travels over a 2D mesh with 64 GB/s link bandwidth per core. Inter-chip communication traverses specialized global routing switches (GRS) and inter-chip links providing 100 GB/s bandwidth.
- **Latency Hierarchy**: On-chip communication hop latency is normalized to $\alpha = 1.0$ cycle, whereas off-chip communication crossing chip boundaries incurs a hierarchical penalty ratio of $\beta = 5.0$ to $10.0$ cycles per hop.

### Optimization Objective
In contrast to traditional spatial mapping frameworks that minimize aggregate communication energy (summed wirelength or total hop count), the paper targets a **pipelined streaming execution model**. The optimization objective is defined by Equation 4 of the paper:
$$\min_{P} L(P) = \min_{P} \max_{k} T(k \mid P)$$
where $L(P)$ is the system latency under placement $P$, determined strictly by the bottleneck pipeline stage $k$. Here, $T(k \mid P) = T_{\text{comp}}(k) + T_{\text{comm}}(k \mid P)$ represents the sum of computation and communication latency within stage $k$. The throughput of a streaming multi-chip accelerator is bounded by its slowest pipeline stage; placing non-communicating tasks closer together yields no throughput benefit if the critical path remains bottlenecked elsewhere.

### Expected Reproduction Milestones
The paper reported dramatic improvements over classical mapping heuristics on full-scale DNN workloads:
- **AlexNet**: Reported 50.5% latency reduction compared to a sequential baseline (BS), 38.4% reduction compared to random search (RS), and 18.6% reduction compared to simulated annealing (SA).
- **VGG16**: Reported 48.2% latency reduction over BS, 35.1% over RS, and 16.4% over SA.
- **ResNet50**: Reported 42.1% latency reduction over BS, 31.8% over RS, and 14.2% over SA.
- **Routing Locality**: Figure 11 reported a $4.5\times$ reduction in traffic-weighted average hop counts on AlexNet fully-connected layers compared to sequential placement.

The initial goal of the repository was to build or reconstruct the missing simulator, faithfully implement the DDPG algorithm described in Section 3, achieve comparable performance gains, and subsequently explore architectural extensions such as attention mechanisms and graph neural network encoders.

---

## 2. Starting Repository

The project began by cloning an open-source GitHub repository: `WOOSHIK-M/Core_Placement_with_Reinforcement_Learning` (commit `5745429`).

### Initial Codebase Contents and Assumptions
The starting repository was fundamentally different from the system described in the TODAES 2020 paper:
1. **Single-Chip Netlists**: The environment mapped static logic netlists (provided as ASCII lookup tables such as `toy_lut.txt`) onto a single $4 \times 4$ or $8 \times 8$ mesh chip.
2. **Algorithm**: It implemented discrete Proximal Policy Optimization (PPO) using a multi-layer perceptron (MLP) policy (`actor.py`, `agents.py`) that predicted $(x, y)$ coordinates on a 2D grid.
3. **Objective Function**: The reward function in `env/reward.py` computed the total communication cost as the sum of Manhattan distances over all communicating netlist pairs:
   $$C_{\text{comm}} = \sum_{i, j} \text{Traffic}(i, j) \cdot (|x_i - x_j| + |y_i - y_j|)$$
   It also included artificial deadlock penalty terms and Euclidean distance heuristics.
4. **Collision Handling**: Overlapping core assignments were resolved using a radial circular search (`detect_circle`) that found the nearest vacant grid slot using squared Euclidean distance.

### Limitations and Early Beliefs
Initial project records (e.g., [`docs/archive/prev_version_readmes/README-for_orignal+initialchanges.md`](archive/prev_version_readmes/README-for_orignal+initialchanges.md)) show that the code was initially patched for modern Python 3.10 and NumPy compatibility (resolving `np.int` deprecations and tuple unpacking errors in `CoreMapper.step`). 

At this stage, it was believed that:
- The single-chip PPO netlist placement algorithm was operational and could be trivially scaled up to multi-chip systems.
- The objective was simply minimizing global communication distance.
- Reinforcement learning was already "working" because training loops terminated with positive rewards on the toy lookup table.

In reality, the starting repository lacked every essential component of the 2020 multi-chip paper: it had no multi-chip topology, no hierarchical interconnect model, no layer partitioning into VMM and VVA logic cores, no pipeline bottleneck latency evaluation, and no DDPG implementation.

---

## 3. Initial DDPG Work

Between commits `4329fa3` and `41f9d02` (late August to early September 2026), the repository underwent a major rewrite to introduce the multi-chip environment and the paper's DDPG framework.

### Implementation of the Multi-Chip DDPG Framework
1. **Multi-Chip Topology (`src/multi_chip_topology.py`)**: Implemented a hierarchical coordinate system representing a grid of chips, each containing a 2D mesh of cores. Inter-chip communication hops were penalized relative to on-chip hops by a configurable factor ($\alpha = 1.0, \beta = 5.0$).
2. **Channel-Partitioned DNN Workload Extraction (`extract_model_task_graph`)**: Added PyTorch forward hooks to inspect torchvision models (`AlexNet`, `VGG16`, `ResNet50`) and partition convolutional and linear layers into VMM and VVA logic cores according to Section 3.1.1 of the paper.
3. **Continuous DDPG Agent (`src/run_multi_chip.py`)**: Implemented an actor-critic DDPG pipeline (`DDPGAgent`, `Actor`, `Critic`). To handle multi-core placement, the actor emitted continuous coordinates in $[-1, 1]^{2z}$ representing a batch of $z$ logic cores placed per step. These outputs were scaled, floored to integers, and repaired via Manhattan distance if the target core was occupied.
4. **Sparse Baseline Reward**: Replaced intermediate step rewards with the paper's terminal sparse reward:
   $$r_T = \sqrt{B} - \sqrt{L(P)}$$
   where $B$ was the latency of a random-search baseline placement computed prior to training.
5. **Performance Engineering**:
   - Replaced full-matrix $O(N^2)$ communication scans with a cached sparse adjacency list (`pipeline_latency`), achieving a **$19.4\times$ speedup** on 906-task workloads.
   - Introduced action batching ($z=3$) and gradient update decimation (`--train_every 5`) to prevent hours of compute overhead per episode.

### The Initial "46.8% AlexNet Improvement"
On September 6, 2026, an extensive CPU training run was executed on AlexNet (configured with 1,445 logic cores across 4,096 physical cores) over ~870 episodes (commit `bc94bea`). The experiment yielded the following metrics:
- Random Search Baseline ($B$): $12,697,984$ proxy cost units
- Best DDPG Placement Cost: $6,749,184$ proxy cost units
- **Reported Gain**: **~46.8% reduction in latency** compared to the random-search baseline.

This result was documented in [`docs/archive/prev_version_readmes/claude_compressed_readme_6thsep.md`](archive/prev_version_readmes/claude_compressed_readme_6thsep.md) and celebrated as strong preliminary evidence that DDPG was learning effectively and nearing the paper's 50.5% headline figure. Checkpoints were saved to `src/alexnet_ckpt.pt` (commit `41f9d02`), and terminal logs were saved to `temp.txt`.

---

## 4. DDPG Investigation

Despite the apparent success of the 46.8% reduction, closer inspection of the training dynamics revealed severe anomalies.

### The Oscillation Anomaly: Current Cost vs. Best Cost
In a properly converging reinforcement learning agent, the policy's rollout cost should tighten around the historical best cost as exploration noise decays. However, across every recorded run:
- While `Best Cost` dropped monotonically from 12.7M to 6.7M, `Current Cost` (the cost of the placement generated at episode $t$) continued to oscillate wildly between 8.0M and 14.5M throughout all 870 episodes.
- Even in late episodes where exploration noise was minimal, the agent never reproduced its best placement.

```
Cost
 ^
 |  /\    /\        /\  /\    /\        Current Cost (wildly oscillating)
 | /  \  /  \  /\  /  \/  \  /  \  /\
 |/____\/____\/__\/________\/____\/__\
 |----------------------------------- Best Cost (stagnant after early lucky drop)
 +----------------------------------------> Episodes
```

### The 3,000-Placement Kaggle Audit (October 8, 2026)
To investigate the issue rigorously, an extensive 3,000-placement AlexNet-CONV training run was launched on Kaggle with paired untrained controls and explicit zero-noise deterministic evaluation ([`docs/RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md)). The audit produced devastating empirical findings:
1. **Trained vs. Untrained Control**:
   - Best placement found by trained DDPG: **37.6356 microseconds**.
   - Best placement found by an **untrained, random-weight DDPG agent**: **37.4466 microseconds**.
   - The trained agent was *worse* than the untrained control.
2. **Massive Collision Repair Rates**:
   - Out of 183 logic cores, an average of **176 to 180 cores required heuristic collision repair** on every single episode.
   - The continuous actor's outputs were saturating at the extreme boundaries (`-1.0` or `+1.0`). Consequently, the actor intended only 3 to 7 unique physical cores for the entire 183-task network.
   - The remaining 176 tasks were placed entirely by the deterministic Manhattan spiral heuristic (`detect_nearest_free_core`), completely bypassing the neural network.
3. **Critic Divergence**:
   - Critic loss exploded to approximately **$1.48 \times 10^{12}$**.
   - Because the executed placements were determined by collision repairs rather than the actor's intended outputs, the transitions stored in the replay buffer $(s_t, a_t, r_t, s_{t+1})$ misaligned the actor's actions with the environmental transitions, causing complete policy collapse.
4. **Deterministic Policy Collapse**:
   - When exploration noise was turned off to evaluate the learned policy deterministically, the policy placed tasks in a degenerate clump, yielding latencies far worse than random search.

### Conclusion of the Investigation
**The apparent 46.8% improvement reported in Stage 4 was an artifact of random exploration.** The continuous actor had learned nothing; it was merely acting as a random number generator that fed a deterministic collision-repair heuristic. The best placements were discovered entirely by chance during exploration noise, not by policy optimization.

---

## 5. Evaluator and Paper-Reproduction Corrections

Following the discovery of DDPG's failure, research pivoted to verifying whether the underlying environment and evaluation metrics were mathematically sound (commit `c655138` and [`docs/RECONCILIATION.md`](RECONCILIATION.md)).

### Flaws in the Initial Evaluator
An audit of the early simulator revealed critical discrepancies against the TODAES 2020 paper:
1. **Flat Objective on CONV Layers**:
   - The initial workload extractor partitioned AlexNet CONV2 into a single output group $(M=1, N=61)$, creating a single massive VVA accumulation task.
   - Because all 61 partial sums fed into one core, that single VVA task took $5.2488$ milliseconds of compute time regardless of where it was placed.
   - As a result, random placements across the entire chip varied by only **$0.0015\%$** (a placement-insensitive flat objective). The agent was trying to optimize an objective that had virtually zero placement headroom.
2. **Missing Physical Units**:
   - The original code evaluated costs in arbitrary abstract "proxy units" rather than physical execution seconds, clock cycles, and byte volumes.
3. **Unpublished Routing and Link Contention**:
   - The paper assumed dimension-ordered XY routing and global routing switches (GRS), but omitted exact routing algorithms and link contention models.
4. **Conflating CONV and FC Placements**:
   - The paper explicitly notes that CONV and FC layers occupy separate, dedicated hardware regions. The initial code attempted to map entire networks onto a single unconstrained grid.

### Implemented Corrections (The Reconciled Engine)
Between September 10 and September 30, 2026, the evaluation engine was completely rebuilt:
- **Deterministic Workload Partitioning (`src/compute_model.py`)**: Built an exact dynamic-programming partitioner matching the paper's Figure 6 aggregate core counts (AlexNet: 183 CONV, 932 FC; VGG16: 1,024 CONV, 1,924 FC; ResNet50: 512 CONV, 37 FC). Enforced the 64 KB weight-tile SRAM capacity constraint and balanced estimated VMM and VVA cycle times.
- **Physical Timing Model (`paper_pipeline`)**: Latencies were converted to physical seconds based on 400 MHz core clocks, 128 MACs/cycle VMM throughput, and Table 1 bandwidths.
- **Dimension-Ordered XY Gateway Routing (`paper_xy`)**: Implemented dimension-ordered X-then-Y mesh routing within chips, routing through lower-left peripheral gateway switches for inter-chip traffic, and calculating shared directed-link contention.
- **Placement Sensitivity Preflight**: Added an automated preflight check that samples random valid placements and halts optimization if the relative objective span is below 0.1%.
- **Corrected Headroom**: On the corrected pure-model AlexNet partitions, 256 random placements demonstrated a healthy **16.34% objective span on CONV** and **4.05% on FC**, confirming that the evaluator was now placement-sensitive.

---

## 6. Simulated Annealing and Adaptive Simulated Annealing

With a verified, physically grounded evaluator in place, focus shifted to establishing robust classical optimization baselines against which any reinforcement learning method must be judged.

### Algorithms Implemented
1. **Sequential Baseline (`bs`)**: Deterministically places tasks in topological order following physical core coordinates in chip-major, row-major order.
2. **Random Search (`random`)**: Uniformly samples $N$ valid placement configurations and returns the minimum bottleneck latency.
3. **Fixed Simulated Annealing (`sa`)**:
   - Follows a geometric cooling schedule $T_{k+1} = \alpha T_k$ with $\alpha = 0.99$.
   - **Correction of Temperature Units**: The original code used hardcoded absolute temperatures ($T_0 = 100, T_{\text{end}} = 0.1$) against a seconds-valued objective ($10^{-5}$ s), which caused the Metropolis acceptance probability $\exp(-\Delta C / T)$ to evaluate to $1.0$ for virtually every uphill move. This bug was repaired by dynamically measuring $T_0$ from an initial sample of uphill objective deltas to achieve an initial acceptance rate of exactly 80%.
   - Perturbations relocate a randomly selected logic core to a randomly selected vacant physical core on the grid.
4. **Adaptive Simulated Annealing (`asa`)**:
   - Adjusts temperature dynamically over 100-evaluation windows to track an optimal acceptance rate of 30%.
   - If acceptance falls too low, cooling slows or reheats; if acceptance is too high, cooling accelerates.
   - Adjusts perturbation neighborhood radius dynamically based on stagnation.

### Empirical Baseline Results (AlexNet FC, 1 Million Evaluations)
In October 2026, an extensive 1,000,000-proposal baseline calibration was executed on AlexNet FC (`80aec4f`):
- **Sequential Baseline (BS)**: $24.9203$ microseconds
- **Random Search (RS)**: $23.9373$ microseconds ($0.9606 \times \text{BS}$)
- **Adaptive Simulated Annealing (ASA)**: $23.0933$ microseconds ($0.9267 \times \text{BS}$)
- **Fixed Simulated Annealing (SA)**: **$22.7040$ microseconds** ($0.9111 \times \text{BS}$)

Fixed SA beat Random Search by **5.15%** and proved to be the single strongest classical optimizer in the repository, outperforming Adaptive SA by 1.71%. Across all multi-seed benchmarks, Fixed SA established the true performance ceiling.

---

## 7. PPO Investigation

Recognizing that continuous DDPG was structurally mismatched for discrete placement, the project explored Proximal Policy Optimization (PPO).

### Constructive Categorical Masked PPO
In early October 2026, an autoregressive masked categorical PPO agent was implemented (`--algo ppo_masked`, commit `e3576b0`):
- Instead of predicting continuous coordinates, the agent placed tasks sequentially.
- At each step $i$, an invalid-action mask set the logits of all occupied physical cores to $-\infty$, guaranteeing **zero illegal placements and zero collision repairs**.
- Minibatch PPO updates utilized clipped surrogate objectives:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$

### Results of Masked PPO
1. **AlexNet CONV (Single-Chip, 5 Seeds, 3,000 Placements)**:
   - Trained PPO achieved a late deterministic latency of **42.6118 microseconds**, beating its frozen-policy control (**45.3588 microseconds**) across all 5 seeds (a statistically resolved **6.06% policy improvement**, 95% CI [1.10, 4.40] µs).
   - *However*, best retained latency only improved by 1.26% (39.9989 µs vs 40.5094 µs), showing that policy learning was modest relative to search exploration.
2. **AlexNet FC (Multi-Chip, 4 Chips, 932 Tasks)**:
   - Tested on multi-chip AlexNet FC across 4,101 evaluations (`b803ff5`).
   - Late deterministic improvement dropped to a negligible **0.163%** in short runs and **0.558%** in 3,000-placement runs.
   - In best placement cost, PPO achieved 24.1080 µs, which was **worse than Random Search (24.1155 µs) and substantially worse than Fixed SA (23.4318 µs)**.
   - Constructive PPO placed early tasks without knowing where late tasks would land; error accumulated over the 932-step sequence.

### PPO-Guided ASA (Proposal Ranking)
Rather than placing cores from scratch, PPO was repurposed to act as a **learned neighbor proposal generator** inside simulated annealing (`--algo ppo_asa`, commits `1c5c633` to `bdcd801`):
- At each annealing step, a candidate pool of $K=16$ legal swap/relocation moves was generated.
- A permutation-equivariant neural network scored the 16 candidates; the candidate with the highest logit was proposed to the SA controller.
- The true objective was evaluated *only* for the selected candidate.
- Simulated annealing retained control over the temperature schedule and Metropolis acceptance.

#### The 1.25 Million Proposal Decision Run (`bdcd801`, October 11, 2026)
To eliminate sample insufficiency as a confounder, a massive Kaggle run was executed: 1,250,000 proposals, 1,250 PPO update epochs, and a 10-seed frozen holdout on AlexNet FC (6.47 GPU hours):
- **Trained PPO-ASA**: $23.5026$ microseconds
- **Uniform Proposal Control**: $23.5072$ microseconds
- **Plain Uninterrupted ASA**: **$23.4894$ microseconds**
- **Outcome**: The trained policy won only 3 of 10 holdout seeds against uniform control, tied 5, and lost 2. Policy entropy converged to the uniform maximum ($\ln 16 \approx 2.77$).

The candidate-scoring PPO-ASA architecture was definitively rejected and closed as a documented negative result.

---

## 8. Oracle and Supervised-Ranker Experiments

To determine whether the failure of PPO-ASA was caused by RL credit assignment or a fundamental absence of learning signal in the 16-candidate pools, the project conducted **oracle diagnostic experiments** ([`docs/archive/CLAUDE_AUDIT_RESPONSE.md`](archive/CLAUDE_AUDIT_RESPONSE.md) and commits `8fe5da4`, `164edfe`, `91a70bc`).

### The Oracle Headroom Audit (E0)
Every candidate in 8,000 pools of 16 legal moves was evaluated against the true placement objective:
- **Headroom Exists**: The best candidate in a pool yielded an average **0.6639% latency reduction** over a uniformly chosen candidate.
- **Improving Moves are Rare**: Across annealing trajectories, **58.56% of proposed moves were objective-neutral** ($\Delta C = 0$), and only **39.04% of pools contained at least one improving move**.
- **Handcrafted Proxy Failure**: A cheap routed-distance proxy feature (`proxy_gain`) matched the true oracle-best candidate in only **4.95% of pools**—worse than random guessing (1 in 16 = 6.25%).

### Supervised Candidate Ranking and Trajectory Leakage
Next, an offline dataset of exact objective gains was exported to train a supervised listwise ranking policy:
1. **Random Pool Split (Provisional Win, `8fe5da4`)**:
   - Pools were randomly split into 80% train and 20% validation.
   - The supervised ranker captured **86.02% of best-of-pool advantage** and beat ASA in 4 of 5 holdout seeds (0.18% advantage).
2. **The Trajectory Leakage Discovery (`164edfe`)**:
   - An audit revealed that random pool splitting leaked correlated physical layouts between adjacent annealing steps across train and test sets.
   - The protocol was corrected to use **Leave-One-Chain-Out cross-validation** (10 independently initialized annealing chains; 8 for training, 2 held out).
   - **Result on Held-Out Chains**: Validation oracle capture collapsed from 86.02% to **3.36%**. The 95% confidence interval for selected advantage crossed zero ([-0.0015, +0.0059] µs).
   - When tested on fresh annealing trajectories, the supervised model had memorized specific grid coordinates and failed to generalize.
3. **The Improvement-Only Target Gate (`91a70bc`)**:
   - Post-hoc analysis showed that predicting signed delta penalized neutral moves heavily. Re-training on an improvement-only utility target ($\max(\Delta C, 0)$) over 20 fresh 4,100-proposal chains achieved **7.03% late-chain oracle capture**, but failed the prespecified 10% gate.

**Key Lesson**: While weak ranking signals existed, complex neural networks trained via policy gradients or supervised listwise ranking failed to generalize across unseen annealing trajectories.

---

## 9. RL-Guided SA

Synthesizing the lessons of previous failures, the project formulated a radical simplification: **Contextual-Bandit-Guided Fixed Simulated Annealing** (commits `6a4c2fb` through `1e9dee2`, October 13–14, 2026).

### Algorithmic Formulation
The contextual bandit framework completely discards multi-step Bellman value estimation, discount factors ($\gamma = 0$), and generalized advantage estimation ($\lambda = 0$). It treats candidate proposal ranking as an **online 1-step contextual bandit problem**:

```
Current Annealing State S_t ──> Generate 16 Legal Candidates
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
50% Bottleneck-Focused Moves                       50% Global Uniform Moves
(Tasks on bottleneck pipeline stage)               (Random tasks across graph)
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     ▼
                      Extract 9 Candidate Features
                                     │
                                     ▼
                      MLP Regressor (ProposalBandit)
                         Predicts Signed Gain ΔC
                                     │
                                     ▼
                      ε-Greedy Selection (1.0 → 0.5)
                                     │
                                     ▼
                      True Objective Evaluation L(P)
                                     │
                                     ▼
                      Metropolis Acceptance (Fixed SA)
                                     │
                                     ▼
                      Online Loss: (ΔC_pred - ΔC_true)²
```

### Components and Separation of Concerns
1. **What Learning Controls**: The bandit controls **only candidate proposal selection**. Given a pool of 16 legal moves, it predicts the immediate signed objective change $\Delta C = C_{\text{current}} - C_{\text{candidate}}$ and selects the move with probability $1 - \epsilon$.
2. **What SA Controls**: Fixed Simulated Annealing retains complete authority over **move acceptance and temperature**. The selected move is subjected to the standard Metropolis criterion:
   $$P_{\text{accept}} = \min\left(1, \, \exp\left(\frac{-\Delta C}{T_k}\right)\right)$$
   using the geometrically cooled schedule ($T_{k+1} = 0.99 T_k$) calibrated from initial uphill deltas.
3. **Candidate Pool Structuring (`--ppo_asa_focus_fraction 0.5`)**:
   - 8 candidates are anchored on tasks located in the slowest pipeline stage, guaranteeing that the bottleneck task is moved.
   - 8 candidates are sampled uniformly across all tasks to maintain global exploration.
4. **Candidate Feature Vector (9 Dimensions)**:
   - `in_bottleneck` (binary), `is_vva` (binary), `task_weight` (normalized MACs), `stage_task_fraction` (fraction of stage operations), `predecessor_hops_delta`, `successor_hops_delta`, `weighted_hops_delta`, `same_chip_delta`, and `proxy_gain`.
5. **Causal Control**: Every experiment is evaluated against a **Focused-Uniform Frozen Control** (`--ppo_asa_disable_learning`), which executes the identical candidate generator, bottleneck focusing, annealing schedule, and random seed, but selects candidates uniformly at random (zero weights).

### Iterative Bandit Evolution
- **Version 1 (Positive-Only Targets, `6a4c2fb`)**: Failed (-8.04% loss). Positive-only targets and rapid $\epsilon$ decay ($1.0 \rightarrow 0.1$) caused severe overprediction of gains and harmful exploitation.
- **Version 2 (Signed Targets, Full-Run Decay, `a52c073`)**: Succeeded on short gate (+0.12% win over control).
- **Version 3 (Epsilon Schedule Bug Fix, `0608cf0`)**: Fixed an exploration bug where replay eviction stopped $\epsilon$ decay at 0.75; ensured smooth linear decay to $\epsilon = 0.50$ across the full 100,000-proposal horizon.

---

## 10. Workload Generalization

To ensure that the contextual bandit was not overfitting to a single network, the frozen method was evaluated across multiple distinct DNN workloads and random seeds.

### 1. AlexNet Fully-Connected Layers (AlexNet-FC)
- **Workload**: 932 logic cores partitioned across 4 chips ($16 \times 16$ cores/chip = 1,024 cores total).
- **Exploratory Seeds (0, 1, 2)**: Won 2 of 3 pairs (+0.664% mean improvement over focused-uniform control).
- **Held-Out Seeds (3, 4, 5)**: Won **3 of 3 pairs** (+1.965% mean improvement over control, paired 95% CI [1.008%, 2.922%]).
- **Combined 6-Seed Result**: Beat focused-uniform control in **5 of 6 seeds** (+1.314% mean advantage). Effectively tied with plain Fixed SA (-0.161% difference, CI crossing zero) and beat plain ASA by 8.572%.

### 2. AlexNet Convolutional Layers (AlexNet-CONV)
- **Workload**: 183 logic cores partitioned onto 1 chip (256 cores total).
- **Short Gate (20k evals, seed 0)**: Beat control by +1.136%, Fixed SA by +2.246%, and ASA by +4.394%.
- **Extended Gate (100k evals, seeds 0, 1, 2)**: All algorithms (Learned RL+SA, Frozen Control, and Fixed SA) converged to the **exact same latency floor of 37.02117 microseconds** in every seed.
- **Conclusion**: On small single-chip workloads with 183 tasks, candidate pooling accelerated convergence, but the final quality ceiling was reached by both learned and classical methods alike (a quality tie).

### 3. VGG16 Fully-Connected Layers (VGG16-FC)
- **Workload**: 1,924 logic cores partitioned across 8 chips ($16 \times 16$ cores/chip = 2,048 cores total). A much larger, heavily communication-constrained multi-chip problem.
- **Exploratory Seeds (0, 1, 2)**: Won **3 of 3 pairs** (+2.255% mean improvement over control, 95% CI [1.107%, 3.403%]).
- **Held-Out Seeds (3, 4, 5)**: Won **3 of 3 pairs** (+1.637% mean improvement over control, 95% CI [0.336%, 2.938%]).
- **Combined 6-Seed Result**: Beat focused-uniform control in **6 of 6 seeds** (+1.946% mean improvement, 95% CI [1.362%, 2.530%], exact two-sided sign test $p = 0.03125$).
- **Comparison to Baselines**: Beat Random Search by 22.41%, beat plain ASA by 4.605%, and beat plain Fixed SA by 0.515% (4 of 6 seeds won, CI [-0.20%, +1.23%]).

---

## 11. Important Failures and Lessons

The project’s history is defined by rigorous negative results that prevented false scientific claims:

```
[Continuous DDPG] ──> Fails: Discrete coordinate saturation & critic explosion (1.48e12)
        │
[Guided DDPG] ──────> Fails: Replay demonstrations fail to prevent late deterministic collapse
        │
[Constructive PPO] ─> Fails: 932-step autoregressive error accumulation on multi-chip FC
        │
[PPO-Guided ASA] ───> Fails: 1.25M proposals; shared encoders converge to uniform entropy
        │
[Supervised Rank] ──> Fails: 86% validation capture collapses to 3.3% on held-out chains
        │
[Contextual Bandit] ─> PASSES: Lightweight 1-step regression + Fixed SA cooling succeeds
```

1. **Continuous Action Spaces for Discrete Core Mapping**: Continuous policy gradient methods (DDPG) are fundamentally ill-suited for discrete spatial mapping. Discretization via flooring creates a many-to-one mapping, and heuristic collision repair breaks the Bellman Markov property, guaranteeing policy divergence.
2. **The "Best Cost" Illusion**: A decreasing historical best cost curve is completely meaningless in reinforcement learning if unaccompanied by zero-noise deterministic policy rollouts. Exploration noise alone will discover good placements that the neural network cannot replicate.
3. **The Flat Objective Trap**: In hardware simulators, partition imbalances (such as a single massive accumulation node) can create a placement-independent compute bottleneck, reducing objective sensitivity to near zero. Evaluators must be tested for variance across random layouts before running learning agents.
4. **Trajectory Leakage in Annealing Data**: Splitting optimization trajectories randomly into train and test sets leaks spatial layout correlations, producing spectacular but spurious offline accuracy (86%) that evaporates on fresh optimization runs (3%).
5. **Adaptive SA Instability**: Complex reheating controllers in Adaptive SA frequently destabilize convergence compared to simple, calibrated geometric Fixed SA schedules. Adding reinforcement learning to an unstable adaptive controller compounds failure.
6. **Simplicity Over Complexity**: Discarding multi-step RL (PPO actor-critics) in favor of a 1-step online contextual bandit regressor paired with a rock-solid Fixed SA cooling engine produced the project's first reproducible, cross-workload positive evidence.

---

## 12. Present Verified State

The current repository on branch `cleaned-up` (commit `6fa6e84`) maintains strict scientific boundaries:

### Verified Implementation Capabilities
- **Deterministic Workload Partitioning**: Validated FX extraction matching exact Figure 6 task counts for AlexNet, VGG16, and ResNet50 with 64 KB weight buffer constraints.
- **Hierarchical Interconnect Evaluator**: Verified dimension-ordered XY routing, peripheral gateways, and shared directed-link contention modeling.
- **Robust Classical Baselines**: Deterministic sequential baseline (BS), uniform Random Search (RS), calibrated Fixed Simulated Annealing (Fixed SA), and Adaptive Simulated Annealing (ASA).
- **Regression and Smoke Test Suites**: `tests/run_all_tests.py` and `src/test_reconciliation.py` providing 21 non-torch and 17 torch regression invariants.

### Promising but Preliminary Results
- **Contextual-Bandit Fixed SA**: Demonstrates positive matched-control evidence on tested workloads; across 6 seeds on AlexNet-FC and 6 seeds on VGG16-FC, learned candidate ranking achieved an average **~1.3% to ~1.9% latency reduction** over matched focused-uniform controls under identical evaluation budgets. Further held-out and longer-budget confirmation is required before making a peer-reviewed publication claim.

### Documented Negative Results
- **Continuous DDPG**: Formally rejected due to critic divergence and collision-repair policy collapse.
- **Guided DDPG with ASA Demonstrations**: Formally rejected after 5-seed paired ablation.
- **Constructive Masked PPO**: Rejected for multi-chip scaling due to sequence error accumulation.
- **PPO-Guided ASA (Shared & Decoupled)**: Formally rejected after 1.25M-proposal decision runs converged to uniform entropy.

### Claims That Remain Unresolved
- **Quantitative Reproduction of Paper Percentages**: The TODAES 2020 paper's headline numbers (50.5% AlexNet reduction) cannot be quantitatively matched because the authors omitted critical simulator parameters (VVA throughput, internal buffer stalls, and exact GRS routing).
- **BS Normalization Dependency**: Latency normalized to the sequential baseline is heavily sensitive to unpublished VMM/VVA logical indexing conventions.

---

## 13. Current Research Direction

The active research program focuses exclusively on **Contextual-Bandit-Guided Simulated Annealing**:

```text
Current State:
  Online Contextual Bandit (Signed ΔC MLP) + Fixed SA Cooling + 50% Bottleneck Pools
         │
         ├── Verified: Wins 6/6 seeds on VGG16-FC over focused-uniform control (+1.95%)
         ├── Verified: Wins 5/6 seeds on AlexNet-FC over focused-uniform control (+1.31%)
         └── Verified: Identical 37.02 µs floor reached on AlexNet-CONV
```

### Why This Direction Was Selected
1. **Mathematical Stability**: It avoids multi-step Bellman instability, policy gradient variance, and critic divergence.
2. **Computational Tractability**: Training takes milliseconds per step, allowing full 100,000-proposal optimization runs to complete within practical cloud GPU budgets.
3. **Empirical Grounding**: It is the only learned method in the repository that has demonstrated positive, statistically resolved improvements over a rigorous causal control across multiple DNN workloads.

### Experiments Required Before Publication Claims
1. **Matched Proxy-Greedy Comparator (`bandit-proxy-short`)**: Evaluate whether the learned MLP is genuinely discovering complex multi-feature interactions, or whether simply selecting moves that maximize the hand-built `proxy_gain` feature achieves the same benefit without learning.
2. **20-Seed Large-Scale Confirmation**: Scale VGG16-FC and AlexNet-FC to 20 independent seeds to establish narrow, publication-grade confidence intervals.
3. **ResNet50 Cross-Workload Evaluation**: Test the frozen contextual bandit on the branched ResNet50 workload after upgrading graph extraction to handle skip-connection additions cleanly.
4. **Wall-Clock Overhead Profiling**: Characterize the wall-clock overhead of candidate feature extraction and MLP inference against the raw evaluation speed of classical Fixed SA.

---

## 14. Reproducibility and Evidence Index

### Key Repository Documents
- [`README.md`](../README.md): Master repository overview, setup, and CLI reference.
- [`docs/CODEBASE_GUIDE.md`](CODEBASE_GUIDE.md): Architectural layout, evaluator data flow, and new contributor guide.
- [`docs/IMP_README.md`](IMP_README.md): Critical developer rules, branch hygiene, and contextual bandit execution commands.
- [`docs/PROJECT_STATE.md`](PROJECT_STATE.md): Source of truth for benchmark numbers, statistical intervals, and hardware reconciliation details.
- [`docs/NEXT_STEPS.md`](NEXT_STEPS.md): Comprehensive 66-item experimental checklist and backlog.
- [`docs/RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md): In-depth literature audit, DDPG failure analysis, and VVA sensitivity review.
- [`docs/RECONCILIATION.md`](RECONCILIATION.md): Hardware parameters, timing formulas, and reconciliation against TODAES 2020.
- [`docs/archive/README.md`](archive/README.md): Indexed historical decision memos, comparative audits, and previous README versions.

### Key Scripts and Entry Points
- `src/run_multi_chip.py`: Primary optimization runner supporting BS, RS, SA, ASA, DDPG, PPO, and Contextual Bandit SA.
- `scripts/run_kaggle_rl_fixed_sa_gate.sh`: Canonical automation script for contextual bandit experiments on cloud GPUs.
- `scripts/run_kaggle_paper_baseline_calibration.sh`: 1,000,000-proposal classical baseline calibration suite.
- `scripts/run_kaggle_evaluator_vva_sensitivity.sh`: Parameter sweep diagnosing VVA throughput sensitivity.
- `tests/run_all_tests.py`: Unified test runner executing smoke and reconciliation unit tests.

### Important Archived Git Commits
- `5745429`: Initial upstream commit (`WOOSHIK-M` single-chip netlist repository).
- `4329fa3`: Introduction of multi-chip topology and continuous DDPG framework.
- `41f9d02`: `origin/main` commit archiving initial 46.8% AlexNet run and `alexnet_ckpt.pt`.
- `c655138`: Initial reconciliation commit introducing physical units and Kahn DAG staging.
- `3499336`: Five-seed AlexNet-CONV study exposing DDPG's failure against untrained controls.
- `e3576b0`: Implementation and validation of constructive masked categorical PPO.
- `bdcd801`: 1.25-million proposal sufficient-scale PPO-ASA decision run (negative result).
- `80aec4f`: 1-million proposal paper baseline calibration diagnosing VVA compute floor.
- `0341745`: Fixed-grid VVA execution rate sweep establishing communication-sensitive regimes.
- `168add0`: Implementation of positive-delta proposal learning in Fixed SA.
- `0608cf0`: Epsilon exploration repair for contextual bandit.
- `073f003`: First positive held-out validation of contextual bandit on AlexNet-FC (seeds 3–5).
- `bacc334`: Positive held-out validation of contextual bandit on VGG16-FC (seeds 3–5).
- `1e9dee2`: `origin/cpu` head adding VGG holdout records and proxy control.
- `6fa6e84`: `cleaned-up` branch head; repository reorganization, codebase guide, and post-cleanup corrections.

---

## 15. Timeline

| Stage | Date | Main Change | Experiment Executed | Key Measured Result | Decision Taken | Evidence Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Fork & CPU Port** | Summer 2026 | Forked `WOOSHIK-M`; patched for Python 3.10 and CPU. | Single-chip toy LUT run. | Finished without syntax errors; reward > 768. | Proceed to multi-chip paper implementation. | [`docs/archive/prev_version_readmes/README-for_orignal+initialchanges.md`](archive/prev_version_readmes/README-for_orignal+initialchanges.md) |
| **Stage 2: Multi-Chip Standup** | Late Aug 2026 | Added `multi_chip_topology.py`, channel partitioning for `SimpleCNN`. | Multi-chip DDPG on SimpleCNN (906 cores). | 40 episodes took ~10 min; massive scaling bottlenecks observed. | Profile and optimize pipeline evaluation. | [`docs/archive/prev_version_readmes/old_section_41-50_of_readme_by_gpt.md`](archive/prev_version_readmes/old_section_41-50_of_readme_by_gpt.md) |
| **Stage 3: Optimization & Scale** | Early Sept 2026 | Sparse adjacency cache ($19.4\times$ speedup), batched actions ($z=3$), `--train_every`. | SimpleCNN 10,000 episodes on CPU. | Best DDPG: 340k vs Baseline 444k (23.3% reduction). | Proceed to real torchvision workloads. | [`docs/archive/prev_version_readmes/README_before_GPTplus.md`](archive/prev_version_readmes/README_before_GPTplus.md) |
| **Stage 4: First AlexNet Full Run** | Sept 6, 2026 | Hook extraction for AlexNet (1,445 cores across 4,096 physical cores). | ~870 episodes AlexNet CPU training. | Best DDPG: 6.75M vs Baseline 12.7M (46.8% reduction). | Believed paper reproduction was nearly achieved; saved `alexnet_ckpt.pt`. | [`docs/archive/prev_version_readmes/claude_compressed_readme_6thsep.md`](archive/prev_version_readmes/claude_compressed_readme_6thsep.md), Commit `41f9d02` |
| **Stage 5: Evaluator Reconciliation** | Sept 10–30, 2026 | Exact Figure 6 core counts (183 CONV, 932 FC); physical units (seconds, cycles); Kahn DAG stages. | Evaluated objective sensitivity across random layouts. | Discovered earlier 2,080-ep CONV run had 0.001% span due to $M=1, N=61$ VVA bottleneck. Corrected grids yielded 16.3% span. | Invalidate all pre-reconciliation results; enforce sensitivity preflight. | [`docs/RECONCILIATION.md`](RECONCILIATION.md) |
| **Stage 6: Control Study & BN Fix** | Sept 30, 2026 | Paired untrained controls; fixed target network BatchNorm synchronization. | 5-seed AlexNet-CONV bounded diagnostic (12 ep). | Trained DDPG 38.24 µs vs Untrained Control 38.47 µs (0.59% diff, CI crosses 0). Seed 4 collapsed to 181 collisions. | DDPG gains do not prove learning; continuous actor saturates. | `runs/local-cpu-5seed-2026-09-30/STUDY.md`, `runs/NORMALIZATION_EXPERIMENTS.md` |
| **Stage 7: DDPG Audit & Rejection** | Oct 8, 2026 | Implemented Guided DDPG with permanent ASA demonstration replay. | 3,000-ep AlexNet-CONV Kaggle gate; 5-seed Guided DDPG ablation. | Standard DDPG critic loss exploded to $1.48 \times 10^{12}$; 176/183 collisions. Guided DDPG lost all 5 seeds to control (-0.35 µs). | Definitively reject continuous DDPG; pivot to discrete masked PPO. | [`docs/RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md), [`docs/archive/DDPG_REFERENCE_AUDIT.md`](archive/DDPG_REFERENCE_AUDIT.md) |
| **Stage 8: Masked PPO & PPO-ASA** | Oct 9–11, 2026 | Autoregressive Masked PPO; candidate-scoring PPO-guided ASA ($K=16$). | 5-seed CONV Masked PPO; 4,101-eval FC PPO; 1.25M-proposal PPO-ASA decision run. | Masked PPO achieved 6.06% policy win on single-chip CONV, but failed on multi-chip FC. 1.25M PPO-ASA won 3/10 seeds vs control; entropy stayed uniform. | Reject candidate-scoring PPO-ASA; close multi-step RL proposal path. | `runs/kaggle-ppo-asa-v1250-2026-10-11/STUDY.md`, Commit `bdcd801` |
| **Stage 9: Oracle & Supervised Ranker** | Oct 11, 2026 | Decoupled PPO networks; pool oracle audit; Leave-One-Chain-Out supervised ranker. | 8,000 candidate pools evaluated; supervised training on exact gains. | Oracle headroom verified (0.66%), but Leave-One-Chain-Out validation capture collapsed to 3.36% (CI crossed 0). | Complex supervised/PPO rankers fail to generalize across trajectories. | [`docs/archive/CLAUDE_AUDIT_RESPONSE.md`](archive/CLAUDE_AUDIT_RESPONSE.md), Commits `8fe5da4`, `164edfe` |
| **Stage 10: Baseline Calibration** | Oct 12, 2026 | Calibrated Fixed SA $T_0$ from uphill deltas; ran 1M proposals on AlexNet-FC. | 1,000,000 proposals for BS, RS, Fixed SA, ASA. | BS 24.92 µs; RS 23.94 µs (0.96 BS); Fixed SA 22.70 µs (0.91 BS). Compute floor is 22.40 µs (0.899 BS). | Evaluator rejected as quantitative reproduction of Figure 10 due to 1 op/cycle VVA assumption. | `runs/kaggle-paper-baseline-calibration-2026-10-12`, Commit `80aec4f` |
| **Stage 11: VVA Sweep & Logic Order** | Oct 12–13, 2026 | Decoupled partition VVA rate from execution VVA rate; swept rates 1 to 128; ablated task indexing. | Matched 20k sweeps on AlexNet-FC. | Rate 16 Fixed SA/BS reached 0.5232 (near paper 0.50). Logical ordering ablated BS from 3.92 µs to 9.92 µs. | BS-normalized ratios are uncalibrated without absolute latencies; use paired controls. | `runs/kaggle-evaluator-vva-sensitivity-2026-10-12`, Commits `0341745`, `7ee8bc2` |
| **Stage 12: Fixed-SA + RL Formulation** | Oct 13, 2026 | Placed learned proposals inside Fixed SA cooling ($\gamma=0, \lambda=0$ contextual bandit). | 20k short gate; 100k 3-seed extended gate; adaptive SA gate. | Positive delta won short gate (+0.63%) and extended gate (+0.105% mean, won 2/3 seeds). Adaptive SA hybrid lost (-0.36%). | Reject RL+Adaptive SA; establish RL+Fixed SA as primary method. | `runs/kaggle-rl-fixed-sa-gate-2026-10-13`, Commits `168add0`, `8e7c7e9`, `d4d9e38` |
| **Stage 13: Contextual Bandit Validation** | Oct 14, 2026 | Online MLP predicting signed $\Delta C$; full-run $\epsilon$ decay ($1.0 \rightarrow 0.5$); 50% bottleneck pools. | AlexNet-FC held-out (seeds 3-5); AlexNet-CONV (seeds 0-2); VGG16-FC held-out (seeds 3-5). | Won 3/3 held-out AlexNet-FC seeds (+1.97%). CONV tied at 37.02 µs floor. Won 3/3 held-out VGG16-FC seeds (+1.64%); 6/6 VGG16 overall (+1.95%, $p=0.031$). | Primary validated active method; shows positive matched-control evidence on tested workloads. | `runs/kaggle-bandit-vgg16-fc-*`, [`docs/PROJECT_STATE.md`](PROJECT_STATE.md), Commits `073f003`, `bacc334`, `1e9dee2` |
| **Stage 14: Repository Reorganization** | Oct 14, 2026 | Reorganized into `docs/`, `legacy/`, `tests/`, `src/analysis/`, `src/experiments/`; created codebase guide and audit. | Full test suite (`tests/run_all_tests.py`) and CLI help checks. | 21/21 non-torch tests pass cleanly; working tree clean; zero whitespace errors. | Finalize repository reorganization with unified codebase guide. | [`CODEBASE_GUIDE.md`](../CODEBASE_GUIDE.md), Commit `6fa6e84` |

---

## Questions Requiring Human Confirmation

1. **Attribution Disambiguation (Learned Scorer vs. Heuristic `proxy_gain`)**:
   - The contextual bandit demonstrates an average ~1.95% advantage over a focused-uniform control on VGG16-FC across all 6 seeds. However, candidate feature 9 is a cheap analytical communication gain (`proxy_gain`). Commit `1e9dee2` implemented the `--ppo_asa_bandit_control proxy_greedy` comparator. It remains to be run via `bandit-proxy-short` to confirm whether the neural network has learned subtle non-linear topological features, or whether a simple non-learned greedy selector (`argmax proxy_gain`) achieves the same performance without PyTorch overhead.
2. **True Accelerator Throughput vs. Latency Ratios**:
   - The paper’s Figure 10 reports system performance in terms of relative latency reduction. In streaming multi-chip accelerators, true throughput across continuous inference workloads requires modeling batch filling, steady-state execution, and draining phases across pipeline stages. The current inverse objective ratio $B / L(P)$ serves as a proxy metric, but cannot be published as measured hardware throughput without implementing a cycle-accurate pipeline queue simulator.
3. **The Unresolved VVA Throughput and Simulator Code**:
   - Despite extensive literature searches, the exact hardware execution throughput of the VVA units in the TODAES 2020 paper remains unpublished. Whether the original authors evaluated VVA at 1, 8, or 16 operations per cycle—or modeled compute and communication overlap in hardware queues—cannot be resolved without direct communication with the original authors.
4. **Comparison Between Restricted-Pool Control (`16` Candidates) and Unrestricted Plain Fixed SA**:
   - On held-out AlexNet-FC seeds 3–5 (`558cf4b`), Contextual Bandit + Fixed SA (`2.20656` µs) beat its paired 16-candidate frozen uniform control (`2.25077` µs) in 3/3 seeds (+1.965%), and beat unrestricted Plain Fixed SA (`2.22160` µs) in 2/3 seeds (+0.669% mean, CI crossing zero), while across all 6 seeds 0–5 it trailed Plain Fixed SA by `-0.161%`. A future ablation could investigate whether sampling 16 candidates without replacement slightly restricts exploration compared to single-proposal Plain Fixed SA.
