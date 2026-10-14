# Historical Documentation & Research Archive Index

This directory archives historical documentation, obsolete decision memos, comparative reference audits, and superseded README iterations from the development of the `DNN_MAPPING` multi-chip core placement project.

All material in this archive is preserved to maintain an unbroken audit trail of hypotheses, diagnostic experiments, code refactorings, and empirical outcomes.

---

## Index of Archived Artifacts

### 1. Research Decision Memos & Diagnostic Audits

| Document | Date / Revision | Original Context & Topic | Why It Was Archived |
| :--- | :--- | :--- | :--- |
| [`CLAUDE_AUDIT_RESPONSE.md`](CLAUDE_AUDIT_RESPONSE.md) | 2026-10-11 (`1fd75c4`) | Diagnostic engineering evaluation of an external audit regarding null candidate-set PPO results. Designed the 3-stage diagnostic experiments E0 (pool oracle), E1 (planted proxy reward), and E2 (real-reward pilot). | Diagnostic experiments were completed and superseded by the contextual-bandit and fixed-SA baseline calibration results recorded in `docs/PROJECT_STATE.md`. |
| [`DDPG_REFERENCE_AUDIT.md`](DDPG_REFERENCE_AUDIT.md) | 2026-10-08 | Code-level comparative audit comparing repository DDPG and Guided DDPG implementations against reference baselines (OpenAI Spinning Up, Stable-Baselines3, TD3, Wolpertinger, DDPGfD). | Diagnosed the core failure modes of continuous DDPG in discrete placement spaces. Guided DDPG was subsequently formally closed after 5-seed empirical ablation. |
| [`GITHUB_IMPLEMENTATION_AUDIT.md`](GITHUB_IMPLEMENTATION_AUDIT.md) | 2026-10-11 | Systematic comparative survey of public placement repositories on GitHub (Qualcomm `RL-Based-SA-Public`, `WOOSHIK-M`, Google `circuit_training`, `MacroPlacement`). | Analyzed why initial PPO-ASA gates had weak signal, identifying a $\sim 2,500\times$ training transition discrepancy. Informed the multi-chain training structure and decoupled actor-critic architecture. |
| [`PAPER_SCALE_DECISION.md`](PAPER_SCALE_DECISION.md) | 2026-10-10 | Decision memo establishing the sufficient-scale training protocol (1,250,000 proposals, 1,250 PPO update cycles) for PPO-ASA proposal scoring. | The 1.25M run was executed (`bdcd801`, 6.47 GPU hours) and definitively failed holdout gates ($p > 0.05$, policy entropy stayed at uniform). The architecture was formally closed. |
| [`Two_Month_Publication_Roadmap.docx`](Two_Month_Publication_Roadmap.docx) | Sept 2026 | Milestone and timeline planning document for research manuscript preparation. | Planning document; preserved for historical project management context. |

---

### 2. Historical README Iterations (`prev_version_readmes/`)

The [`prev_version_readmes/`](prev_version_readmes/) subdirectory preserves the chronological documentation lineage that preceded the modern multi-chip codebase:

| File | Historical Milestone | Core Content & Key Differences from Present State |
| :--- | :--- | :--- |
| [`README-for_orignal+initialchanges.md`](prev_version_readmes/README-for_orignal+initialchanges.md) | **Milestone 1** (Initial Fork) | Documents the upstream single-chip PPO repository on netlists (`WOOSHIK-M`). Contains line-by-line patch records for Python 3.10 and CPU compatibility. Single-chip only; uses L1 Manhattan cost rather than bottleneck pipeline latency. |
| [`old_section_41-50_of_readme_by_gpt.md`](prev_version_readmes/old_section_41-50_of_readme_by_gpt.md) | **Milestone 2** (Early Multi-Chip Draft) | Working draft of Sections 41–50 standing up the multi-chip DDPG environment on `SimpleCNN` (906 logic cores). Discovered the initial scaling bottleneck ($\sim 10$ min for 40 episodes) before sparse evaluation. |
| [`README_before_GPTplus.md`](prev_version_readmes/README_before_GPTplus.md) | **Milestone 3** (Exhaustive Manual) | Massive 50-section, 1,775-line manual capturing the repository immediately after sparse pipeline evaluation ($19.4\times$ speedup) and update decimation were implemented. Documents SimpleCNN 10,000-episode completion ($\sim 23.3\%$ improvement over random search). |
| [`claude_compressed_readme_6thsep.md`](prev_version_readmes/claude_compressed_readme_6thsep.md) | **Milestone 4** (Sept 6 Snapshot) | Compressed 13-section technical summary. Reports the completed AlexNet run ($46.8\%$ gain over RS). First document to identify the policy divergence failure mode ("current cost does not tighten around best cost") and CPU thermal throttling under sustained load. |
| [`Comprehensive Report_ RL-Based Core Placement Optimization.pdf`](prev_version_readmes/Comprehensive%20Report_%20RL-Based%20Core%20Placement%20Optimization.pdf) | Sept 2026 Report | Technical PDF report detailing early DDPG reproduction experiments and multi-chip hardware parameters. |
| [`Overview.png`](prev_version_readmes/Overview.png) | System Diagram | High-level diagram visualizing multi-chip accelerator layout and inter-chip network routing. |
| [`sem-2 25-26_ final screenshots.pdf`](prev_version_readmes/sem-2%2025-26_%20final%20screenshots.pdf) | Academic Presentation | Presentation slides and terminal run screenshots from the semester-2 progress defense. |

---

## Cross-Reference to Active Research Documentation

To read the current, active documentation of the repository:
- **Master Repository Guide**: [`../../README.md`](../../README.md)
- **New Contributor Codebase Guide**: [`../CODEBASE_GUIDE.md`](../CODEBASE_GUIDE.md)
- **Important Contributor Rules & Branch Policies**: [`../IMP_README.md`](../IMP_README.md)
- **Current Technical State & Empirical Benchmark Results**: [`../PROJECT_STATE.md`](../PROJECT_STATE.md)
- **Reproduction Gates & Experimental Backlog**: [`../NEXT_STEPS.md`](../NEXT_STEPS.md)
- **Research Findings, Failure Analyses & Literature Review**: [`../RESEARCH_FINDINGS.md`](../RESEARCH_FINDINGS.md)
- **Model Reconciliation Baseline**: [`../RECONCILIATION.md`](../RECONCILIATION.md)
- **PPO Split Adapter Architecture**: [`../PPO_REFERENCE_ADAPTER.md`](../PPO_REFERENCE_ADAPTER.md)
- **Original Research Paper**: [`../references/2020_TODEAS_Core_Placement_Optimization_Paper.pdf`](../references/2020_TODEAS_Core_Placement_Optimization_Paper.pdf)
