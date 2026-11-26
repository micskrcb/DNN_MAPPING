# Research Paper: Core Placement Optimization for Multi-Chip DNN Accelerators

This directory contains the submission-ready LaTeX source for the paper:

> **"Core Placement Optimization for Multi-Chip DNN Accelerators: Reproduction, Empirical Failure Modes of Deep RL, and Contextual-Bandit-Guided Simulated Annealing"**

Format: **IEEEtran** (Double-column, 10pt conference format).

---

## Directory Contents

- **`main.tex`**: The primary LaTeX document. Fully self-contained, including all mathematical formulations, algorithm pseudocode, empirical result tables, and high-quality vector diagrams rendered directly via TikZ.
- **`references.bib`**: BibTeX bibliography with full citations, DOIs, and venues.
- **`README.md`**: This guide.

---

## How to Compile

### Option 1: Using `pdflatex` & `bibtex` (Command Line)

Run standard compilation sequence:
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or using `latexmk`:
```bash
latexmk -pdf main.tex
```

### Option 2: Overleaf

1. Zip the `paper/` directory:
   - Include `main.tex` and `references.bib`.
2. Open [Overleaf](https://www.overleaf.com), create a **New Project**, and select **Upload Project**.
3. Choose the zip file and click **Compile**.
4. The document compiles immediately using TeX Live (pdflatex).

### Option 3: VS Code (LaTeX Workshop)

1. Open the `paper/` folder in VS Code with the **LaTeX Workshop** extension installed.
2. Open `main.tex`.
3. Press `Ctrl + Alt + B` (or run "Build LaTeX project").
4. View the compiled PDF with `Ctrl + Alt + V`.

---

## Key Experimental Results Documented in the Paper

1. **Failure Modes of Continuous DDPG**:
   - High collision rates ($>96\%$) and action saturation ($<10$ distinct physical core locations).
   - Mathematical proof of reward horizon annihilation: for VGG16-FC ($H = 642$ decisions), terminal reward is discounted by $\gamma^{641} = 0.98^{641} \approx 2.38 \times 10^{-6}$ at the first decision.
   - Matched-budget benchmark (1,334 evaluations): DDPG fails to beat initial random sample ($5.834\,\mu$s), while local search reaches $5.271\,\mu$s (a 9.65% advantage over DDPG).

2. **Contextual-Bandit-Guided Simulated Annealing (CB-SA)**:
   - **AlexNet-FC**: $+1.965\%$ mean latency reduction across 3/3 held-out seeds (95% CI: $[1.008\%, 2.922\%]$).
   - **VGG16-FC**: $+1.946\%$ mean latency reduction across all 6 seeds (6/6 wins, exact sign-test $p = 0.031$).
   - **Long-Horizon Attribution**: Sustained $\approx 2.05\%$ gain over fixed SA at 200,000 evaluations.
   - **AlexNet-CONV**: Converges to the exact same physical floor ($37.021\,\mu$s) across all methods, proving early convergence acceleration without altering compute-bound hardware floors.

3. **Reproduction Gap & Sensitivity Analysis**:
   - The unstated 1~VVA~op/cycle rate creates an insurmountable $89.89\%$ compute floor ($22.40\,\mu$s vs.\ $24.92\,\mu$s BS), proving that the reference paper's claimed $50\%$ reduction is mathematically unreachable under this rate.
   - A sensitivity sweep reveals that elevating VVA throughput to 16~ops/cycle recovers an SA/BS ratio of $0.523$, resolving the divergence with published results.

