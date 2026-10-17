#!/usr/bin/env bash
# Certify whether a workload has unresolved Proxy-SA headroom before RL trials.
set -euo pipefail

out="${1:-runs/kaggle-headroom-certification}"
short_budget="${SHORT_BUDGET_OVERRIDE:-20000}"
reference_budget="${REFERENCE_BUDGET_OVERRIDE:-100000}"
seeds=(0 1 2)
[[ -n "${SEEDS_OVERRIDE:-}" ]] && read -r -a seeds <<<"${SEEDS_OVERRIDE}"
[[ "${short_budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid short budget"; exit 2; }
[[ "${reference_budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid reference budget"; exit 2; }
[[ "${reference_budget}" -gt "${short_budget}" ]] || { echo "Reference budget must exceed short budget"; exit 2; }
[[ ! -e "${out}" && ! -e "${out}.zip" ]] || { echo "Use a fresh output path"; exit 2; }
mkdir -p "${out}"
git rev-parse HEAD > "${out}/commit.txt"
git status --short > "${out}/git-status.txt"
python tests/run_all_tests.py > "${out}/tests.log" 2>&1

hardware=(
  --device cpu --cpu_threads 2 --use_cnn
  --partition_mode paper_targets --timing_model paper_pipeline
  --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 64
)
proxy=(
  --algo proxy_sa --proxy_sa_candidates 16
  --proxy_sa_focus_fraction 0.5 --proxy_sa_focus_stage_slack 0.05
  --proxy_sa_epsilon_final 0.25
)
# Workloads are deliberately fixed before results are inspected.
workloads=("alexnet conv" "vgg16 conv" "resnet50 conv")
for entry in "${workloads[@]}"; do
  read -r model region <<<"${entry}"
  for seed in "${seeds[@]}"; do
    for scale in short reference; do
      if [[ "${scale}" == short ]]; then budget="${short_budget}"; else budget="${reference_budget}"; fi
      echo "===== ${model}-${region} seed ${seed} ${scale} (${budget}) ====="
      PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
        "${hardware[@]}" "${proxy[@]}" \
        --model "${model}" --workload_region "${region}" --seed "${seed}" \
        --iters "${budget}" --report "${out}/${model}-${region}-seed${seed}-${scale}.json" \
        > "${out}/${model}-${region}-seed${seed}-${scale}.log" 2>&1
    done
  done
done

python - "${out}" "${short_budget}" "${reference_budget}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
short_budget, reference_budget = map(int, sys.argv[2:])
rows = {}
for path in root.glob("*-seed*-*.json"):
    m = re.fullmatch(r"(alexnet|vgg16|resnet50)-(conv)-seed(\d+)-(short|reference)", path.stem)
    if not m:
        continue
    model, region, seed, scale = m.groups()
    row = json.loads(path.read_text())
    expected = (short_budget if scale == "short" else reference_budget) + 1
    assert row["complete_placement_evaluations"] == expected, path
    rows.setdefault(f"{model}-{region}", {}).setdefault(int(seed), {})[scale] = row

summary = {"protocol": "Proxy-SA short-vs-reference headroom certification",
           "short_budget": short_budget, "reference_budget": reference_budget,
           "eligibility_rule": "median short cost is at least 2% above median reference cost",
           "workloads": {}}
for workload, seed_rows in sorted(rows.items()):
    assert all(set(pair) == {"short", "reference"} for pair in seed_rows.values())
    short = [pair["short"]["best_cost"] for pair in seed_rows.values()]
    ref = [pair["reference"]["best_cost"] for pair in seed_rows.values()]
    initial = [pair["short"]["algorithm_metadata"]["initial_cost"] for pair in seed_rows.values()]
    sensitivity = [pair["short"]["objective_sensitivity"] for pair in seed_rows.values()]
    median_short, median_ref = statistics.median(short), statistics.median(ref)
    unresolved = 100 * (median_short - median_ref) / median_short
    floor_gap = [100 * (cost - s["best_sample_breakdown"]["compute_only_floor"]) / cost
                 for cost, s in zip(ref, sensitivity)]
    summary["workloads"][workload] = {
        "seeds": sorted(seed_rows),
        "short_costs_seconds": short,
        "reference_costs_seconds": ref,
        "initial_costs_seconds": initial,
        "median_short_seconds": median_short,
        "median_reference_seconds": median_ref,
        "unresolved_headroom_percent": unresolved,
        "reference_gap_above_compute_floor_percent": statistics.median(floor_gap),
        "median_random_placement_span_percent": statistics.median(
            100 * s["relative_span"] for s in sensitivity),
        "rl_eligible": unresolved >= 2.0,
    }
summary["eligible_workloads"] = [name for name, row in summary["workloads"].items()
                                 if row["rl_eligible"]]
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

python - "${out}" "${out}.zip" <<'PY'
import sys, zipfile
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            bundle.write(path, path.relative_to(source.parent))
print(f"Saved {destination}")
PY
echo "FINISHED: ${out}.zip"
sha256sum "${out}.zip"
