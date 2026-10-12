#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-evaluator-vva-sensitivity}"
budget="${2:-20000}"
archive="${output_dir%/}.zip"
rates=(1 8 16 32 64 128)

if (( budget < 1000 )); then
  echo "Use at least 1,000 proposals for this exploratory sweep." >&2
  exit 2
fi
mkdir -p "${output_dir}"

archive_results() {
  [[ -d "${output_dir}" ]] || return 0
  python - "${output_dir}" "${archive}" <<'PY'
import os
import sys
import zipfile

source, destination = sys.argv[1:]
temporary = destination + ".tmp"
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
    for root, _, files in os.walk(source):
        for name in files:
            path = os.path.join(root, name)
            archive.write(path, os.path.relpath(path, os.path.dirname(source)))
os.replace(temporary, destination)
print(f"Saved recoverable results to {destination}")
PY
}
trap archive_results EXIT

echo "Running reconciliation tests."
PYTHONPATH=src python -m unittest src.test_reconciliation

python - "${output_dir}/experiment-manifest.json" "${budget}" "${rates[@]}" <<'PY'
import json
import subprocess
import sys

destination = sys.argv[1]
budget = int(sys.argv[2])
rates = [float(value) for value in sys.argv[3:]]
manifest = {
    "experiment": "AlexNet-FC VVA-throughput sensitivity",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "status": "exploratory simulator-assumption diagnosis",
    "motivating_calibration_commit": "80aec4fb694bf69de6b5553bcd0c8ba646a2bc7f",
    "configuration": {
        "model": "alexnet",
        "region": "fc",
        "partition_mode": "paper_targets",
        "timing_model": "paper_pipeline",
        "routing_model": "paper_xy",
        "seed": 0,
        "vva_additions_per_cycle": rates,
        "random_search_samples_per_rate": budget,
        "fixed_sa_proposals_per_rate": budget,
    },
    "paper_figure_visual_estimates": {
        "random_search_normalized_to_bs": 0.59,
        "fixed_sa_normalized_to_bs": 0.50,
        "warning": "Approximate readings from Figure 10, not tabulated values.",
    },
    "decision_rule": (
        "Use the sweep to identify whether the unpublished VVA rate controls "
        "the missing objective headroom. Do not claim calibration or select an "
        "RL result from this exploratory data. Any candidate setting must be "
        "frozen and checked at full budget and on a fresh workload."
    ),
}
with open(destination, "w") as stream:
    json.dump(manifest, stream, indent=2)
PY

common=(
  --device cpu
  --use_cnn --model alexnet
  --partition_mode paper_targets --workload_region fc
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 256 --seed 0
)

for rate in "${rates[@]}"; do
  rate_dir="${output_dir}/vva-${rate}"
  mkdir -p "${rate_dir}"
  echo "===== VVA ${rate} additions/cycle ====="

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
    --algo bs "${common[@]}" --vva_ops_per_cycle "${rate}" \
    --report "${rate_dir}/bs-report.json" \
    >"${rate_dir}/bs.log" 2>&1

  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python src/run_multi_chip.py \
      --algo random "${common[@]}" --vva_ops_per_cycle "${rate}" \
      --iters "${budget}" --report "${rate_dir}/random-report.json" \
      2>&1 | tee "${rate_dir}/random.log" &
  random_pid=$!

  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python src/run_multi_chip.py \
      --algo sa "${common[@]}" --vva_ops_per_cycle "${rate}" \
      --iters "${budget}" --sa_initial_acceptance 0.8 \
      --sa_final_temperature_ratio 1e-6 --sa_calibration_trials 32 \
      --report "${rate_dir}/sa-report.json" \
      2>&1 | tee "${rate_dir}/sa.log" &
  sa_pid=$!

  status=0
  wait "${random_pid}" || status=1
  wait "${sa_pid}" || status=1
  archive_results
  if (( status != 0 )); then
    echo "A search failed for VVA rate ${rate}; inspect archived logs." >&2
    exit 1
  fi
done

python - "${output_dir}" "${budget}" "${rates[@]}" <<'PY'
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
budget = int(sys.argv[2])
rates = sys.argv[3:]
rows = []
for rate in rates:
    reports = {}
    for method in ("bs", "random", "sa"):
        with (out / f"vva-{rate}" / f"{method}-report.json").open() as stream:
            reports[method] = json.load(stream)
    bs = reports["bs"]["best_cost"]
    sensitivity = reports["bs"]["objective_sensitivity"]
    floor = sensitivity["best_sample_breakdown"]["compute_only_floor"]
    random_cost = reports["random"]["best_cost"]
    sa_cost = reports["sa"]["best_cost"]
    rows.append({
        "vva_additions_per_cycle": float(rate),
        "bs_microseconds": bs * 1e6,
        "random_microseconds": random_cost * 1e6,
        "fixed_sa_microseconds": sa_cost * 1e6,
        "random_normalized_to_bs": random_cost / bs,
        "fixed_sa_normalized_to_bs": sa_cost / bs,
        "compute_floor_normalized_to_bs": floor / bs,
        "random_improvement_over_bs_percent": (bs - random_cost) / bs * 100,
        "fixed_sa_improvement_over_bs_percent": (bs - sa_cost) / bs * 100,
        "fixed_sa_improvement_over_random_percent":
            (random_cost - sa_cost) / random_cost * 100,
        "random_sample_relative_span": sensitivity["relative_span"],
        "random_seconds_elapsed": reports["random"]["seconds_elapsed"],
        "fixed_sa_seconds_elapsed": reports["sa"]["seconds_elapsed"],
    })

summary = {
    "protocol": "exploratory AlexNet-FC VVA-throughput sensitivity",
    "proposal_budget_per_search": budget,
    "paper_figure_visual_estimates": {
        "random_normalized_to_bs": 0.59,
        "fixed_sa_normalized_to_bs": 0.50,
    },
    "results": rows,
    "interpretation_warning": (
        "This sweep diagnoses an unpublished simulator parameter. It cannot "
        "by itself validate a selected rate or establish paper reproduction."
    ),
}
with (out / "summary.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2))
PY

archive_results
trap - EXIT
echo "FINISHED"
echo "Download ${archive} from the Kaggle Output/Files panel."
sha256sum "${archive}"
