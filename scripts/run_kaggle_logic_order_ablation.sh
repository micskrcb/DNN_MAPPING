#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-logic-order-ablation}"
budget="${2:-20000}"
vva_rate="${3:-16}"
partition_vva_rate="${4:-1}"
archive="${output_dir%/}.zip"
orders=(output_group_interleaved all_vmm_then_vva)

if (( budget < 1000 )); then
  echo "Use at least 1,000 proposals for this exploratory ablation." >&2
  exit 2
fi
python - "${vva_rate}" "${partition_vva_rate}" <<'PY'
import math
import sys
for label, raw in zip(("execution", "partition"), sys.argv[1:]):
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{label} VVA rate must be finite and positive")
PY
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

python - "${output_dir}/experiment-manifest.json" "${budget}" \
  "${vva_rate}" "${partition_vva_rate}" <<'PY'
import json
import subprocess
import sys

destination, budget, execution, partition = sys.argv[1:]
manifest = {
    "experiment": "AlexNet-FC logical-core indexing ablation",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "status": "exploratory reconstruction-assumption diagnosis",
    "configuration": {
        "model": "alexnet",
        "region": "fc",
        "partition_mode": "paper_targets",
        "timing_model": "paper_pipeline",
        "routing_model": "paper_xy",
        "seed": 0,
        "vva_additions_per_cycle": float(execution),
        "partition_vva_additions_per_cycle": float(partition),
        "proposals_per_search": int(budget),
        "orders": ["output_group_interleaved", "all_vmm_then_vva"],
    },
    "decision_rule": (
        "The paper specifies BS physical ordering but not logical VMM/VVA "
        "numbering. Compare matched BS, RS, and fixed SA under two explicit "
        "numberings. This does not select a true numbering or establish paper reproduction."
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
  --vva_ops_per_cycle "${vva_rate}"
  --partition_vva_ops_per_cycle "${partition_vva_rate}"
  --sensitivity_trials 256 --seed 0
)

for order in "${orders[@]}"; do
  order_dir="${output_dir}/${order}"
  mkdir -p "${order_dir}"
  echo "===== Logical order: ${order} ====="

  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
    --algo bs "${common[@]}" --logic_task_order "${order}" \
    --report "${order_dir}/bs-report.json" \
    >"${order_dir}/bs.log" 2>&1

  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python src/run_multi_chip.py \
      --algo random "${common[@]}" --logic_task_order "${order}" \
      --iters "${budget}" --report "${order_dir}/random-report.json" \
      2>&1 | tee "${order_dir}/random.log" &
  random_pid=$!

  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python src/run_multi_chip.py \
      --algo sa "${common[@]}" --logic_task_order "${order}" \
      --iters "${budget}" --sa_initial_acceptance 0.8 \
      --sa_final_temperature_ratio 1e-6 --sa_calibration_trials 32 \
      --report "${order_dir}/sa-report.json" \
      2>&1 | tee "${order_dir}/sa.log" &
  sa_pid=$!

  status=0
  wait "${random_pid}" || status=1
  wait "${sa_pid}" || status=1
  archive_results
  if (( status != 0 )); then
    echo "A search failed for logical order ${order}; inspect archived logs." >&2
    exit 1
  fi
done

python - "${output_dir}" <<'PY'
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
rows = []
for order in ("output_group_interleaved", "all_vmm_then_vva"):
    reports = {}
    for method in ("bs", "random", "sa"):
        with (out / order / f"{method}-report.json").open() as stream:
            reports[method] = json.load(stream)
    bs = reports["bs"]["best_cost"]
    rows.append({
        "logic_task_order": order,
        "bs_microseconds": bs * 1e6,
        "random_microseconds": reports["random"]["best_cost"] * 1e6,
        "fixed_sa_microseconds": reports["sa"]["best_cost"] * 1e6,
        "random_normalized_to_bs": reports["random"]["best_cost"] / bs,
        "fixed_sa_normalized_to_bs": reports["sa"]["best_cost"] / bs,
        "fixed_sa_improvement_over_random_percent": (
            (reports["random"]["best_cost"] - reports["sa"]["best_cost"])
            / reports["random"]["best_cost"] * 100),
        "routing_diagnostics": {
            method: reports[method]["routing_diagnostics"]
            for method in ("bs", "random", "sa")
        },
    })
summary = {
    "protocol": "AlexNet-FC matched logical-core indexing ablation",
    "results": rows,
    "interpretation_warning": (
        "Logical numbering is unpublished. This ablation quantifies baseline "
        "sensitivity and cannot identify the authors' original numbering."
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
