#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-runs/kaggle-paper-baseline-calibration}"
budget="${2:-1000000}"
vva_rate="${3:-1}"
partition_vva_rate="${4:-${vva_rate}}"
archive="${output_dir%/}.zip"
mkdir -p "${output_dir}"

if (( budget < 1000000 )); then
  echo "The confirmatory calibration requires at least 1,000,000 proposals." >&2
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

destination, budget = sys.argv[1], int(sys.argv[2])
vva_rate, partition_vva_rate = map(float, sys.argv[3:5])
manifest = {
    "experiment": "AlexNet-FC paper-scale baseline calibration",
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "prespecified_before_data_generation": True,
    "source_protocol": {
        "paper_section": "TODEAS 2020 Section 4.2 and Algorithm 2",
        "random_search_samples": budget,
        "fixed_sa_proposals": budget,
        "adaptive_sa_proposals": budget,
        "cooldown": 0.99,
        "neighbor_fraction": 0.01,
        "temperature_note": (
            "The paper says T0 and Tend are application-specific but does not "
            "publish them. Fixed SA measures T0 in objective units, then uses "
            "a frozen geometric schedule; ASA is an explicitly labeled extension."
        ),
    },
    "configuration": {
        "model": "alexnet",
        "region": "fc",
        "partition_mode": "paper_targets",
        "timing_model": "paper_pipeline",
        "routing_model": "paper_xy",
        "seed": 0,
        "vva_additions_per_cycle": vva_rate,
        "partition_vva_additions_per_cycle": partition_vva_rate,
    },
    "decision": (
        "Use normalized-to-BS costs and RS/SA ordering to assess simulator "
        "calibration. Do not reinterpret this one-seed run as an RL result."
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
  --sensitivity_trials 64 --seed 0
  --vva_ops_per_cycle "${vva_rate}"
  --partition_vva_ops_per_cycle "${partition_vva_rate}"
)

echo "Computing the sequential baseline."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
  --algo bs "${common[@]}" \
  --report "${output_dir}/bs-report.json" \
  >"${output_dir}/bs.log" 2>&1

echo "Launching matched paper-scale RS, fixed SA, and adaptive SA in parallel."
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
  --algo random "${common[@]}" --iters "${budget}" \
  --report "${output_dir}/random-report.json" \
  2>&1 | tee "${output_dir}/random.log" &
random_pid=$!

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
  --algo sa "${common[@]}" --iters "${budget}" \
  --sa_initial_acceptance 0.8 --sa_final_temperature_ratio 1e-6 \
  --sa_calibration_trials 32 \
  --report "${output_dir}/sa-report.json" \
  2>&1 | tee "${output_dir}/sa.log" &
sa_pid=$!

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_multi_chip.py \
  --algo asa "${common[@]}" --iters "${budget}" \
  --asa_initial_acceptance 0.8 --asa_target_acceptance 0.30 \
  --asa_calibration_trials 32 --asa_adapt_window 100 \
  --asa_diagnostics "${output_dir}/asa.jsonl" \
  --report "${output_dir}/asa-report.json" \
  2>&1 | tee "${output_dir}/asa.log" &
asa_pid=$!

status=0
wait "${random_pid}" || status=1
echo "Random search finished."
wait "${sa_pid}" || status=1
echo "Fixed SA finished."
wait "${asa_pid}" || status=1
echo "Adaptive SA finished."
archive_results
if (( status != 0 )); then
  echo "At least one baseline failed; inspect the archived logs." >&2
  exit 1
fi

python - "${output_dir}" "${budget}" <<'PY'
from pathlib import Path
import json
import sys

out, budget = Path(sys.argv[1]), int(sys.argv[2])
reports = {}
for method in ("bs", "random", "sa", "asa"):
    with (out / f"{method}-report.json").open() as stream:
        reports[method] = json.load(stream)

bs = reports["bs"]["best_cost"]
rows = {}
for method, report in reports.items():
    metadata = report.get("algorithm_metadata") or {}
    rows[method] = {
        "best_cost_seconds": report["best_cost"],
        "best_cost_microseconds": report["best_cost"] * 1e6,
        "normalized_to_bs": report["best_cost"] / bs,
        "improvement_over_bs_percent": (bs - report["best_cost"]) / bs * 100,
        "reported_complete_placement_evaluations":
            report["complete_placement_evaluations"],
        "initial_placement_evaluations":
            metadata.get("initial_placement_evaluations", 0),
        "total_objective_evaluations": metadata.get(
            "total_objective_evaluations",
            report["complete_placement_evaluations"]),
        "seconds_elapsed": report["seconds_elapsed"],
        "algorithm_metadata": metadata,
    }

summary = {
    "protocol": "AlexNet-FC, seed 0, paper-scale baseline calibration",
    "proposal_budget_per_search_method": budget,
    "vva_additions_per_cycle": reports["bs"]["config"]["vva_ops_per_cycle"],
    "partition_vva_additions_per_cycle":
        reports["bs"]["config"]["partition_vva_ops_per_cycle"],
    "budget_note": (
        "RS samples exactly the declared number of complete placements. "
        "SA and ASA evaluate one initialization plus the declared proposals; "
        "both counts are reported explicitly."
    ),
    "results": rows,
    "fixed_sa_improvement_over_random_percent": (
        (reports["random"]["best_cost"] - reports["sa"]["best_cost"])
        / reports["random"]["best_cost"] * 100
    ),
    "adaptive_sa_improvement_over_random_percent": (
        (reports["random"]["best_cost"] - reports["asa"]["best_cost"])
        / reports["random"]["best_cost"] * 100
    ),
    "adaptive_sa_improvement_over_fixed_sa_percent": (
        (reports["sa"]["best_cost"] - reports["asa"]["best_cost"])
        / reports["sa"]["best_cost"] * 100
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
