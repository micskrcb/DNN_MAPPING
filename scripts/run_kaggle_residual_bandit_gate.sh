#!/usr/bin/env bash
set -euo pipefail

# From the repository root: bash scripts/run_kaggle_residual_bandit_gate.sh
# Extended runs are opt-in: bash scripts/run_kaggle_residual_bandit_gate.sh extended NEW_DIR
# Defaults: seeds 3 4 5, VGG16 CONV. SEEDS_OVERRIDE="3 4 5" is supported.
# DEVICE_OVERRIDE=cpu BUDGET_OVERRIDE=64 enables a small smoke check, not a gate verdict.
# Use a fresh output directory; logs, provenance and results are archived on every exit.
mode="${1:-short}"
output_dir="${2:-runs/kaggle-residual-bandit-${mode}}"
output_dir="${output_dir%/}"
archive="${output_dir}.zip"
model="${MODEL_OVERRIDE:-vgg16}"
region="${WORKLOAD_REGION_OVERRIDE:-conv}"
device="${DEVICE_OVERRIDE:-cuda}"
seeds=(3 4 5)

case "${mode}" in
  short) budget=20000 ;;
  extended) budget=100000 ;;
  *) echo "Usage: $0 [short|extended] [fresh-output-directory]" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
  echo "Usage: $0 [short|extended] [fresh-output-directory]" >&2
  exit 2
fi
smoke_test=false
if [[ -n "${BUDGET_OVERRIDE:-}" ]]; then
  budget="${BUDGET_OVERRIDE}"
  smoke_test=true
fi
if [[ ! "${budget}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Budget must be a positive integer." >&2
  exit 2
fi
complete_budget=$((budget + 1))
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a seeds <<<"${SEEDS_OVERRIDE}"
fi
if [[ "${device}" != cuda && "${device}" != cpu ]]; then
  echo "DEVICE_OVERRIDE must be cuda or cpu." >&2
  exit 2
fi
if [[ -e "${output_dir}" || -e "${archive}" ]]; then
  echo "Choose a fresh output directory and archive path: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}"
active_pids=()

archive_results() {
  python - "${output_dir}" "${archive}" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
temporary = str(destination) + ".tmp"
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            bundle.write(path, path.relative_to(source.parent))
os.replace(temporary, destination)
print(f"Saved recoverable results to {destination}")
PY
}

finalize() {
  status=$?
  trap - EXIT INT TERM
  set +e
  for pid in "${active_pids[@]}"; do kill "${pid}" 2>/dev/null; done
  for pid in "${active_pids[@]}"; do wait "${pid}" 2>/dev/null; done
  python - "${output_dir}/experiment-manifest.json" "${status}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    manifest = json.loads(path.read_text())
    ended = datetime.now(timezone.utc)
    manifest.update({
        "finished_at_utc": ended.isoformat(),
        "elapsed_wall_seconds": (
            ended - datetime.fromisoformat(manifest["started_at_utc"])
        ).total_seconds(),
        "exit_status": int(sys.argv[2]),
    })
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
PY
  metadata_status=$?
  archive_results
  archive_status=$?
  if (( status == 0 && (metadata_status != 0 || archive_status != 0) )); then
    status=1
  fi
  if (( status == 0 )); then
    echo "FINISHED: ${archive}"
    sha256sum "${archive}"
  fi
  exit "${status}"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Preserve every original contention-gate hyperparameter, adding only the
# residual scoring options below to both learned and proxy bandit runs.
common=(
  --use_cnn --model "${model}"
  --partition_mode paper_targets --workload_region "${region}"
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --vva_ops_per_cycle 16 --partition_vva_ops_per_cycle 1
  --sensitivity_trials 256
)
bandit_common=(
  --algo ppo_asa --ppo_asa_annealing fixed
  --ppo_asa_learner bandit
  --ppo_asa_candidates 16
  --ppo_asa_focus_fraction 0.5
  --ppo_asa_focus_stage_slack 0.05
  --ppo_asa_bandit_feature_set contention12
  --ppo_asa_bandit_exploration epsilon_top_m
  --ppo_asa_bandit_epsilon_final 0.25
  --ppo_asa_bandit_epsilon_decay 1.0
  --ppo_asa_bandit_top_m 4
  --ppo_asa_bandit_global_epsilon 0.05
  --ppo_asa_update_style reference_separate
  --ppo_asa_reward_mode proposed_delta
  --ppo_asa_rollout_steps 256
  --ppo_update_epochs 4
  --ppo_minibatch_size 256
  --ppo_hidden_dim 128
  --ppo_learning_rate 3e-4
  --ppo_asa_weight_decay 1e-4
  --ppo_gamma 0
  --ppo_gae_lambda 0
  --ppo_entropy_coef 0.001
  --ppo_asa_bandit_scoring proxy_residual
  --ppo_asa_bandit_residual_limit 0.25
  --ppo_asa_state_encoding relative_cost
)

python - "${output_dir}" "${mode}" "${budget}" "${seeds[*]}" \
  "${model}" "${region}" "${device}" "${smoke_test}" \
  "${common[@]}" "${bandit_common[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

directory, mode, budget, seeds, model, region, device, smoke = sys.argv[1:9]
root = Path(directory)
seeds = [int(value) for value in seeds.split()]
if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
    raise ValueError("Seeds must be distinct nonnegative integers")
status = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True)
diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
# A new runner may be untracked; include its actual contents in the patch too.
runner = "scripts/run_kaggle_residual_bandit_gate.sh"
tracked = subprocess.run(
    ["git", "ls-files", "--error-unmatch", runner], capture_output=True)
if tracked.returncode:
    addition = subprocess.run(
        ["git", "diff", "--no-index", "--binary", "/dev/null", runner],
        capture_output=True)
    if addition.returncode not in (0, 1):
        raise RuntimeError(addition.stderr.decode())
    diff += addition.stdout
(root / "repository.diff").write_bytes(diff)
(root / "repository-status.txt").write_text(status)
manifest = {
    "experiment": "proxy-residual contextual-bandit fixed-SA directional gate",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "repository_dirty": bool(status.strip()),
    "repository_status": status,
    "repository_diff_file": "repository.diff",
    "repository_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "mode": mode,
    "smoke_test": smoke == "true",
    "complete_objective_budget_per_method_per_seed": int(budget) + 1,
    "iters_by_method": {
        "learned": int(budget), "proxy": int(budget),
        "fixed-sa": int(budget), "random": int(budget) + 1,
    },
    "budget_accounting": (
        "Bandit and fixed SA count one initialization plus --iters proposals; "
        "random counts --iters complete placements. Sensitivity probes are "
        "separate, identical setup diagnostics for every method."
    ),
    "seeds": seeds,
    "model": model,
    "region": region,
    "bandit_device": device,
    "baseline_device": "cpu",
    "common_and_bandit_arguments": sys.argv[9:],
    "proxy_extra_arguments": [
        "--ppo_asa_disable_learning", "--ppo_asa_bandit_control", "proxy_greedy"
    ],
    "gate_prespecified": True,
    "gate_rule": (
        "Positive mean paired percentage improvement AND wins on at least "
        "two thirds of seeds against EACH of proxy and fixed SA. "
        "Random is contextual only. No post-hoc threshold changes."
    ),
    "extended_requires_explicit_invocation": True,
}
(root / "experiment-manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=False) + "\n")
PY

# Tests must pass before GPU discovery or any experiment starts.
PYTHONPATH=src python -m unittest src.test_reconciliation \
  2>&1 | tee "${output_dir}/unit-tests.log"
gpu_count="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except ImportError:
    print(0)
PY
)"
if [[ "${device}" == cuda ]] && (( gpu_count < 1 )); then
  echo "A CUDA GPU is required; DEVICE_OVERRIDE=cpu is available for smoke checks." >&2
  exit 1
fi

run_bandit() {
  local method="$1" gpu="$2"
  local extra=()
  if [[ "${method}" == proxy ]]; then
    extra=(--ppo_asa_disable_learning --ppo_asa_bandit_control proxy_greedy)
  else
    extra=(--save_checkpoint "${seed_dir}/learned.pt" --checkpoint_every "${budget}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" "${bandit_common[@]}" "${extra[@]}" --device "${device}" \
    --seed "${seed}" --iters "${budget}" \
    --asa_diagnostics "${seed_dir}/${method}.jsonl" \
    --report "${seed_dir}/${method}-report.json" \
    >"${seed_dir}/${method}.log" 2>&1
}

for seed in "${seeds[@]}"; do
  seed_dir="${output_dir}/seed-${seed}"
  mkdir -p "${seed_dir}"
  echo "Seed ${seed}: learned residual and proxy control"
  status=0
  if [[ "${device}" == cuda ]] && (( gpu_count >= 2 )); then
    run_bandit learned 0 &
    active_pids+=("$!")
    run_bandit proxy 1 &
    active_pids+=("$!")
    for pid in "${active_pids[@]}"; do wait "${pid}" || status=1; done
    active_pids=()
  else
    run_bandit learned 0 || status=1
    if (( status == 0 )); then run_bandit proxy 0 || status=1; fi
  fi
  archive_results
  if (( status != 0 )); then
    echo "Learned/proxy pair failed; inspect archived logs." >&2
    exit 1
  fi

  echo "Seed ${seed}: fixed SA and random CPU baselines"
  PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" --algo sa --device cpu \
    --seed "${seed}" --iters "${budget}" \
    --report "${seed_dir}/fixed-sa-report.json" \
    >"${seed_dir}/fixed-sa.log" 2>&1
  PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${common[@]}" --algo random --device cpu \
    --seed "${seed}" --iters "${complete_budget}" \
    --report "${seed_dir}/random-report.json" \
    >"${seed_dir}/random.log" 2>&1
  archive_results
done

python - "${output_dir}" <<'PY' 2>&1 | tee "${output_dir}/summary-generation.log"
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "experiment-manifest.json").read_text())
methods = ("learned", "proxy", "fixed-sa", "random")
expected_budget = manifest["complete_objective_budget_per_method_per_seed"]


def finite_numbers(value, location):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric: {location}={value}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{location}[{index}]")


rows = []
for seed in manifest["seeds"]:
    row = {"seed": seed}
    initial_costs = {}
    for method in methods:
        path = root / f"seed-{seed}" / f"{method}-report.json"
        report = json.loads(path.read_text())
        finite_numbers(report, str(path))
        assert report["config"]["seed"] == seed, f"Seed mismatch: {path}"
        assert report["config"]["iters"] == manifest["iters_by_method"][method], path
        assert report["complete_placement_evaluations"] == expected_budget, (
            f"Complete objective budget mismatch: {path}")
        metadata = report.get("algorithm_metadata") or {}
        if method != "random":
            assert metadata["total_objective_evaluations"] == expected_budget, (
                f"Algorithm objective budget mismatch: {path}")
        if "initial_cost" in metadata:
            initial_costs[method] = metadata["initial_cost"]
        assert report["best_cost"] > 0, f"Nonpositive objective: {path}"
        assert report["seconds_elapsed"] >= 0, f"Invalid elapsed time: {path}"
        row[f"{method}_microseconds"] = report["best_cost"] * 1e6
        row[f"{method}_seconds_elapsed"] = report["seconds_elapsed"]
        row[f"{method}_complete_objective_evaluations"] = (
            report["complete_placement_evaluations"])
        if method in ("learned", "proxy"):
            for line in (path.parent / f"{method}.jsonl").read_text().splitlines():
                if line.strip():
                    finite_numbers(json.loads(line), f"seed-{seed}/{method}.jsonl")
    assert "learned" in initial_costs and "proxy" in initial_costs
    assert all(value == initial_costs["learned"] for value in initial_costs.values()), (
        f"Initial costs do not match for seed {seed}: {initial_costs}")
    row["initial_costs"] = initial_costs
    for comparator in methods[1:]:
        reference = row[f"{comparator}_microseconds"]
        row[f"learned_improvement_over_{comparator}_percent"] = (
            (reference - row["learned_microseconds"]) / reference * 100.0)
    rows.append(row)

aggregates = {}
for method in methods:
    values = [row[f"{method}_microseconds"] for row in rows]
    aggregates[method] = {
        "mean_microseconds": statistics.mean(values),
        "sample_std_microseconds": statistics.stdev(values) if len(values) > 1 else None,
        "individual_microseconds": values,
    }
comparisons = {}
for comparator in methods[1:]:
    values = [row[f"learned_improvement_over_{comparator}_percent"] for row in rows]
    comparisons[f"learned_vs_{comparator}"] = {
        "mean_improvement_percent": statistics.mean(values),
        "sample_std_improvement_percent": statistics.stdev(values) if len(values) > 1 else None,
        "min_improvement_percent": min(values),
        "max_improvement_percent": max(values),
        "individual_improvement_percent": values,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
    }
minimum_wins = math.ceil(2 * len(rows) / 3)
criteria_met = all(
    comparisons[f"learned_vs_{comparator}"]["mean_improvement_percent"] > 0
    and comparisons[f"learned_vs_{comparator}"]["wins"] >= minimum_wins
    for comparator in ("proxy", "fixed-sa")
)
summary = {
    "rows": rows,
    "aggregates": aggregates,
    "comparisons": comparisons,
    "complete_objective_budget_per_method_per_seed": expected_budget,
    "budget_initial_match_and_finite_metric_checks_pass": True,
    "minimum_wins_against_each_required_comparator": minimum_wins,
    "directional_gate_pass": None if manifest["smoke_test"] else criteria_met,
    "smoke_test": manifest["smoke_test"],
    "gate_rule": manifest["gate_rule"],
    "uncertainty": (
        f"Only {len(rows)} seeds: paired individual effects, range and sample "
        "standard deviation are descriptive uncertainty, not evidence of "
        "statistical significance. Smoke checks do not produce a gate verdict."
    ),
    "elapsed_wall_seconds_at_summary": (
        datetime.now(timezone.utc) - datetime.fromisoformat(manifest["started_at_utc"])
    ).total_seconds(),
}
finite_numbers(summary, "summary")
(root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps(summary, indent=2, allow_nan=False))
PY
