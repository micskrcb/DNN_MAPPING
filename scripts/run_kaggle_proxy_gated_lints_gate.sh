#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-proxy-gated-lints-gate}"
budget="${BUDGET_OVERRIDE:-20000}"
archive="${out}.zip"
seeds=(0 1 2)
[[ -n "${SEEDS_OVERRIDE:-}" ]] && read -r -a seeds <<<"${SEEDS_OVERRIDE}"
[[ "${budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid budget"; exit 2; }
[[ ! -e "${out}" && ! -e "${archive}" ]] || { echo "Use a fresh output path"; exit 2; }
mkdir -p "${out}"
git rev-parse HEAD > "${out}/commit.txt"
git status --short > "${out}/git-status.txt"
python tests/run_all_tests.py > "${out}/tests.log" 2>&1

common=(
  --device cpu --cpu_threads 2 --use_cnn --model vgg16
  --partition_mode paper_targets --workload_region conv
  --timing_model paper_pipeline --routing_model paper_xy
  --chips_x 4 --chips_y 4 --rows 16 --cols 16 --sensitivity_trials 16
  --operator_sa_discount 0.98 --operator_sa_minimum_exploration 0.03
  --operator_sa_phase_bins 3 --operator_sa_thompson_scale 0.25
  --operator_sa_reward_window 50
)

for seed in "${seeds[@]}"; do
  for method in lints_outcome lints_signed proxy_unthrottled; do
    extra=()
    case "${method}" in
      lints_outcome)
        extra=(--operator_sa_selection_policy lints
               --operator_sa_proxy_gate_slate 2
               --operator_sa_reward outcome)
        ;;
      lints_signed)
        extra=(--operator_sa_selection_policy lints
               --operator_sa_proxy_gate_slate 2
               --operator_sa_reward signed)
        ;;
      proxy_unthrottled)
        extra=(--operator_sa_control proxy
               --operator_sa_proxy_gate_slate 1)
        ;;
    esac
    echo "===== seed ${seed}, ${method} ====="
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      --algo operator_sa --iters "${budget}" --seed "${seed}" \
      --report "${out}/seed${seed}-${method}.json" \
      "${extra[@]}" "${common[@]}" \
      > "${out}/seed${seed}-${method}.log" 2>&1
  done
done

python - "${out}" "${budget}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root, budget = Path(sys.argv[1]), int(sys.argv[2])
names = {"lints_outcome", "lints_signed", "proxy_unthrottled"}
rows = {}
for path in root.glob("seed*.json"):
    match = re.fullmatch(
        r"seed(\d+)-(lints_outcome|lints_signed|proxy_unthrottled)",
        path.stem)
    if match:
        rows.setdefault(int(match.group(1)), {})[match.group(2)] = json.loads(
            path.read_text())

pairs, protocol_pass = [], True
for seed, methods in sorted(rows.items()):
    valid = set(methods) == names
    valid &= all(row["complete_placement_evaluations"] == budget + 1
                 for row in methods.values())
    for name in ("lints_outcome", "lints_signed"):
        metadata = methods[name]["algorithm_metadata"]
        valid &= metadata["supervised_operator_labels"] == 0
        valid &= metadata["proxy_gate_candidate_proposals"] == (
            2 * metadata["decision_steps"])
    costs = {name: methods[name]["best_cost"] for name in names}
    proxy = costs["proxy_unthrottled"]
    outcome_gain = 100 * (proxy - costs["lints_outcome"]) / proxy
    signed_gain = 100 * (proxy - costs["lints_signed"]) / proxy
    pairs.append({
        "seed": seed,
        "costs": costs,
        "outcome_improvement_over_proxy_percent": outcome_gain,
        "signed_improvement_over_proxy_percent": signed_gain,
        "outcome_operator_counts": methods["lints_outcome"][
            "algorithm_metadata"]["operator_selection_counts"],
        "outcome_gate_overrides": methods["lints_outcome"][
            "algorithm_metadata"]["proxy_gate_overrides"],
        "accounting_pass": valid,
    })
    protocol_pass &= valid

outcome = [row["outcome_improvement_over_proxy_percent"] for row in pairs]
signed = [row["signed_improvement_over_proxy_percent"] for row in pairs]
summary = {
    "budget": budget,
    "pairs": pairs,
    "protocol_pass": protocol_pass,
    "mean_outcome_improvement_over_proxy_percent": statistics.mean(outcome),
    "outcome_wins_over_proxy": sum(value > 0 for value in outcome),
    "mean_signed_improvement_over_proxy_percent": statistics.mean(signed),
    "signed_wins_over_proxy": sum(value > 0 for value in signed),
}
summary["advancement_pass"] = (
    len(pairs) >= 3 and summary["outcome_wins_over_proxy"] >= 2 and
    summary["mean_outcome_improvement_over_proxy_percent"] >= 1.0)
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

python - "${out}" "${archive}" <<'PY'
import sys, zipfile
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            bundle.write(path, path.relative_to(source.parent))
print(f"Saved {destination}")
PY

echo "FINISHED: ${archive}"
sha256sum "${archive}"
