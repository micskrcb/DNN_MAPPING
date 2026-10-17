#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-macro-operator-gate}"
budget="${BUDGET_OVERRIDE:-20000}"
archive="${out}.zip"
seeds=(9 10 11)
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
  --operator_sa_discount 0.98 --operator_sa_minimum_exploration 0.05
  --operator_sa_selection_policy lints --operator_sa_thompson_scale 0.25
  --operator_sa_proxy_gate_slate 2 --operator_sa_macro_horizon 64
)

for seed in "${seeds[@]}"; do
  for method in macro_learned macro_uniform proxy_unthrottled; do
    extra=()
    case "${method}" in
      macro_learned) extra=() ;;
      macro_uniform) extra=(--operator_sa_control uniform) ;;
      proxy_unthrottled)
        extra=(--operator_sa_control proxy
               --operator_sa_proxy_gate_slate 1
               --operator_sa_macro_horizon 0)
        ;;
    esac
    echo "===== seed ${seed}, ${method} ====="
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      --algo operator_sa --iters "${budget}" --seed "${seed}" \
      --report "${out}/seed${seed}-${method}.json" \
      "${common[@]}" "${extra[@]}" \
      > "${out}/seed${seed}-${method}.log" 2>&1
  done
done

python - "${out}" "${budget}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root, budget = Path(sys.argv[1]), int(sys.argv[2])
names = {"macro_learned", "macro_uniform", "proxy_unthrottled"}
rows = {}
for path in root.glob("seed*.json"):
    match = re.fullmatch(
        r"seed(\d+)-(macro_learned|macro_uniform|proxy_unthrottled)",
        path.stem)
    if match:
        rows.setdefault(int(match.group(1)), {})[match.group(2)] = json.loads(
            path.read_text())

pairs, protocol_pass = [], True
for seed, methods in sorted(rows.items()):
    valid = set(methods) == names
    valid &= all(row["complete_placement_evaluations"] == budget + 1
                 for row in methods.values())
    learned = methods["macro_learned"]["algorithm_metadata"]
    uniform = methods["macro_uniform"]["algorithm_metadata"]
    proxy_meta = methods["proxy_unthrottled"]["algorithm_metadata"]
    valid &= learned["macro_horizon"] == uniform["macro_horizon"] == 64
    valid &= proxy_meta["macro_horizon"] == 0
    valid &= learned["macro_updates"] > 0 and uniform["macro_updates"] == 0
    valid &= sum(learned["macro_action_counts"].values()) == learned["macro_updates"]
    valid &= sum(uniform["macro_action_counts"].values()) > 0
    costs = {name: methods[name]["best_cost"] for name in names}
    proxy = costs["proxy_unthrottled"]
    frozen = costs["macro_uniform"]
    proxy_gain = 100 * (proxy - costs["macro_learned"]) / proxy
    frozen_gain = 100 * (frozen - costs["macro_learned"]) / frozen
    pairs.append({
        "seed": seed,
        "costs": costs,
        "learned_improvement_over_proxy_percent": proxy_gain,
        "learned_improvement_over_uniform_macro_percent": frozen_gain,
        "learned_macro_action_counts": learned["macro_action_counts"],
        "learned_macro_reward_sums": learned["macro_reward_sums"],
        "accounting_pass": valid,
    })
    protocol_pass &= valid

proxy_gains = [row["learned_improvement_over_proxy_percent"] for row in pairs]
frozen_gains = [row["learned_improvement_over_uniform_macro_percent"]
                for row in pairs]
summary = {
    "budget": budget,
    "seeds": sorted(rows),
    "pairs": pairs,
    "protocol_pass": protocol_pass,
    "mean_improvement_over_proxy_percent": statistics.mean(proxy_gains),
    "wins_over_proxy": sum(value > 0 for value in proxy_gains),
    "mean_improvement_over_uniform_macro_percent": statistics.mean(frozen_gains),
    "wins_over_uniform_macro": sum(value > 0 for value in frozen_gains),
}
summary["advancement_pass"] = (
    len(pairs) >= 3 and protocol_pass and
    summary["wins_over_proxy"] >= 2 and
    summary["wins_over_uniform_macro"] >= 2 and
    summary["mean_improvement_over_proxy_percent"] >= 1.0)
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
