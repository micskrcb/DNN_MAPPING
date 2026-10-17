#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-full-information-operator-gate}"
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
  --operator_sa_full_information_stride 20
  --operator_sa_minimum_exploration 0
)

for seed in "${seeds[@]}"; do
  for method in learned proxy_control; do
    extra=()
    [[ "${method}" == proxy_control ]] && extra=(--operator_sa_control proxy)
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
rows = {}
for path in root.glob("seed*.json"):
    match = re.fullmatch(r"seed(\d+)-(learned|proxy_control)", path.stem)
    if not match:
        continue
    seed, method = int(match.group(1)), match.group(2)
    rows.setdefault(seed, {})[method] = json.loads(path.read_text())
pairs, protocol_pass = [], True
for seed, methods in sorted(rows.items()):
    valid = set(methods) == {"learned", "proxy_control"}
    valid &= all(d["complete_placement_evaluations"] == budget + 1
                 for d in methods.values())
    learned = methods["learned"]["best_cost"]
    proxy = methods["proxy_control"]["best_cost"]
    improvement = 100 * (proxy - learned) / proxy
    lm, pm = methods["learned"]["algorithm_metadata"], methods["proxy_control"]["algorithm_metadata"]
    valid &= lm["supervised_operator_labels"] == pm["supervised_operator_labels"]
    pairs.append({"seed": seed, "learned_cost": learned,
                  "matched_proxy_cost": proxy,
                  "learned_improvement_percent": improvement,
                  "supervised_pools": lm["supervised_pools"],
                  "supervised_operator_labels": lm["supervised_operator_labels"],
                  "learned_operator_counts": lm["operator_selection_counts"],
                  "accounting_pass": valid})
    protocol_pass &= valid
values = [p["learned_improvement_percent"] for p in pairs]
summary = {"budget": budget, "pairs": pairs, "protocol_pass": protocol_pass,
           "mean_learned_improvement_percent": statistics.mean(values),
           "learned_wins": sum(v > 0 for v in values)}
summary["advancement_pass"] = (
    len(pairs) >= 3 and summary["learned_wins"] >= 2 and
    summary["mean_learned_improvement_percent"] >= 1.0)
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
