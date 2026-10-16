#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-operator-sa-gate}"
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

run_profile() {
  local profile="$1"; shift
  for seed in "${seeds[@]}"; do
    for method in operator operator_control proxy sa random; do
      local algo="${method}" iterations="${budget}" extra=()
      case "${method}" in
        operator) algo=operator_sa ;;
        operator_control) algo=operator_sa; extra=(--operator_sa_disable_learning) ;;
        proxy) algo=proxy_sa ;;
        sa) algo=sa ;;
        random) algo=random; iterations=$((budget + 1)) ;;
      esac
      echo "===== ${profile}, seed ${seed}, ${method} ====="
      PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
        --algo "${algo}" --iters "${iterations}" --seed "${seed}" \
        --device cpu --cpu_threads 2 --sensitivity_trials 16 \
        --report "${out}/${profile}-seed${seed}-${method}.json" \
        "${extra[@]}" "$@" \
        > "${out}/${profile}-seed${seed}-${method}.log" 2>&1
    done
  done
}

run_profile alexnet-fc \
  --use_cnn --model alexnet --partition_mode paper_targets \
  --workload_region fc --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16

run_profile vgg16-conv \
  --use_cnn --model vgg16 --partition_mode paper_targets \
  --workload_region conv --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16

python - "${out}" "${budget}" <<'PY'
import json, re, statistics, sys
from pathlib import Path
root, budget = Path(sys.argv[1]), int(sys.argv[2])
rows = {}
pattern = re.compile(r"(.+)-seed(\d+)-(operator_control|operator|proxy|sa|random)$")
for path in root.glob("*.json"):
    match = pattern.fullmatch(path.stem)
    if not match:
        continue
    profile, seed, method = match.group(1), int(match.group(2)), match.group(3)
    data = json.loads(path.read_text())
    rows.setdefault(profile, {}).setdefault(seed, {})[method] = {
        "cost": data["best_cost"],
        "evaluations": data["complete_placement_evaluations"],
        "seconds": data["seconds_elapsed"],
        "metadata": data.get("algorithm_metadata"),
    }

summary = {"budget": budget, "profiles": {}, "protocol_pass": True}
for profile, by_seed in sorted(rows.items()):
    pairs = []
    for seed, methods in sorted(by_seed.items()):
        valid = set(methods) == {"operator", "operator_control", "proxy", "sa", "random"}
        valid &= all(row["evaluations"] == budget + 1 for row in methods.values())
        operator = methods["operator"]["cost"]
        improvements = {name: 100 * (methods[name]["cost"] - operator) /
                        methods[name]["cost"]
                        for name in ("operator_control", "proxy", "sa", "random")}
        pairs.append({"seed": seed, "costs": {k: v["cost"] for k, v in methods.items()},
                      "operator_improvement_percent": improvements,
                      "accounting_pass": valid})
        summary["protocol_pass"] &= valid
    proxy_values = [p["operator_improvement_percent"]["proxy"] for p in pairs]
    aggregate = {
        "seeds": len(pairs),
        "mean_operator_improvement_percent": {
            name: statistics.mean(
                p["operator_improvement_percent"][name] for p in pairs)
            for name in ("operator_control", "proxy", "sa", "random")},
        "operator_wins": {
            name: sum(p["operator_improvement_percent"][name] > 0 for p in pairs)
            for name in ("operator_control", "proxy", "sa", "random")},
    }
    aggregate["advancement_pass"] = (
        len(pairs) >= 3 and aggregate["operator_wins"]["proxy"] >= 2 and
        statistics.mean(proxy_values) >= 1.0 and
        aggregate["operator_wins"]["sa"] >= 2 and
        aggregate["operator_wins"]["random"] >= 2)
    summary["profiles"][profile] = {"pairs": pairs, "aggregate": aggregate}
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
