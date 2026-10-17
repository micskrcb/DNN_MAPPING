#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-amortized-bandit-gate}"
train_budget="${TRAIN_BUDGET_OVERRIDE:-20000}"
test_budget="${TEST_BUDGET_OVERRIDE:-20000}"
preflight_budget="${PREFLIGHT_BUDGET_OVERRIDE:-4000}"
heldout_model="${HELDOUT_MODEL_OVERRIDE:-resnet50}"
heldout_region="${HELDOUT_REGION_OVERRIDE:-conv}"
preflight_seed="${PREFLIGHT_SEED_OVERRIDE:-19}"
archive="${out}.zip"
seeds=(20 21 22)
[[ -n "${SEEDS_OVERRIDE:-}" ]] && read -r -a seeds <<<"${SEEDS_OVERRIDE}"
[[ "${train_budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid training budget"; exit 2; }
[[ "${test_budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid test budget"; exit 2; }
[[ "${preflight_budget}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid preflight budget"; exit 2; }
[[ ! -e "${out}" && ! -e "${archive}" ]] || { echo "Use a fresh output path"; exit 2; }
mkdir -p "${out}"
git rev-parse HEAD > "${out}/commit.txt"
git status --short > "${out}/git-status.txt"
python tests/run_all_tests.py > "${out}/tests.log" 2>&1

hardware=(
  --device cpu --cpu_threads 2 --use_cnn
  --partition_mode paper_targets --timing_model paper_pipeline
  --routing_model paper_xy --chips_x 4 --chips_y 4 --rows 16 --cols 16
  --sensitivity_trials 16
)
bandit=(
  --algo ppo_asa --ppo_asa_learner bandit
  --ppo_asa_candidates 16 --ppo_asa_bandit_feature_set topology16
  --ppo_asa_bandit_scoring proxy_residual
  --ppo_asa_bandit_residual_limit 0.25
  --ppo_asa_state_encoding relative_cost
  --ppo_asa_reward_mode positive_delta
  --ppo_asa_annealing fixed
  --ppo_asa_focus_fraction 0.5 --ppo_asa_focus_stage_slack 0.05
  --ppo_asa_bandit_exploration epsilon_top_m --ppo_asa_bandit_top_m 4
  --ppo_asa_bandit_global_epsilon 0.05
  --ppo_asa_bandit_epsilon_final 0.25
  --ppo_asa_bandit_epsilon_decay 1.0
  --ppo_asa_bandit_warmup 512 --ppo_asa_bandit_replay 50000
  --ppo_asa_rollout_steps 256 --ppo_update_epochs 4
  --ppo_minibatch_size 256 --ppo_hidden_dim 128
  --ppo_learning_rate 0.0003
)

checkpoint1="${out}/alexnet-fc.pt"
checkpoint2="${out}/alexnet-fc-conv.pt"

# A transfer comparison has no learning headroom when proxy and uniform reach
# the same target result.  Screen this before spending the training budget.
for method in proxy_greedy uniform; do
  extra=(--ppo_asa_disable_learning --ppo_asa_bandit_control "${method}")
  echo "===== ${heldout_model}-${heldout_region} preflight ${method} ====="
  PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
    "${hardware[@]}" "${bandit[@]}" "${extra[@]}" \
    --model "${heldout_model}" --workload_region "${heldout_region}" \
    --seed "${preflight_seed}" --iters "${preflight_budget}" \
    --report "${out}/preflight-${method}.json" \
    > "${out}/preflight-${method}.log" 2>&1
done

python - "${out}" "${preflight_budget}" <<'PY'
import json, sys
from pathlib import Path
root, budget = Path(sys.argv[1]), int(sys.argv[2])
rows = {name: json.loads((root / f"preflight-{name}.json").read_text())
        for name in ("proxy_greedy", "uniform")}
assert all(row["complete_placement_evaluations"] == budget + 1
           for row in rows.values()), "preflight accounting failed"
proxy, uniform = rows["proxy_greedy"]["best_cost"], rows["uniform"]["best_cost"]
result = {"budget": budget, "proxy_cost": proxy, "uniform_cost": uniform,
          "headroom_present": proxy != uniform,
          "proxy_improvement_percent": 100 * (uniform - proxy) / uniform}
(root / "preflight-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if not result["headroom_present"]:
    raise SystemExit("STOP: target has no proxy-vs-uniform headroom at preflight budget")
PY

echo "===== amortized training stage 1: AlexNet-FC ====="
PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
  "${hardware[@]}" "${bandit[@]}" \
  --model alexnet --workload_region fc --seed 101 \
  --iters "${train_budget}" --save_checkpoint "${checkpoint1}" \
  --report "${out}/train-alexnet-fc.json" \
  > "${out}/train-alexnet-fc.log" 2>&1

echo "===== amortized training stage 2: AlexNet-CONV ====="
PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
  "${hardware[@]}" "${bandit[@]}" \
  --model alexnet --workload_region conv --seed 102 \
  --iters "${train_budget}" --load_checkpoint "${checkpoint1}" \
  --ppo_asa_finetune_loaded --save_checkpoint "${checkpoint2}" \
  --report "${out}/train-alexnet-conv.json" \
  > "${out}/train-alexnet-conv.log" 2>&1

for seed in "${seeds[@]}"; do
  for method in frozen_checkpoint proxy_greedy uniform; do
    extra=(--ppo_asa_disable_learning)
    case "${method}" in
      frozen_checkpoint)
        extra+=(--ppo_asa_bandit_control checkpoint
                --load_checkpoint "${checkpoint2}")
        ;;
      proxy_greedy)
        extra+=(--ppo_asa_bandit_control proxy_greedy)
        ;;
      uniform)
        extra+=(--ppo_asa_bandit_control uniform)
        ;;
    esac
    echo "===== ${heldout_model}-${heldout_region} seed ${seed}, ${method} ====="
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      "${hardware[@]}" "${bandit[@]}" "${extra[@]}" \
      --model "${heldout_model}" --workload_region "${heldout_region}" --seed "${seed}" \
      --iters "${test_budget}" \
      --report "${out}/seed${seed}-${method}.json" \
      > "${out}/seed${seed}-${method}.log" 2>&1
  done
done

python - "${out}" "${train_budget}" "${test_budget}" "${heldout_model}-${heldout_region}" <<'PY'
import json, re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
train_budget, test_budget = map(int, sys.argv[2:4])
heldout_workload = sys.argv[4]
training = [json.loads((root / name).read_text()) for name in
            ("train-alexnet-fc.json", "train-alexnet-conv.json")]
training_pass = all(
    row["complete_placement_evaluations"] == train_budget + 1 and
    row["algorithm_metadata"]["update_count"] > 0 and
    row["algorithm_metadata"]["bandit_feature_set"] == "topology16"
    for row in training)

names = {"frozen_checkpoint", "proxy_greedy", "uniform"}
rows = {}
for path in root.glob("seed*.json"):
    match = re.fullmatch(
        r"seed(\d+)-(frozen_checkpoint|proxy_greedy|uniform)", path.stem)
    if match:
        rows.setdefault(int(match.group(1)), {})[match.group(2)] = json.loads(
            path.read_text())

pairs, protocol_pass = [], training_pass
for seed, methods in sorted(rows.items()):
    valid = set(methods) == names
    valid &= all(row["complete_placement_evaluations"] == test_budget + 1
                 for row in methods.values())
    frozen = methods["frozen_checkpoint"]["algorithm_metadata"]
    valid &= frozen["update_count"] == 0
    valid &= frozen["model_initialization"] == "loaded_checkpoint"
    valid &= frozen["proposal_policy"] == "frozen_checkpoint_contextual_bandit"
    costs = {name: methods[name]["best_cost"] for name in names}
    proxy_gain = 100 * (costs["proxy_greedy"] - costs["frozen_checkpoint"]) / costs["proxy_greedy"]
    uniform_gain = 100 * (costs["uniform"] - costs["frozen_checkpoint"]) / costs["uniform"]
    pairs.append({
        "seed": seed,
        "costs": costs,
        "checkpoint_improvement_over_proxy_percent": proxy_gain,
        "checkpoint_improvement_over_uniform_percent": uniform_gain,
        "accounting_pass": valid,
    })
    protocol_pass &= valid

proxy_gains = [row["checkpoint_improvement_over_proxy_percent"] for row in pairs]
uniform_gains = [row["checkpoint_improvement_over_uniform_percent"] for row in pairs]
summary = {
    "training_workloads": ["alexnet-fc", "alexnet-conv"],
    "heldout_workload": heldout_workload,
    "train_budget_per_stage": train_budget,
    "test_budget_per_method_seed": test_budget,
    "training_pass": training_pass,
    "protocol_pass": protocol_pass,
    "pairs": pairs,
    "mean_improvement_over_proxy_percent": statistics.mean(proxy_gains),
    "wins_over_proxy": sum(value > 0 for value in proxy_gains),
    "mean_improvement_over_uniform_percent": statistics.mean(uniform_gains),
    "wins_over_uniform": sum(value > 0 for value in uniform_gains),
}
summary["advancement_pass"] = (
    len(pairs) >= 3 and protocol_pass and
    summary["wins_over_proxy"] >= 2 and
    summary["wins_over_uniform"] >= 2 and
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
