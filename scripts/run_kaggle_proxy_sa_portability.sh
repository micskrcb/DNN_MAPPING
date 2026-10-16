#!/usr/bin/env bash
set -euo pipefail

out="${1:-runs/kaggle-proxy-sa-portability}"
budget="${BUDGET_OVERRIDE:-2000}"
archive="${out}.zip"
[[ "${budget}" =~ ^[1-9][0-9]*$ ]] || { echo "BUDGET_OVERRIDE must be positive"; exit 2; }
[[ ! -e "${out}" && ! -e "${archive}" ]] || { echo "Use a fresh output path"; exit 2; }
mkdir -p "${out}"

git rev-parse HEAD > "${out}/commit.txt"
git status --short > "${out}/git-status.txt"
python tests/run_all_tests.py | tee "${out}/tests.log"

run_profile() {
  local profile="$1"; shift
  for method in proxy_sa sa random; do
    local iterations="${budget}"
    [[ "${method}" == random ]] && iterations=$((budget + 1))
    echo "===== ${profile}: ${method} ====="
    PYTHONUNBUFFERED=1 python src/run_multi_chip.py \
      --algo "${method}" --iters "${iterations}" --seed 0 \
      --device cpu --sensitivity_trials 16 \
      --report "${out}/${profile}-${method}.json" "$@" \
      > "${out}/${profile}-${method}.log" 2>&1
  done
}

# Paper reconstruction: established AlexNet-FC graph and 4x4x16x16 mesh.
run_profile alexnet-fc-paper \
  --use_cnn --model alexnet --partition_mode paper_targets \
  --workload_region fc --timing_model paper_pipeline --routing_model paper_xy \
  --chips_x 4 --chips_y 4 --rows 16 --cols 16

# Different torchvision graph under generic full-frame timing.
run_profile resnet18-all-mesh \
  --use_cnn --model resnet18 --partition_mode uniform \
  --channels_per_partition 64 --workload_region all --timing_model full_frame \
  --routing_model legacy_distance --chips_x 4 --chips_y 4 --rows 8 --cols 8

# User-supplied graph and changed hardware/topology.
run_profile custom-cnn-torus \
  --use_cnn --model custom_portability \
  --custom_model examples/custom_portability_cnn.py --partition_mode uniform \
  --channels_per_partition 64 --workload_region all --timing_model full_frame \
  --routing_model legacy_distance --topology torus \
  --chips_x 3 --chips_y 2 --rows 8 --cols 8

python - "${out}" "${budget}" <<'PY'
import json, sys
from pathlib import Path

root, budget = Path(sys.argv[1]), int(sys.argv[2])
summary = {"purpose": "bounded portability smoke test", "budget": budget,
           "profiles": {}, "pass": True}
for path in sorted(root.glob("*.json")):
    if path.name == "summary.json":
        continue
    data = json.loads(path.read_text())
    profile, method = path.stem.rsplit("-", 1)
    meta = data.get("algorithm_metadata", {})
    expected = budget + 1
    row = {"best_cost": data["best_cost"],
           "objective_units": data["objective_units"],
           "evaluations": data["complete_placement_evaluations"],
           "seconds": data["seconds_elapsed"]}
    if method == "proxy_sa":
        row.update(unsupervised=meta.get("unsupervised"),
                   updates=meta.get("update_count"))
        valid = row["unsupervised"] is True and row["updates"] == 0
    else:
        valid = True
    valid = valid and row["evaluations"] == expected
    row["accounting_pass"] = valid
    summary["pass"] &= valid
    summary["profiles"].setdefault(profile, {})[method] = row
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if not summary["pass"]:
    raise SystemExit("Portability/accounting gate failed")
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
