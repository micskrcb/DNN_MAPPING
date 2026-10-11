"""Train and evaluate a PPO-ASA candidate scorer from exact oracle costs.

The split is by complete, independently restarted annealing chains.  No pool
from a validation trajectory appears in training.  Targets are listwise
distributions derived from within-pool standardized exact gains.  This stage
diagnoses whether the actor features can represent useful proposal rankings
before another expensive RL run is attempted.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_multi_chip import ProposalPPOActor, ProposalPPOCritic


def interval_95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half_width = (1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
                  if len(values) > 1 else 0.0)
    return [mean - half_width, mean + half_width]


def soft_targets(gains, temperature):
    centered = gains - gains.mean(axis=1, keepdims=True)
    scale = gains.std(axis=1, keepdims=True)
    informative = scale[:, 0] > np.finfo(np.float64).eps
    normalized = np.zeros_like(centered)
    normalized[informative] = (
        centered[informative] / scale[informative] / temperature)
    normalized -= normalized.max(axis=1, keepdims=True)
    weights = np.exp(normalized)
    return weights / weights.sum(axis=1, keepdims=True), informative


def ranking_metrics(logits, gains):
    selected = logits.argmax(axis=1)
    selected_gains = gains[np.arange(len(gains)), selected]
    uniform_gains = gains.mean(axis=1)
    oracle_gains = gains.max(axis=1)
    advantage = selected_gains - uniform_gains
    oracle_advantage = oracle_gains - uniform_gains
    tolerance = np.maximum(
        np.abs(oracle_gains) * 1e-10, np.finfo(np.float64).eps * 8)
    optimal_mask = gains >= oracle_gains[:, None] - tolerance[:, None]
    informative = np.ptp(gains, axis=1) > tolerance
    if np.any(informative):
        top1 = float(np.mean(optimal_mask[
            np.arange(len(gains)), selected][informative]))
        random_top1 = float(np.mean(
            optimal_mask[informative].mean(axis=1)))
    else:
        top1 = random_top1 = 1.0
    oracle_mean = float(oracle_advantage.mean())
    selected_mean = float(advantage.mean())
    capture = selected_mean / oracle_mean if oracle_mean > 0 else None
    return {
        "pools": int(len(gains)),
        "informative_pools": int(informative.sum()),
        "mean_selected_gain_seconds": float(selected_gains.mean()),
        "mean_uniform_gain_seconds": float(uniform_gains.mean()),
        "mean_oracle_gain_seconds": float(oracle_gains.mean()),
        "mean_selected_advantage_over_uniform_seconds": selected_mean,
        "selected_advantage_95_percent_interval_seconds": interval_95(advantage),
        "mean_oracle_advantage_over_uniform_seconds": oracle_mean,
        "oracle_advantage_capture_fraction": capture,
        "top1_oracle_match_fraction_on_informative_pools": top1,
        "random_expected_oracle_match_fraction_on_informative_pools": random_top1,
    }


def chain_bootstrap_metrics(logits, raw_gains, scored_gains, group_ids, *,
                            progress=None, minimum_progress=0.0,
                            n_bootstrap=5000, seed=0):
    """Measure ranking transfer with complete trajectories as the sample unit."""
    logits = np.asarray(logits, dtype=np.float64)
    raw_gains = np.asarray(raw_gains, dtype=np.float64)
    scored_gains = np.asarray(scored_gains, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    if progress is None:
        progress = np.zeros(len(group_ids), dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64)
    mask = progress >= minimum_progress
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    rows = []
    for group in np.unique(group_ids[mask]):
        selected_rows = mask & (group_ids == group)
        if selected_rows.sum() < 20:
            continue
        scored = scored_gains[selected_rows]
        raw = raw_gains[selected_rows]
        group_logits = logits[selected_rows]
        group_probabilities = probabilities[selected_rows]
        selected = group_logits.argmax(axis=1)
        index = np.arange(len(selected))
        uniform = scored.mean(axis=1)
        oracle_advantage = scored.max(axis=1) - uniform
        oracle_total = float(oracle_advantage.sum())
        if oracle_total <= np.finfo(np.float64).eps:
            continue
        selected_advantage = scored[index, selected] - uniform
        sampled_advantage = (
            (group_probabilities * scored).sum(axis=1) - uniform)
        tolerance = np.maximum(
            np.abs(raw).max(axis=1) * 1e-10,
            np.finfo(np.float64).eps * 8)
        rows.append({
            "group": int(group),
            "pools": int(len(selected)),
            "capture_fraction": float(
                selected_advantage.sum() / oracle_total),
            "sampled_capture_fraction": float(
                sampled_advantage.sum() / oracle_total),
            "mean_selected_advantage_seconds": float(
                selected_advantage.mean()),
            "selected_improving_fraction": float(np.mean(
                raw[index, selected] > tolerance)),
            "candidate_improving_fraction": float(np.mean(
                raw > tolerance[:, None])),
        })
    if not rows:
        raise ValueError("no eligible trajectory groups for chain metrics")
    captures = np.asarray(
        [row["capture_fraction"] for row in rows], dtype=np.float64)
    sampled = np.asarray(
        [row["sampled_capture_fraction"] for row in rows], dtype=np.float64)
    selected_rates = np.asarray(
        [row["selected_improving_fraction"] for row in rows],
        dtype=np.float64)
    base_rates = np.asarray(
        [row["candidate_improving_fraction"] for row in rows],
        dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(
        0, len(rows), size=(n_bootstrap, len(rows)))
    bootstrap_capture = captures[choices].mean(axis=1)
    bootstrap_sampled = sampled[choices].mean(axis=1)
    mean_base = float(base_rates.mean())
    return {
        "minimum_progress": float(minimum_progress),
        "groups": int(len(rows)),
        "positive_capture_groups": int(np.sum(captures > 0)),
        "mean_capture_fraction": float(captures.mean()),
        "capture_chain_bootstrap_95_percent_interval": [
            float(value) for value in np.quantile(
                bootstrap_capture, [0.025, 0.975])],
        "mean_sampled_capture_fraction": float(sampled.mean()),
        "sampled_capture_chain_bootstrap_95_percent_interval": [
            float(value) for value in np.quantile(
                bootstrap_sampled, [0.025, 0.975])],
        "mean_selected_improving_fraction": float(selected_rates.mean()),
        "mean_candidate_improving_fraction": mean_base,
        "selected_to_candidate_improvement_ratio": (
            float(selected_rates.mean() / mean_base)
            if mean_base > 0 else None),
        "per_group": rows,
    }


def train_ranker(global_states, candidate_features, true_gains, *,
                 group_ids=None,
                 target_mode="signed_gain",
                 hidden_dim=128, epochs=200, batch_size=256,
                 learning_rate=3e-4, weight_decay=1e-4,
                 target_temperature=0.5, validation_fraction=0.2,
                 patience=30, seed=0, device="cpu"):
    if len(global_states) < 10:
        raise ValueError("oracle ranking needs at least ten candidate pools")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if min(hidden_dim, epochs, batch_size, patience) <= 0:
        raise ValueError("training sizes must be positive")
    if target_mode not in {"signed_gain", "improvement_only"}:
        raise ValueError("unknown oracle ranking target mode")
    torch_device = torch.device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if group_ids is None:
        group_ids = np.arange(len(global_states), dtype=np.int64)
    group_ids = np.asarray(group_ids)
    if group_ids.shape != (len(global_states),):
        raise ValueError("group_ids must contain one id per candidate pool")
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 2:
        raise ValueError(
            "oracle ranking needs at least two independent trajectory groups")
    group_permutation = rng.permutation(unique_groups)
    validation_group_count = min(
        len(unique_groups) - 1,
        max(1, round(len(unique_groups) * validation_fraction)))
    validation_groups = group_permutation[:validation_group_count]
    training_groups = group_permutation[validation_group_count:]
    validation_indices = np.flatnonzero(
        np.isin(group_ids, validation_groups))
    training_indices = np.flatnonzero(np.isin(group_ids, training_groups))
    scored_gains = (true_gains if target_mode == "signed_gain" else
                    np.maximum(true_gains, 0.0))
    targets, informative = soft_targets(scored_gains, target_temperature)
    informative_training = training_indices[informative[training_indices]]
    if len(informative_training) == 0:
        raise ValueError("oracle dataset has no informative training pools")

    global_tensor = torch.as_tensor(
        global_states, dtype=torch.float32, device=torch_device)
    candidate_tensor = torch.as_tensor(
        candidate_features, dtype=torch.float32, device=torch_device)
    target_tensor = torch.as_tensor(
        targets, dtype=torch.float32, device=torch_device)
    actor = ProposalPPOActor(
        global_states.shape[1], candidate_features.shape[2],
        hidden_dim=hidden_dim).to(torch_device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = None
    best_validation_loss = math.inf
    stale_epochs = 0
    history = []
    validation_index_tensor = torch.as_tensor(
        validation_indices, dtype=torch.long, device=torch_device)
    for epoch in range(1, epochs + 1):
        actor.train()
        shuffled = rng.permutation(informative_training)
        training_losses = []
        for start in range(0, len(shuffled), batch_size):
            indices = torch.as_tensor(
                shuffled[start:start + batch_size], dtype=torch.long,
                device=torch_device)
            logits = actor.distribution(
                global_tensor[indices], candidate_tensor[indices]).logits
            loss = -(target_tensor[indices] * F.log_softmax(
                logits, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))

        actor.eval()
        with torch.no_grad():
            validation_logits = actor.distribution(
                global_tensor[validation_index_tensor],
                candidate_tensor[validation_index_tensor]).logits
            validation_loss = float((
                -(target_tensor[validation_index_tensor] * F.log_softmax(
                    validation_logits, dim=-1)).sum(dim=-1).mean()).cpu())
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation_loss - 1e-7:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in actor.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    actor.load_state_dict(best_state)
    actor.to(torch_device).eval()
    with torch.no_grad():
        all_logits = actor.distribution(
            global_tensor, candidate_tensor).logits.cpu().numpy()
    return {
        "actor": actor,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "training_groups": training_groups,
        "validation_groups": validation_groups,
        "target_mode": target_mode,
        "all_logits": all_logits,
        "scored_gains": scored_gains,
        "training_metrics": ranking_metrics(
            all_logits[training_indices], scored_gains[training_indices]),
        "validation_metrics": ranking_metrics(
            all_logits[validation_indices], scored_gains[validation_indices]),
        "history": history,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "informative_training_pools": int(len(informative_training)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.5)
    parser.add_argument(
        "--target-mode", choices=["signed_gain", "improvement_only"],
        default="signed_gain",
        help="Rank signed exact gain or only positive improvement opportunity")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--strict-chain-gate", action="store_true",
        help="Require the prespecified full-chain improvement-only gate")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")

    with np.load(args.dataset, allow_pickle=False) as dataset:
        global_states = dataset["global_states"]
        candidate_features = dataset["candidate_features"]
        true_gains = dataset["true_gains"]
        if "chain_ids" not in dataset:
            raise ValueError(
                "oracle dataset lacks chain_ids; regenerate it with a "
                "trajectory-safe exporter")
        chain_ids = dataset["chain_ids"]
        candidate_count = int(dataset["candidate_count"])
    result = train_ranker(
        global_states, candidate_features, true_gains, group_ids=chain_ids,
        target_mode=args.target_mode,
        hidden_dim=args.hidden_dim, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_temperature=args.target_temperature,
        validation_fraction=args.validation_fraction,
        patience=args.patience, seed=args.seed, device=args.device)
    validation = result["validation_metrics"]
    interval = validation["selected_advantage_95_percent_interval_seconds"]
    validation_indices = result["validation_indices"]
    chain_all = chain_bootstrap_metrics(
        result["all_logits"][validation_indices],
        true_gains[validation_indices],
        result["scored_gains"][validation_indices],
        chain_ids[validation_indices],
        progress=global_states[validation_indices, 0],
        minimum_progress=0.0, n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 1)
    chain_late = chain_bootstrap_metrics(
        result["all_logits"][validation_indices],
        true_gains[validation_indices],
        result["scored_gains"][validation_indices],
        chain_ids[validation_indices],
        progress=global_states[validation_indices, 0],
        minimum_progress=0.5, n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 2)
    if args.strict_chain_gate:
        if args.target_mode != "improvement_only":
            raise ValueError(
                "the strict chain gate requires --target-mode improvement_only")
        lower_capture = chain_late[
            "capture_chain_bootstrap_95_percent_interval"][0]
        ratio = chain_late["selected_to_candidate_improvement_ratio"]
        passed = bool(
            chain_late["groups"] >= 10 and
            chain_late["mean_capture_fraction"] >= 0.10 and
            lower_capture >= 0.05 and
            ratio is not None and ratio >= 1.5 and
            chain_late["positive_capture_groups"] >= 8 and
            chain_late["mean_sampled_capture_fraction"] >= 0.05)
        criterion = (
            "on at least ten held-out full trajectories, late-half argmax "
            "capture >=10% with chain-bootstrap lower bound >=5%, selected "
            "improvement rate >=1.5x the candidate base rate, at least eight "
            "positive chains, and sampled-policy capture >=5%")
    else:
        passed = bool(
            interval[0] > 0 and
            validation["oracle_advantage_capture_fraction"] is not None and
            validation["oracle_advantage_capture_fraction"] > 0 and
            validation["top1_oracle_match_fraction_on_informative_pools"] >
            validation[
                "random_expected_oracle_match_fraction_on_informative_pools"])
        criterion = (
            "held-out selected advantage has a positive pool-level 95% "
            "interval, captures positive oracle headroom, and beats random "
            "top-1 matching")
    report = {
        "verdict": ("SUPERVISED_RANKING_SIGNAL" if passed
                    else "NO_SUPERVISED_RANKING_SIGNAL"),
        "pass": passed,
        "criterion": criterion,
        "target_mode": args.target_mode,
        "strict_chain_gate": args.strict_chain_gate,
        "dataset_pools": int(len(global_states)),
        "candidate_count": candidate_count,
        "training_pools": int(len(result["training_indices"])),
        "validation_pools": int(len(result["validation_indices"])),
        "dataset_groups": int(len(np.unique(chain_ids))),
        "training_groups": [
            int(value) for value in result["training_groups"]],
        "validation_groups": [
            int(value) for value in result["validation_groups"]],
        "informative_training_pools": result["informative_training_pools"],
        "epochs_completed": result["epochs_completed"],
        "best_validation_loss": result["best_validation_loss"],
        "training": result["training_metrics"],
        "validation": validation,
        "validation_chain_metrics": {
            "all": chain_all,
            "late_half": chain_late,
        },
        "history": result["history"],
    }

    critic = ProposalPPOCritic(
        global_states.shape[1], candidate_features.shape[2],
        hidden_dim=args.hidden_dim)
    checkpoint = {
        "algorithm": "ppo_asa_reference_separate_v2",
        "resumable": False,
        "initialization_type": "oracle_supervised_candidate_ranker",
        "evaluations": int(len(global_states) * candidate_count),
        "best_cost": None,
        "seed": args.seed,
        "candidate_count": candidate_count,
        "learning_enabled": True,
        "update_count": result["epochs_completed"],
        "hidden_dim": args.hidden_dim,
        "focus_fraction": 0.5,
        "restart_interval": 4100,
        "update_at_chain_end": False,
        "independent_chains": None,
        "update_style": "reference_separate",
        "reward_mode": "oracle_supervised",
        "supervised_target_mode": args.target_mode,
        "actor": result["actor"].cpu().state_dict(),
        "critic": critic.state_dict(),
        "supervised_report": report,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    with args.report.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved {args.checkpoint}")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
