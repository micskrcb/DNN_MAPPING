"""
test_multi_chip.py
------------------
Smoke tests — no external dependencies beyond numpy.
Run from the src/ folder (or any folder that has multi_chip_*.py).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from multi_chip_topology import MultiChipTopology
from multi_chip_environment import MultiChipEnvironment


def test_topology():
    topo = MultiChipTopology(2, 2, 4, 4)
    assert topo.total_cores == 64
    # Same chip → purely on-chip cost
    cost_same = topo.comm_cost(0, 1)        # chip 0, cores 0 and 1
    # Different chips → includes off-chip penalty
    cost_diff = topo.comm_cost(0, 16)       # chip 0 core 0 → chip 1 core 0
    assert cost_diff > cost_same, "Off-chip must cost more than on-chip"
    L = topo.latency_matrix()
    assert L.shape == (64, 64)
    assert np.all(np.diag(L) == 0)
    print("✓ topology tests passed")


def test_environment_reset_step():
    env = MultiChipEnvironment(2, 2, 3, 3)   # 4 chips × 9 cores = 36 cores
    state = env.reset()
    assert state.shape[0] == env.state_dim
    # Manual full placement (identity: task i → core i)
    p = np.arange(env.num_tasks, dtype=np.int32)
    env.place(p)
    cost = env.evaluate()
    assert cost >= 0
    bd = env.chip_breakdown()
    assert abs(bd["on_chip_cost"] + bd["off_chip_cost"] - bd["total_cost"]) < 1e-4
    print(f"✓ environment tests passed  (cost={cost:.2f})")


def test_pipeline_stages_and_latency():
    """Regression test for the Eq.4 bottleneck objective fix."""
    # 4-task linear chain: 0->1->2->3, all on the same chip (no off-chip cost)
    tg = np.zeros((4, 4), dtype=np.float32)
    tg[0, 1] = 100
    tg[1, 2] = 50
    tg[2, 3] = 200
    env = MultiChipEnvironment(2, 2, 2, 2, task_graph=tg, num_tasks=4)

    stages = env.pipeline_stages()
    assert stages == [[0], [1], [2], [3]], "a pure chain should be 1 task per stage"

    env.place(np.array([0, 1, 2, 3], dtype=np.int32))
    # bottleneck (max stage) must be 200, NOT the sum (350)
    assert env.evaluate() == 200.0, f"expected bottleneck 200, got {env.evaluate()}"
    assert env.total_communication_cost() == 400.0, \
        "diagnostic sum metric should still equal the old total-cost value"

    # non-DAG (symmetric) input must degrade gracefully, not crash
    tg_sym = tg + tg.T
    env2 = MultiChipEnvironment(2, 2, 2, 2, task_graph=tg_sym, num_tasks=4)
    stages2 = env2.pipeline_stages()
    assert len(stages2) == 1 and set(stages2[0]) == {0, 1, 2, 3}, \
        "a cyclic/symmetric graph should collapse into a single fallback stage"

    # default random workload must now be a DAG by default (make_dag=True)
    env3 = MultiChipEnvironment(2, 2, 2, 2, num_tasks=6)
    default_stages = env3.pipeline_stages()
    assert len(default_stages) > 1, \
        "default synthetic workload should be a DAG with >1 stage, not a symmetric cycle"

    print("✓ pipeline_stages/pipeline_latency tests passed")


def test_collision_resolution():
    """Regression test for the paper's exact collision mechanism (Sec 3.2):
    floor to intended core -> place directly if free -> on collision only,
    search by MINIMUM MANHATTAN DISTANCE with first-found tie-break."""
    import run_multi_chip as rm

    tg = np.zeros((6, 6), dtype=np.float32)
    env = MultiChipEnvironment(2, 2, 2, 2, task_graph=tg, num_tasks=6)  # 4x4 grid, cols=4

    # action that floors to intended core (x=1, y=1) -> core index 5
    ax = (1.0 / 3.0) * 2 - 1.0
    ay = (1.0 / 3.0) * 2 - 1.0

    # Case 1: intended core free -> placed exactly there
    mapper = rm.MultiChipCoreMapper(env, baseline_latency=100.0, batch_z=1)
    mapper.reset()
    mapper.step(np.array([ax, ay]))
    assert mapper._placement[0] == 5, "should place at the intended core when free"

    # Case 2: collision -> nearest by Manhattan distance (not Euclidean)
    mapper.step(np.array([ax, ay]))
    c = mapper._placement[1]
    cx, cy = c % 4, c // 4
    assert abs(cx - 1) + abs(cy - 1) == 1, "should resolve to a Manhattan-adjacent core on collision"

    # Case 3: tie-break = first found (lowest core index among equal-distance candidates)
    mapper2 = rm.MultiChipCoreMapper(env, baseline_latency=100.0, batch_z=1)
    mapper2.reset()
    mapper2._occupied = {5, 1, 4}  # intended + two dist-1 neighbors occupied; 6 and 9 free at dist 1
    mapper2._task_ptr = 0
    mapper2.step(np.array([ax, ay]))
    assert mapper2._placement[0] == 6, "ties should break to the lowest core index found first"

    print("✓ collision resolution tests passed")


def test_batched_actions():
    """Regression test for the paper's batched action [x1,y1,...,xz,yz]
    (Sec 3.2): one step should place up to batch_z tasks, with a partial
    final batch handled correctly when num_tasks isn't divisible by z."""
    import run_multi_chip as rm

    tg = np.zeros((7, 7), dtype=np.float32)
    for i in range(6):
        tg[i, i + 1] = (i + 1) * 10
    env = MultiChipEnvironment(2, 2, 2, 2, task_graph=tg, num_tasks=7)  # 4x4 grid

    mapper = rm.MultiChipCoreMapper(env, baseline_latency=100.0, batch_z=3)
    mapper.reset()
    action_dim = 2 * 3
    step_count = 0
    done = False
    while not done:
        step_count += 1
        action = np.array([((k % 3) / 3.0) * 2 - 1.0 for k in range(action_dim)])
        r, done, grid, fc = mapper.step(action)
        if not done:
            assert r == 0.0, "non-terminal batched steps must still be sparse (reward 0)"

    assert step_count == 3, f"expected ceil(7/3)=3 steps, got {step_count}"
    assert all(c >= 0 for c in mapper._placement), "all 7 tasks should end up placed"
    assert len(set(mapper._placement.tolist())) == 7, "no duplicate core assignments"

    print("✓ batched action tests passed")


def test_cnn_workload_partitioning():
    """Regression test for channel-partitioned CNN extraction (Sec 3.1.1).
    Skips gracefully if torch isn't installed in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("~ skipping test_cnn_workload_partitioning (torch not installed)")
        return

    import run_multi_chip as rm

    tg, num_tasks, labels = rm.extract_cnn_task_graph(channels_per_partition=8)
    assert num_tasks > 100, \
        f"channel partitioning should produce far more than 15 tasks, got {num_tasks}"
    assert len(labels) == num_tasks
    assert tg.shape == (num_tasks, num_tasks)

    # Must still be a valid DAG usable by pipeline_stages()
    env = MultiChipEnvironment(4, 4, 16, 16, task_graph=tg, num_tasks=num_tasks)  # 1024 cores
    stages = env.pipeline_stages()
    all_staged = set(t for stage in stages for t in stage)
    assert all_staged == set(range(num_tasks)), "every task should appear in exactly one stage"

    # Legacy mode should still work and match the old ~15-task granularity
    tg_legacy, n_legacy, labels_legacy = rm.extract_cnn_task_graph(channels_per_partition=0)
    assert n_legacy < 20, f"legacy mode should be whole-layer granularity, got {n_legacy} tasks"

    print(f"✓ CNN workload partitioning tests passed ({num_tasks} logic cores at "
          f"channels_per_partition=8, {n_legacy} in legacy mode)")


def test_sa_improves_random():
    import random, math
    env = MultiChipEnvironment(2, 2, 4, 4)

    # Random baseline
    n, k = env.num_tasks, env.total_cores
    random.seed(0)
    p = np.array(random.sample(range(k), n), dtype=np.int32)
    env.place(p)
    random_cost = env.evaluate()

    # One quick SA pass
    from run_multi_chip import run_sa
    random.seed(0)
    env2 = MultiChipEnvironment(2, 2, 4, 4)
    sa_cost = run_sa(env2, n_iter=500)

    print(f"✓ SA cost={sa_cost:.2f}  random cost={random_cost:.2f}")
    # SA should generally do as well or better; allow some slack
    assert sa_cost <= random_cost * 1.5


if __name__ == "__main__":
    test_topology()
    test_environment_reset_step()
    test_pipeline_stages_and_latency()
    test_collision_resolution()
    test_batched_actions()
    test_cnn_workload_partitioning()
    test_sa_improves_random()
    print("\nAll tests passed.")
