# Offline repair of candidate ranking

The VGG16 frozen-transfer checkpoint lost to Proxy-SA on all three seeds.
The next test addresses the scorer directly; no new SA operator is introduced.

## Fixed protocol

- VGG16-CONV, reconstructed pipeline, execution VVA=1, partition VVA=1.
- Six independent Proxy-SA chains: seeds 30--33 train, 34--35 test.
- 4,100 proposals per chain; 16-candidate pools audited every 16 steps.
- Up to 256 pools/chain. All oracle calls are retained in accounting, including
  extra labels. Sensitivity sampling adds 16 nonselectable calls per chain.
- Same existing scorer architecture, 128 hidden units, 100 offline epochs,
  learning rate 0.0003; no test-based early stopping or model selection.
- Compare bounded positive-gain regression (current scoring design), direct
  signed pairwise ranking, and direct positive-part pairwise ranking.
- Report advantage over proxy and uniform, oracle capture, and harm frequency
  per held-out chain and in the late half. Measure unreachable positive targets.

The current-regressor comparison is retrained offline on the same full-pool
labels, not a rerun of its old selected-feedback checkpoint. Thus this test
isolates a scorer/target package, not training-distribution equivalence to the
old deployment. Two test chains give exploratory evidence only. Do not promote
an optimizer or infer a statistically established win from this diagnostic.
A positive result requires a separate frozen-rollout study. Failure should
inform representation changes, not repeated tuning on seeds 34--35.

## Run

```bash
bash scripts/run_kaggle_ranking_repair.sh runs/kaggle-ranking-repair
```

The runner writes a recoverable ZIP on ordinary exit/failure. Hard termination
may prevent its trap from running. It uses CPU (2 threads); GPU allocation is
not required if PyTorch and torchvision are installed. Runtime is unmeasured.
Use the first completed chain's duration to estimate the six-chain collection.

No external implementation was copied. Pairwise logistic ranking is implemented
locally. The planted-order test checks that its gradient learns a known ranking.
Local metric tests pass; neural tests require PyTorch and run on Kaggle.
