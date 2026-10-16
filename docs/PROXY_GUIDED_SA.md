# Proxy-guided simulated annealing

`--algo proxy_sa` is a training-free search method. At every SA step it builds
several legal neighboring placements and ranks them using a cheap local proxy:
the estimated reduction in communication bytes multiplied by routing distance.
It usually selects a high-ranked candidate, retains controlled exploration, and
uses the configured full latency objective for the Metropolis accept/reject
decision. The proxy proposes; the evaluator decides.

It is **unsupervised** in the practical sense: it uses no labeled placements,
pretrained weights, checkpoints, or gradient updates. It is not a neural RL
method. It adapts automatically to each input graph and configured topology.

## Portability contract

The code can extract built-in `simple`, FX-traceable torchvision classifiers,
or a custom PyTorch model supplied through `--custom_model`. The custom file
must define `build_model()` and return either a model or `(model, dummy_input)`.
Current graph extraction models `Conv2d` and `Linear` partitions and traced
dependencies. Grouped convolution, concatenation timing, dynamic control flow,
and non-CNN operators may require extractor work.

Hardware is a configurable simulated rectangular multi-chip mesh or torus:
chip counts, cores per chip, bandwidths, link penalties, VVA rate, and routing
mode are parameters. Arbitrary accelerator topologies, memory hierarchies, and
measured device-specific latency are not inferred automatically.

For a general model use `--partition_mode uniform --workload_region all
--timing_model full_frame`. Paper-specific partition totals and pipeline timing
are valid only for the documented AlexNet/VGG16/ResNet50 paper reconstruction.

Run the bounded portability check with:

```bash
bash scripts/run_kaggle_proxy_sa_portability.sh runs/proxy-sa-portability
```

This verifies execution and accounting on multiple model/hardware settings; it
is not a claim that one parameter setting is optimal for every accelerator.
