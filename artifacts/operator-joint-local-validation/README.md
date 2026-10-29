# Operator joint local validation

Four focused tests (including reused pool tests) and51 regression tests passed. Real VGG16-FC CLI smoke:601 evaluations per arm,seed35,sensitivity_trials4. Both learners completed16 optimizer steps; frozen completed0. Strict analyzer passed; a mutation of frozen learning rate was rejected. Legality tests cover all four operator types, exact changed-task sets and unavailable-free-core fallback. Checkpoint feature mismatch rejects loading.

This reduced run is integration validation only, not research performance evidence. Full run starts anew at200001 evaluations with256 sensitivity trials. No performance-based tuning was made from these smoke costs.
