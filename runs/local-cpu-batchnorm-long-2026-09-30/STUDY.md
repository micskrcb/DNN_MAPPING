# Long-run failure diagnostic

Seed 4 resumed from 12 to 100 complete training placements using commit
9499fb7, two CPU threads, potential shaping, paper_cnn and retained deterministic
evaluations. The checkpoint restores optimizer state but not replay; this is a
staged diagnostic, not an uninterrupted experiment.

The run completed. Best latency stayed at 37.635625 microseconds. At placement
100, noisy latency was 48.50671875 microseconds and deterministic latency was
48.081328125 microseconds. The deterministic actor requested only three unique
cores and required 181 occupied-core repairs for 183 tasks. Its critic loss was
6243.17 at that placement, after much higher intermediate values.

This contradicts a claim of stable learning from the earlier 12-placement
diagnostic. Keeping the best early placement protects the returned solution,
but does not show that the final policy improved. The short-run BatchNorm result
must not be interpreted as a successful learning repair.

Next hypothesis: hard-copying online BatchNorm statistics into slowly updated
target weights creates inconsistent target networks. A fresh, independently
fingerprinted run tests Polyak updates for floating-point buffers as well as
weights. This is an experiment, with no success claim until measured.
