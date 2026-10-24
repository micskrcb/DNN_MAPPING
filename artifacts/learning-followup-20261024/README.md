# Learning follow-up: 24 October 2026

Source: `kaggle-learning-followup-20261024T044809Z.zip`

SHA256: `13c8c2d70631c46c56a804c2bb170131c18122505a6df01592b07ea30525c448`

Reports, summaries, provenance and test/completion logs are preserved. Large checkpoints and per-step logs remain in the original archive; no checkpoints were deserialized. Both stages exited successfully at revision2828508.

VGG16-FC seed19: 100001 evaluations per arm; exploit4.34080µs, fast_exploit4.34352µs, fixedSA4.42912µs. Existing exploit beats SA1.99408%; faster decay is0.06266% worse than exploit.389 update events each. Single-seed comparison, not published-baseline evidence.

DDPG seed20:30 placements per arm; both retained6.07168µs from random baseline. Clipped final critic loss39.448 vs8655.256, deterministic6.864 vs7.296µs. Clipped loss was3438 at episode20, so endpoint improvement is not proof of stable convergence. Both still repair nearly every placement action. No DDPG optimization win.
