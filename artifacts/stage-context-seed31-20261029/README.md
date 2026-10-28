# Stage-context seed31 result

Source kaggle-stage-context.zip, SHA256 27e002fa23429bf38a3e3290eddb1230b8ee1ec8ad33e03f85b6aa397ef060cc. Revision e562ee6. All arms complete/exit0, 200001 evaluations, matching initial cost6.800 us.

Pool-context4.28048 us; stage-context4.33872 us; fixedSA4.37616 us. Stage loses1.36060% versus pool. Gains versusSA: pool2.18639%, stage0.85554%. Both learners780 updates/12480 optimizer steps. Total measured runtime6.53703 hours. Stage led near50k but trailed at100k/150k/180k/200k. Features remained nonconstant late; mere feature variance did not imply usefulness.

Stage neutral proposals103477 vs pool77296; improving2770 vs2913. This motivates but does not establish a neutral-learning failure mechanism. Stage24 rejected; pool18/16 retained as development candidate. Raw traces remain in source ZIP; reports retained here. No published-DDPG comparison.
