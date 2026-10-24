# DOS paper: applicability to our RL+SA project

Reviewed 24 October2026. Song Wang, Rui Lu, Bo Xiao, Zhaohui Wang, Yunpeng Guo, Xin Sun and Wenjian Yu, **A Dimensional Order Search Based Mapping Algorithm for Deploying DNNs onto Many-Core Neuromorphic IC**, IEEE TCAD. DOI: https://doi.org/10.1109/TCAD.2025.3640508. Source: user-supplied accepted-version PDF. It cites Wu etal TODAES2020 as reference13 and Myung etal policy-gradient placement as reference15. No external implementation copied. Targeted title/DOI/GitHub search did not locate an author code release; that is not proof none exists.

## What the method actually changes

SectionsIII-A/B and Algorithm1: permute logical partition dimensions (X,Y,Z,Q), choose physical mesh axis order, and permute local groups up to eight cores. This is a restricted, structured mapping family; its stated4!×2!×8! bound must not be described as exhaustive enumeration of every possible arbitrary graph placement. The paper also changes communication to dimension-specific time-sharing with TianjicX router assumptions. Its theoretical arguments apply within that formulation, not our arbitrary multi-chip graphs.

SectionV-C and TablesI–III compare routing cycles and communication energy. GA/SA receive five minutes per block; RL is described as running until convergence. The abstract calls reductions “mapping time”, but the experimental tables and discussion identify routing time. Do not use the quoted10.99× RL factor as a speedup in optimizer execution time. Baseline-order wording for VGG16/Inception differs between abstract and discussion; use table rows directly if recomputing ratios. These results do not establish how many epochs our learner requires.

## Useful adaptation, not direct reproduction

1. Make logical partition coordinates and communication type explicit; verify H/W/Cin/Cout semantics before using tensor-order moves. Current task ordering alone is insufficient to reproduce their4D decomposition.
2. Add a bounded family of structured group/axis reorderings to an RL-controlled proposal pool. Let learned scores choose among structured and existing moves. Evaluate every selected proposal with the unchanged objective.
3. Compare with the same structured pool selected uniformly or by an unlearned rule. Otherwise an apparent RL gain might just come from a better move generator.
4. Keep any time-sharing/router simulator experiment separate, with identical evaluator settings for all methods. Changing only the RL evaluator would invalidate comparison.

This is a promising subsequent action-space direction. It is not a drop-in patch: TianjicX buffering, time-sharing, partitioning and routing differ from our paper_pipeline/paper_xy reconstruction. No claim of beating DOS or publishedDDPG follows from current results. Do not add its assumed deadlock freedom or zero waiting delay to our simulator without implementing and validating those mechanisms.

## Immediate implementation decision

Keep the successful online bandit and fixedSA objective. First test whether four minibatch steps per update underfit the collected moves:16 steps, same100001 evaluation budget, AlexNet-FC and VGG16-FC. This is an independent learning-capacity hypothesis, not a result derived from DOS. It requires no new labels or simulator changes and exposes update count separately from actual optimizer steps. If unsuccessful, preserve the existing winner and pursue structured actions above, rather than repeatedly changing epsilon.
