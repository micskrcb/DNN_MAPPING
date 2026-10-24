# Third-party notices

## RL-Based-SA-Public

Repository: <https://github.com/nathanqiu07/RL-Based-SA-Public>

Audited revision: `6e64bec12b9a7c6db601a4a4ae254b942e4aed79`

This project independently adapts the PPO training structure and formulas from
the repository's `neuralsa/training/ppo.py` to a different placement state,
action space, and objective. The original license follows.

> Copyright (c) 2023 Qualcomm Technologies, Inc.
>
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted (subject to the limitations in the disclaimer
> below) provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
> this list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
> this list of conditions and the following disclaimer in the documentation
> and/or other materials provided with the distribution.
>
> * Neither the name of Qualcomm Technologies, Inc. nor the names of its
> contributors may be used to endorse or promote products derived from this
> software without specific prior written permission.
>
> NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
> THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
> CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
> NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
> PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
> EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
> PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
> OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
> WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
> OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
> ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Communication-group spatial proposals (24 October 2026)

`src/structured_moves.py` is original project code inspired by the structured-search principle in Song Wang et al., *A Dimensional Order Search Based Mapping Algorithm for Deploying DNNs onto Many-Core Neuromorphic IC*, IEEE TCAD, DOI: https://doi.org/10.1109/TCAD.2025.3640508. No third-party implementation was copied. This bounded graph-group reordering is not a reproduction of DOS, its four-dimensional tensor mapping, or its TianjicX communication scheduling. See `docs/STRUCTURED_BANDIT.md` for the precise adaptation and limitations.
