"""Experimental active-region CNN input; all normal runner CLI arguments apply.

Only crops constant masked cells outside the allowed cores' bounding rectangle.
Physical coordinates, collision repair and objective evaluation remain unchanged.
This representation ablation is not asserted to be the authors' implementation.
"""
import numpy as np
import run_multi_chip as runner

OriginalAgent = runner.DDPGAgent
OriginalMapper = runner.MultiChipCoreMapper
original_run_ddpg = runner.run_ddpg
active_shape = None


class CompactMapper(OriginalMapper):
    def _occ_map(self):
        state = super()._occ_map()
        count = self.total_rows * self.total_cols
        grid = state[:count].reshape(self.total_rows, self.total_cols)
        crop = grid[self._region_min_y:self._region_max_y + 1,
                    self._region_min_x:self._region_max_x + 1]
        return np.concatenate((crop.ravel(), state[count:]))


class CompactAgent(OriginalAgent):
    normalization_experiment = 'compact-region-actor-bn-v1'

    def __init__(self, *args, **kwargs):
        if active_shape is None:
            raise ValueError('Compact runner requires an active region')
        rows, cols = active_shape
        kwargs.update(rows=rows, cols=cols,
                      state_dim=rows * cols + kwargs['num_tasks'])
        super().__init__(*args, **kwargs)

    def state_dict(self):
        state = super().state_dict()
        state['normalization_experiment'] = self.normalization_experiment
        return state

    def load_state_dict(self, state):
        if state.get('normalization_experiment') != self.normalization_experiment:
            raise ValueError('Expected a compact-region experiment checkpoint')
        super().load_state_dict(state)


def run_compact(env, *args, **kwargs):
    global active_shape
    mapper = CompactMapper(env, baseline_latency=1.0)
    active_shape = (mapper._region_max_y - mapper._region_min_y + 1,
                    mapper._region_max_x - mapper._region_min_x + 1)
    print(f'[EXPERIMENT] Active-region policy grid: {active_shape}')
    return original_run_ddpg(env, *args, **kwargs)


if __name__ == '__main__':
    runner.MultiChipCoreMapper = CompactMapper
    runner.DDPGAgent = CompactAgent
    runner.run_ddpg = run_compact
    runner.main()
