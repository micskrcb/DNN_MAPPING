config = {
    "lut_info": {
        # Ensure 'toy_lut.txt' exists in your data/ folder
        "lut_file": "toy_lut.txt",

        "is_multicast": False,   # Set to True only if your toy_lut supports it
    },

    "env_config": {
        # TOPOLOGY: (chip_row, chip_col, core_row, core_col)
        # For Single Chip, chips = (1, 1). 
        # Grid = 16x16 = 256 physical cores. 
        # This is a huge search space (256!) for PPO to solve.
        "grid": (1, 1, 16, 16),

        "reward_config": {
            "reward_method": "Communication_cost",
            "deadlock_constraint": False,
            "deadlock_coef": 0.1,
        }    
    },

    # Setting mode to RL for Reinforcement Learning (PPO)
    "mode": "RL",

    "RS_config": {
        "repeat_num": 1000,
    },

    "SA_config": {
        "init_temp_coef": 120, 
        "n_iters": 500,        # Increased for bigger grid
        "gamma": 0.98,
        "temp_threshold": 0.1,
    },

    "RL_config": {
        "use_cuda": False,     # Set to True if you have an NVIDIA GPU
        "device": 0,    

        "batch_size": 128,     # Increased batch size for more stable gradients on large grids
        "ppo_epoch": 10,       # More epochs to learn the complex 256-core layout
        "ppo_clip": 0.2,       # Standard PPO clipping
        "lr": 0.0003,          # Slightly lower learning rate for better convergence in large spaces
    }
}