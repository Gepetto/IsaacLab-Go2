import gymnasium as gym

from . import agents, privileged_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Go2-Lidar-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": privileged_env_cfg.Go2PrivilegedEnvCfg,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Lidar-Velocity-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": privileged_env_cfg.Go2PrivilegedEnvCfg_PLAY,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)
