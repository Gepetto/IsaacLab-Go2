import gymnasium as gym

from . import agents, go2_bling_l2t_env_cfg, lidar_env_cfg, privileged_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Go2-Privileged-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": privileged_env_cfg.Go2PrivilegedEnvCfg,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Privileged-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": privileged_env_cfg.Go2PrivilegedEnvCfg_PLAY,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Lidar-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lidar_env_cfg.Go2LidarEnvCfg,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Lidar-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": lidar_env_cfg.Go2LidarEnvCfg_PLAY,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2PrivilegedPPORunnerCfg",
    },
)

gym.register(
    id="Go2-L2T-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_bling_l2t_env_cfg.Go2L2TTestEnvCfg,
        "rlopt_cfg_entry_point": f"{agents.__name__}:rlopt_l2t_cfg.yaml",
    },
)

gym.register(
    id="Go2-L2T-Velocity-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_bling_l2t_env_cfg.Go2L2TTestEnvCfg_PLAY,
        "rlopt_cfg_entry_point": f"{agents.__name__}:rlopt_l2t_cfg.yaml",
    },
)
