import gymnasium as gym

from go2_locomotion.tasks.utils.cat.cat_env import CaTEnv

from . import agents, cat_flat_env_cfg, flat_env_cfg, flat_torque_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Go2-Velocity-Blind-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Blind-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Blind-Flat-Torque-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_torque_env_cfg.Go2FlatEnvCfg_PLAY,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Blind-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Blind-Rough-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-CaT-Velocity-Blind-Flat",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cat_flat_env_cfg.Go2FlatEnvCfg,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2FlatPPOCaTRunnerCfg",
    },
)

gym.register(
    id="Go2-CaT-Velocity-Blind-Flat-Play",
    entry_point=CaTEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cat_flat_env_cfg.Go2FlatEnvCfg_PLAY,
        "clean_rl_cfg_entry_point": f"{agents.__name__}.clean_rl_ppo_cfg:Go2FlatPPOCaTRunnerCfg",
    },
)
