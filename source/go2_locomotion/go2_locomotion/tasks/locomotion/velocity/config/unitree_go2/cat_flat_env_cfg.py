import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import go2_locomotion.tasks.locomotion.velocity.mdp as mdp
import go2_locomotion.tasks.utils.cat.constraints as constraints
import go2_locomotion.tasks.utils.cat.curriculums as curriculums
from go2_locomotion.tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm

from .go2_base_env_cfg import JOINT_NAMES, GO2BaseEnvCfg


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.001)
    # base_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-0.5,
    #     params={
    #         "target_height": 0.25,
    #     }
    # )
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.05)
    # base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.005)
    # joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.005)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)
    # joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-5e-5)
    # action_pen = RewTerm(func=mdp.action_l2, weight=-0.001)
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.08,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )
    energy = RewTerm(func=mdp.energy, weight=-2e-7)
    foot_pow = RewTerm(
        func=mdp.foot_power,
        weight=-1e-3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base"
                    # , 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip'
                ],
            ),
            "threshold": 1.0,
        },
    )

    upside_down = DoneTerm(
        func=mdp.upside_down,
        params={"limit": 0.1},
    )


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque_hip_thigh = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 20.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FL_hip_joint",
                    "FL_thigh_joint",
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                ],
            ),
        },
    )
    joint_torque_calf = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 40.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FR_calf_joint",
                    "FL_calf_joint",
                    "RR_calf_joint",
                    "RL_calf_joint",
                ],
            ),
        },
    )

    joint_velocity_hip_tigh = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={
            "limit": 13.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FR_hip_joint",
                    "FR_thigh_joint",
                    "FL_hip_joint",
                    "FL_thigh_joint",
                    "RR_hip_joint",
                    "RR_thigh_joint",
                    "RL_hip_joint",
                    "RL_thigh_joint",
                ],
            ),
        },
    )
    joint_velocity_calf = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={
            "limit": 13.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FR_calf_joint",
                    "FL_calf_joint",
                    "RR_calf_joint",
                    "RL_calf_joint",
                ],
            ),
        },
    )
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={
            "limit": 1500.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
            ),
        },
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={
            "limit": 100.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "FL_hip.*",
                    "FL_thigh.*",
                    "FL_calf.*",
                    "FR_hip.*",
                    "FR_thigh.*",
                    "FR_calf.*",
                    "RL_hip.*",
                    "RL_thigh.*",
                    "RL_calf.*",
                    "RR_hip.*",
                    "RR_thigh.*",
                    "RR_calf.*",
                ],
            ),
        },
    )

    # Safety Hard constraints
    # Knee and base
    # contact = ConstraintTerm(
    #     func=constraints.contact,
    #     max_p=1.0,
    #     params={
    #             "asset_cfg": SceneEntityCfg(
    #                 "contact_forces",
    #                 body_names=[
    #                     "base",
    #                     # "Head_upper", "Head_lower",
    #                     "FL_calf.*", "FR_calf.*", "RL_calf.*", "RR_calf.*",
    #                 ]
    #             )},
    # )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=1.0,
        params={
            "limit": 180.0,
            "asset_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"],
            ),
        },
    )
    # front_hfe_position = ConstraintTerm(
    #     func=constraints.joint_position,
    #     max_p=1.0,
    #     params={"limit": 1.3,
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=[
    #                 "FL_thigh.*", "FR_thigh.*"
    #             ])},
    # )

    # This checks whether the robot is flipped over
    # (upside-down) by evaluating gravity
    # direction in the robot's local body frame.

    # Style constraints
    hip_position = ConstraintTerm(
        func=constraints.joint_position_when_moving_forward,
        max_p=0.25,
        params={
            "limit": 0.25,
            "velocity_deadzone": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "RL_hip.*",
                    "RR_hip.*",
                    "FL_hip.*",
                    "FR_hip.*",
                ],
            ),
        },
    )
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation, max_p=0.25, params={"limit": 0.1, "asset_cfg": SceneEntityCfg("robot")}
    )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={
            "limit": 0.3,
            "velocity_deadzone": 0.1,
            "asset_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"],
            ),
        },
    )
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=0.1,
        params={
            "velocity_deadzone": 0.1,
            "joint_vel_limit": 4.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=JOINT_NAMES,
            ),
        },
    )
    # two_foot_contact = ConstraintTerm(
    #     func=constraints.n_foot_contact,
    #     max_p=0.25,
    #     params={
    #         "number_of_desired_feet": 2,
    #         "min_command_value": 0.5,
    #         "asset_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"],
    #         ),
    #     },
    # )
    diag_foot_contact = ConstraintTerm(
        func=constraints.diag_foot_contact,
        max_p=0.25,
        params={
            "number_of_desired_diagonals": 1,
            "min_command_value": 0.5,
            "asset_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FR_foot.*", "FL_foot.*", "RR_foot.*", "RL_foot.*"],
            ),
        },
    )
    # Other constraints:
    base_height = ConstraintTerm(
        func=constraints.base_height,
        max_p=1.0,
        params={
            "low_bound": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
        },
    )


MAX_CURRICULUM_ITERATIONS = 1000


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    # joint_torque = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_torque",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    joint_torque_hip_thigh = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque_hip_thigh",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_torque_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque_calf",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity_hip_tigh = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity_hip_tigh",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity_calf",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_acceleration = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # action_rate = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "action_rate",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )

    # Style constraints
    hip_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # base_orientation = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "air_time",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # two_foot_contact = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "two_foot_contact",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    diag_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "diag_foot_contact",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )


@configclass
class CaTGo2FlatEnvCfg(GO2BaseEnvCfg):

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


class CaTGo2FlatEnvCfg_PLAY(CaTGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
