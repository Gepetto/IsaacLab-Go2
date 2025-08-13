# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import go2_locomotion.tasks.mdps.cat.commands as commands
import go2_locomotion.tasks.mdps.cat.constraints as constraints
import go2_locomotion.tasks.mdps.cat.curriculums as curriculums
import go2_locomotion.tasks.mdps.cat.events as events
import go2_locomotion.tasks.mdps.cat.terminations as terminations
from go2_locomotion.tasks.mdps.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm

"""
* Go2 Isaac joint names:
[
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

* Go2 Isaac body names:
[
    'base',
    'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
    'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
    'Head_upper', 'Head_lower',
    'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot',
    'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot'
]
"""


##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        velocity_deadzone=0.1,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.7, 0.7), ang_vel_z=(-0.78, 0.78)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*",
        ],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001), scale=0.25)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.1)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"], preserve_order=True)},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 100,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (-0.05, 0.05),
        },
    )

    # interval

    # set pushing every step, as only some of the environments are chosen
    # as in the isaacgym cat version
    push_robot = EventTerm(
        # Standard push_by_setting_velocity also works, but interestingly results
        # in a different gait
        func=events.push_by_setting_velocity_with_random_envs,
        mode="interval",
        is_global_time=True,
        interval_range_s=(0.0, 0.005),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )


"""
* Go2 Isaac joint names:
[
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
]

* Go2 Isaac body names:
[
    'base',
    'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
    'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
    'Head_upper', 'Head_lower',
    'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot',
    'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot'
]
"""


@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 15.0,
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
    joint_velocity = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={
            "limit": 25.0,
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
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={
            "limit": 2000.0,
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
    # action_rate = ConstraintTerm(
    #     func=constraints.action_rate,
    #     max_p=0.25,
    #     params={"limit": 150.0,
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 joint_names=[
    #                     "FL_hip.*", "FL_thigh.*", "FL_calf.*",
    #                     "FR_hip.*", "FR_thigh.*", "FR_calf.*",
    #                     "RL_hip.*", "RL_thigh.*", "RL_calf.*",
    #                     "RR_hip.*", "RR_thigh.*", "RR_calf.*",
    #                 ]
    #             )},
    # )

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
    #                     "FL_calf.*", "FR_calf.*", "RL_calf.*", "RR_calf.*",
    #                     "FL_hip.*", "FR_hip.*", "RL_hip.*", "RR_hip.*",
    #                     # "FL_thigh.*", "FR_thigh.*", "RL_thigh.*", "RR_thigh.*",
    #                     # "Head_upper", "Head_lower"
    #                 ]
    #             )},
    # )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=1.0,
        params={
            "limit": 125.0,
            "asset_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"]
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
            "limit": 0.1,
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
    # base_orientation = ConstraintTerm(
    #     func=constraints.base_orientation,
    #     max_p=0.25,
    #     params={"limit": 0.75,
    #             "asset_cfg": SceneEntityCfg("robot")}
    # )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={
            "limit": 0.2,
            "velocity_deadzone": 0.1,
            "asset_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"]
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
    two_foot_contact = ConstraintTerm(
        func=constraints.n_foot_contact,
        max_p=0.25,
        params={
            "number_of_desired_feet": 2,
            "min_command_value": 0.5,
            "asset_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot.*", "FR_foot.*", "RL_foot.*", "RR_foot.*"]
            ),
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
        func=terminations.upside_down,
        params={"limit": 0.1},
    )


MAX_CURRICULUM_ITERATIONS = 1000


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    joint_torque = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_velocity = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity",
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
    two_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "two_foot_contact",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )


##
# Environment configuration
##


@configclass
class Go2FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 150.0

        # simulation settings
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.max_position_iteration_count = 4
        self.sim.max_velocity_iteration_count = 1
        self.sim.bounce_threshold_velocity = 0.2
        self.sim.gpu_max_rigid_contact_count = 33554432
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class Go2FlatEnvCfg_PLAY(Go2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.78, 0.78)
