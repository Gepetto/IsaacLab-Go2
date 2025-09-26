import time
import numpy as np
import torch
from pynput import keyboard

import mujoco
import mujoco.viewer

"""
*** JOINT NAMES IN MUJOCO ID ORDER:
FL_hip_joint
FL_thigh_joint
FL_calf_joint
FR_hip_joint
FR_thigh_joint
FR_calf_joint
RL_hip_joint
RL_thigh_joint
RL_calf_joint
RR_hip_joint
RR_thigh_joint
RR_calf_joint

*** JOINT NAME ORDER IN ISAAC (What the network expect)
JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
"""

ROBOT = "go2"
ROBOT_SCENE = "./mujoco_assets/" + "scene.xml"
SIMULATE_DT = 0.005
VIEWER_DT = 0.02

MUJOCO_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = np.argsort(MUJOCO_TO_ISAAC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = torch.jit.load("./nn/model.pt")
policy.eval()

stand_up_pose = np.array([
    -0.1, 0.8, -1.5,
    0.1, 0.8, -1.5,
    -0.1, 1.0, -1.5,
    0.1, 1.0, -1.5
])

cmd_vel = np.zeros(3)
last_action = np.zeros(12)
robot_mode = "stand"

def mujoco_to_isaac(arr):
    """
    """
    return arr[MUJOCO_TO_ISAAC]

def isaac_to_mujoco(arr):
    """
    """
    return arr[ISAAC_TO_MUJOCO]

def pd_control(target_q, current_q, current_dq, kp=50.0, kd=0.5):
    """
    """
    return kp*(target_q-current_q) - kd*current_dq

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def construct_observation(mj_model, mj_data, cmd_vel, last_action):
    base_vel = mj_data.qvel[:3]
    quat = mj_data.qpos[3:7]
    projected_gravity = get_gravity_orientation(quat)
    q = mujoco_to_isaac(mj_data.qpos[7:])
    dq = mujoco_to_isaac(mj_data.qvel[6:])
    obs = np.concatenate([base_vel, projected_gravity, cmd_vel, q, dq, last_action])
    return np.expand_dims(obs.astype(np.float32), axis=0)

def on_press(key):
    global cmd_vel, robot_mode

    try:
        if key == keyboard.Key.space:
            robot_mode = "stand"
            print("Stand up triggered!")
        if key == keyboard.Key.up:
            cmd_vel[0] = 0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")
        if key == keyboard.Key.down:
            cmd_vel[0] = -0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")
        if key == keyboard.Key.left:
            cmd_vel[1] = 0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")
        if key == keyboard.Key.right:
            cmd_vel[1] = -0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")
        if hasattr(key, 'char') and key.char == ",":
            cmd_vel[2] = 0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")
        if hasattr(key, 'char') and key.char == ".":
            cmd_vel[2] = -0.5
            robot_mode = "walk"
            print("Walk mode")
            print(f"{cmd_vel = }")

    except AttributeError:
        pass

def on_release(key):
    global cmd_vel
    if key == keyboard.Key.esc:
        return False
    cmd_vel[:] = 0.0

def run_simulation(mj_model, mj_data, viewer):
    global last_action, robot_mode

    counter = 0

    while viewer.is_running():
        step_start = time.perf_counter()

        if robot_mode == "stand":
            mj_data.qpos[:3] = np.zeros_like(mj_data.qpos[:3])
            mj_data.qpos[2] = 0.33
            mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            mj_data.qpos[7:] = stand_up_pose
            mj_data.qvel[:] = 0.
            mj_data.ctrl[:] = 0.

        elif robot_mode == "walk":
            obs = construct_observation(mj_model, mj_data, cmd_vel, last_action)
            with torch.no_grad():
                action = policy(torch.tensor(obs, device=device)).cpu().numpy()
            last_action = action.squeeze()

            q = mj_data.qpos[7:]
            dq = mj_data.qvel[6:]
            print(dq)
            # Convert action back to mujoco order
            target_q = isaac_to_mujoco(last_action)
            mj_data.ctrl[:] = pd_control(stand_up_pose, q, dq)

        mujoco.mj_step(mj_model, mj_data)
        # if cmd_vel[0] > 0:
        #     counter += 1
        # if robot_mode == "walk" and counter < 100:
        #     time.sleep(0.5)
        # if counter > 100:
        #     exit(0)

        viewer.sync()

        time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

if __name__ == "__main__":
    # Keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    mj_model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
    print("Number of key frames: ", mj_model.nkey)
    print("Joint names in id order: ")
    for i in range(1, mj_model.njnt):
        print(mj_model.joint(i).name)

    print()
    print()
    print("mj_model.joint(1): ")
    print(f"{mj_model.joint(1)}")

    print()
    print()
    print('name of body 0: ', mj_model.body(0).name)
    print('name of body 1: ', mj_model.body(1).name)
    print('name of body 2: ', mj_model.body(2).name)
    print('name of body 3: ', mj_model.body(3).name)
    print('name of body 4: ', mj_model.body(4).name)
    print()
    print()
    print(f"{[mj_model.geom(i).name for i in range(mj_model.ngeom)] = }")
    print(f"{mj_model.nbody = }")
    print(f"{[mj_model.body(i).name for i in range(mj_model.nbody)] = }")
    print()

    mj_data = mujoco.MjData(mj_model)
    print(f"{mj_data.qpos = }")
    print(f"{type(mj_data.qpos) = }")
    print(f"{len(mj_data.qpos) = }")
    print()
    print(f"{mj_data.qvel = }")
    print(f"{type(mj_data.qvel) = }")
    print(f"{len(mj_data.qvel) = }")
    print()
    print(f"{mj_data.ctrl = }")
    print(f"{len(mj_data.ctrl) = }")

    mj_model.opt.timestep = SIMULATE_DT
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    run_simulation(mj_model, mj_data, viewer)

    listener.stop()
