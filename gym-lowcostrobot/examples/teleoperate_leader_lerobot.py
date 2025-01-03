import argparse

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

def do_sim(args):

    env = gym.make(args.env_name, render_mode="human")

    offsets = [0, -np.pi/2, -np.pi/2, 0, -np.pi/2, 0]
    counts_to_radians = np.pi * 2. / 4096.
    # get the start pos from .cache/calibration directory in your local lerobot
    start_pos = [2063, 2042, 1058, 2131, 3078, 1945] # [2034, 3060, 4026, 1050, 2986, 3110] #[2015, 505, 3043, 2009, 3106, 1948]
    axis_direction = [-1, -1, 1, -1, -1, -1]
    joint_commands = [0,0,0,0,0,0]
    leader_arm = DynamixelMotorsBus(
        port=args.device,
        motors={
            # name: (index, model)
            "shoulder_pan": (1, "xl330-m077"),
            "shoulder_lift": (2, "xl330-m077"),
            "elbow_flex": (3, "xl330-m077"),
            "wrist_flex": (4, "xl330-m077"),
            "wrist_roll": (5, "xl330-m077"),
            "gripper": (6, "xl330-m077"),
        },
    )

    if not leader_arm.is_connected:
        leader_arm.connect()
    
    env.reset()
    rewards = []
    timesteps = []
  
    while env.unwrapped.viewer.is_running():
        positions = leader_arm.read("Present_Position")
        assert len(joint_commands) == len(positions)
        for i in range(len(joint_commands)):
            joint_commands[i] = axis_direction[i] * \
                (positions[i] - start_pos[i]) * counts_to_radians + offsets[i]
        
        ret = env.step(joint_commands)

        rewards.append(ret[1])
        timesteps.append(env.unwrapped.data.time)
            
    plt.plot(timesteps, rewards)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose between 5dof and 6dof lowcost robot simulation.")
    parser.add_argument('--device', type=str, default='/dev/tty.usbmodem58760433271', help='Port name (e.g., COM1, /dev/ttyUSB0, /dev/tty.usbserial-*)')
    # parser.add_argument('--env-name', type=str, default='PushCubeLoop-v0', help='Specify the gym-lowcost robot env to test.')
    parser.add_argument('--env-name', type=str, default='PickPlaceCube-v0', help='Specify the gym-lowcost robot env to test.')
    # parser.add_argument('--sim-config', type=str, default='ReachCube-v0.xml', help='Specify the gym-lowcost robot env to test.')
    args = parser.parse_args()

    do_sim(args)
