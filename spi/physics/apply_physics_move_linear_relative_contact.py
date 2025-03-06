"""
This file is part of DPSE 

Copyright (C) 2025 ArtiMinds Robotics GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from spi.common.pose import Pose
from spi.common.trajectory import Trajectory

from spi.physics.constant_acceleration_model import ConstantAccelerationModel
from spi.physics.physics_utils import add_sensor_noise, inside_circle


Z_FORCE_IDX = 9


def apply_physics(config: dict, simulated_traj: Trajectory, z_height: float, min_force: float, max_force: float,
                  sampling_interval: float = 0.032) -> tuple:
    traj_as_list = simulated_traj.to_list()
    spring_constant = config["spring_constant"]

    brake_start_idx = len(traj_as_list)
    for i in range(1, len(traj_as_list)):
        if traj_as_list[i][2] < z_height:
            displacement = z_height - traj_as_list[i][2]
            force_z = spring_constant * displacement
            traj_as_list[i][Z_FORCE_IDX] = force_z
            if force_z > min_force:
                # Simulate braking with constant acceleration model
                brake_start_idx = i
                break
    end_idx = decelerate(brake_start_idx, config, sampling_interval, spring_constant, traj_as_list, z_height)
    traj = Trajectory.from_list(traj_as_list[:end_idx])
    add_sensor_noise(traj.poses, traj.force_torques,
                     config["force_generation"]["move_linear_relative"]["position_encoder_noise"]["distribution"],
                     config["force_generation"]["move_linear_relative"]["ft_sensor_noise"]["distribution"])
    success = min_force < traj.force_torques[-1].fz < max_force
    return traj, success


def apply_physics_with_hole(config: dict, simulated_traj: Trajectory, hole_pose: Pose, hole_radius: float,
                            min_force: float, max_force: float, sampling_interval: float = 0.032) -> tuple:
    traj_as_list = simulated_traj.to_list()
    spring_constant = config["spring_constant"]
    z_height = hole_pose.position.z

    brake_start_idx = len(traj_as_list)
    for i in range(1, len(traj_as_list)):
        if traj_as_list[i][2] < z_height and not inside_circle(traj_as_list[i][0], traj_as_list[i][1],
                                                               hole_pose.position.x, hole_pose.position.y, hole_radius):
            displacement = z_height - traj_as_list[i][2]
            force_z = spring_constant * displacement
            traj_as_list[i][Z_FORCE_IDX] = force_z
            if force_z > min_force:
                # Simulate braking with constant acceleration model
                brake_start_idx = i
                break

    end_idx = decelerate(brake_start_idx, config, sampling_interval, spring_constant, traj_as_list, z_height)
    traj = Trajectory.from_list(traj_as_list[:end_idx])
    add_sensor_noise(traj.poses, traj.force_torques,
                     config["force_generation"]["move_linear_relative"]["position_encoder_noise"]["distribution"],
                     config["force_generation"]["move_linear_relative"]["ft_sensor_noise"]["distribution"])
    success = traj.force_torques[-1].fz < max_force
    traj.success_label = success
    # print(f"min_force: {min_force:.4f}, traj.force_torques[-1].fz: {traj.force_torques[-1].fz:.4f}, max_force: {max_force:.4f}")
    return traj, success


def decelerate(brake_start_idx, config, sampling_interval, spring_constant, traj_as_list, z_height):
    end_idx = brake_start_idx
    if brake_start_idx < len(traj_as_list):
        vel_z = (traj_as_list[brake_start_idx][2] - traj_as_list[brake_start_idx - 1][2]) / sampling_interval
        cam = ConstantAccelerationModel(sampling_interval)
        # vel_z is negative (motion in -Z), so deceleration is positive
        for i, (z_cam, vel_z_new) in enumerate(cam.simulate(config["braking_deceleration"],
                                                            traj_as_list[brake_start_idx][2], vel_z)):
            if vel_z_new > 0 or end_idx == len(traj_as_list):
                break
            displacement = z_height - z_cam
            force_z = spring_constant * displacement
            traj_as_list[brake_start_idx + i][2] = z_cam
            traj_as_list[brake_start_idx + i][Z_FORCE_IDX] = force_z
            end_idx += 1
    return end_idx
