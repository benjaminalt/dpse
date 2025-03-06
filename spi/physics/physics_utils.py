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

from spi.common.position import Position
import numpy as np


def add_sensor_noise(poses: list, forces: list, position_encoder_noise_distribution: dict, ft_sensor_noise_distribution: dict):
    for pose in poses:
        pose.position = Position.from_parameters([pose.position.parameters()[idx] + np.random.normal(*position_encoder_noise_distribution[dim]) for idx, dim in enumerate(["x","y","z"])])
        # TODO: Add sensor noise for orientation
    for force in forces:
        force.fx = force.fx + np.random.normal(*ft_sensor_noise_distribution["x"])
        force.fy = force.fy + np.random.normal(*ft_sensor_noise_distribution["y"])
        force.fz = force.fz + np.random.normal(*ft_sensor_noise_distribution["z"])


def inside_circle(x: float, y: float, c_x: float, c_y: float, r: float):
    return ((x - c_x) ** 2 + (y - c_y) ** 2) < r ** 2