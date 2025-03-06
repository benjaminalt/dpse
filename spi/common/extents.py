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

pose_dimensions = ["x", "y", "z", "qw", "qx", "qy", "qz"]

def world_origin_list(extents: dict) -> list:
    return [extents["origin"][dim] for dim in pose_dimensions]

def world_origin_pose(extents: dict) -> Pose:
    return Pose.make_absolute_from_xyz_quaternion(world_origin_list(extents))

def trajectory_limits(extents: dict) -> dict:
    return {
        "min": [extents["limits"]["distance"][0]] * 3 + [extents["limits"]["force"][0]] * 3 + [extents["limits"]["torque"][0]] * 3,
        "max": [extents["limits"]["distance"][1]] * 3 + [extents["limits"]["force"][1]] * 3 + [extents["limits"]["torque"][1]] * 3
    }