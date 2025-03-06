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

import pybullet as pb


def load_urdf_pb(urdf_path, pose, static):
    flags = pb.URDF_USE_INERTIA_FROM_FILE
    if static:
        flags |= pb.URDF_ENABLE_SLEEPING
        flags |= pb.URDF_MERGE_FIXED_LINKS
    object_id = pb.loadURDF(urdf_path,
                            # basePosition=pose.position.parameters(),
                            # baseOrientation=pose.orientation.to_qxyzw(),
                            useFixedBase=static, flags=flags)
    pb.resetBasePositionAndOrientation(object_id, pose.position.parameters(), pose.orientation.to_qxyzw())
    return object_id