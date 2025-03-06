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

import os
from abc import ABC
import logging
from typing import List

from urdfpy import URDF
import torch

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class Robot(ABC):
    tcp_frame = "default_point"
    flange_frame = "ee_link"

    def __init__(self, urdf_filepath: str):
        with open(urdf_filepath) as urdf_file:
            self.urdf = URDF.load(urdf_file)

    def show(self, joint_states: torch.Tensor):
        self.urdf.show(joint_states.numpy(), use_collision=True)

    def fk(self, joint_states: torch.Tensor, frame: str) -> torch.Tensor:
        """
        joint_states: shape (num_joints), NOT batched!
        """
        affine = self.urdf.link_fk_torch(joint_states, frame)
        return affine

    @property
    def joint_names(self) -> List[str]:
        return self.urdf.actuated_joint_names

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


class RobotWithGripper(Robot):
    gripper_frames = ["gripper_link_left", "gripper_link_right"]

    @staticmethod
    def gripper_opening(gripper_pose_left: torch.Tensor, gripper_pose_right: torch.Tensor):
        gripper_pose_left_reshaped = gripper_pose_left.reshape((-1, 4, 4))  # Batch of 4x4 matrices
        gripper_pose_right_reshaped = gripper_pose_right.reshape((-1, 4, 4))  # Batch of 4x4 matrices
        position_left = gripper_pose_left_reshaped[:, :3, -1]
        position_right = gripper_pose_right_reshaped[:, :3, -1]
        return torch.norm(position_left - position_right)

    def gripper_opening_from_fk(self, joint_states: torch.Tensor) -> torch.Tensor:
        """
        :param joint_states: shape (num_joints) or (batch_size, num_joints)
        :return: shape (batch_size, 1)
        """
        orig_shape = joint_states.size()
        joint_states = joint_states.reshape(-1, orig_shape[-1])
        gripper_openings = []
        for js in joint_states:
            gripper_opening = self.gripper_opening(self.fk(js, self.gripper_frames[0]),
                                                         self.fk(js, self.gripper_frames[1]))
            gripper_openings.append(gripper_opening)
        return torch.stack(gripper_openings)


class UR5(Robot):
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)    # Suppress URDF warnings about face normals
        super(UR5, self).__init__(os.path.join(SCRIPT_DIR, "urdf", "ur5.urdf"))


class UR5Robotiq(RobotWithGripper):
    gripper_frames = ["finger_left_link_2", "finger_right_link_2"]

    def __init__(self, urdf_filepath: str = None):
        logging.basicConfig(level=logging.ERROR)    # Suppress URDF warnings about face normals
        urdf_filepath = os.path.join(SCRIPT_DIR, "urdf", "ur5_robotiq.urdf") if urdf_filepath is None else urdf_filepath
        super(UR5Robotiq, self).__init__(urdf_filepath)

    def gripper_opening_from_fk(self, joint_states: torch.Tensor) -> torch.Tensor:
        gripper_openings = super().gripper_opening_from_fk(joint_states)
        for i in range(len(gripper_openings)):
            gripper_openings[i] = gripper_openings[i] - 0.012  # Offset due to URDF finger origins
        return gripper_openings


class UR5RobotiqBullet(RobotWithGripper):
    gripper_frames = ["robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"]

    def __init__(self, urdf_filepath):
        logging.basicConfig(level=logging.ERROR)    # Suppress URDF warnings about face normals
        super(UR5RobotiqBullet, self).__init__(urdf_filepath)

    def gripper_opening_from_fk(self, joint_states: torch.Tensor) -> torch.Tensor:
        gripper_openings = super().gripper_opening_from_fk(joint_states)
        for i in range(len(gripper_openings)):
            gripper_openings[i] = gripper_openings[i]
        return gripper_openings


class RobotiqGripper(RobotWithGripper):
    gripper_frames = ["finger_left_link_2", "finger_right_link_2"]
    
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)    # Suppress URDF warnings about face normals
        super(RobotiqGripper, self).__init__(os.path.join(SCRIPT_DIR, "urdf", "robotiq_85.urdf"))

    def gripper_opening_from_fk(self, joint_states: torch.Tensor) -> torch.Tensor:
        gripper_openings = super().gripper_opening_from_fk(joint_states)
        for i in range(len(gripper_openings)):
            gripper_openings[i] = gripper_openings[i] - 0.012  # Offset due to URDF finger origins
        return gripper_openings
