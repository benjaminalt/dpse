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

from typing import List

import torch
import numpy as np
from spi.common.orientation import Orientation
from spi.common.pose import Pose
from spi.common.force_torque import ForceTorque
from spi.neural_templates.neural_template_utils import success_probability
from pyquaternion import Quaternion

from spi.utils.pytorch_utils import pad_padded_sequence


class Trajectory(object):
    def __init__(self, poses, force_torques, gripper_states: List[float] = None, success_label: float = 1.0):
        self.poses = poses
        self.force_torques = force_torques
        self.gripper_states = gripper_states
        self.success_label = success_label

    @staticmethod
    def from_raw(raw_trajectory: list, reference_pose: Pose, success_label: float = 1.0):
        """
        :param raw_trajectory: List of RPS poses (xyz_rxryrz) in meters and radians
        Does NOT contain gripper states
        """
        poses = []
        force_torques = []
        for raw_dp in raw_trajectory:
            if reference_pose is not None:
                poses.append(Pose.make_relative_from_xyz_rxryrz(raw_dp[:6], reference_pose))
            else:
                poses.append(Pose.make_absolute_from_xyz_rxryrz(raw_dp[:6]))
            force_torques.append(ForceTorque.from_parameters(raw_dp[6:12]))
        return Trajectory(poses, force_torques, success_label=success_label)

    def to_raw(self):
        """
        Does NOT contain gripper states
        """
        return [self.poses[i].to_absolute_xyz_rxryrz() + self.force_torques[i].parameters() for i in range(len(self.poses))]

    @staticmethod
    def from_list(arr, success_label: float = 1.0):
        poses = []
        force_torques = []
        gripper_states = []
        for dp_params in arr:
            poses.append(Pose.from_parameters(dp_params[:7]))
            force_torques.append(ForceTorque.from_parameters(dp_params[7:13]))
            if len(dp_params) == 14:
                gripper_states.append(dp_params[13])
        return Trajectory(poses, force_torques, gripper_states if len(gripper_states) > 0 else None,
                          success_label=success_label)

    def to_list(self):
        lst = []
        for i in range(len(self.poses)):
            dp = self.poses[i].parameters() + self.force_torques[i].parameters()
            if self.gripper_states is not None:
                dp.append(self.gripper_states[i])
            lst.append(dp)
        return lst

    @staticmethod
    def from_tensor(tensor: torch.Tensor):
        if tensor.size(-1) > 7 + 6 + 1: # Pose + force + gripper
            # Trajectory contains meta information and may be padded
            unpadded = tensor[tensor[:, 0] < 0.5]
            # Always leave at least 1 datapoint
            if len(unpadded) == 0:
                unpadded = tensor[-1].unsqueeze(0)
            success_label = success_probability(unpadded)
            return Trajectory.from_list(unpadded[:, 2:].tolist(), success_label=success_label.item())
        return Trajectory.from_list(tensor.tolist())

    def to_tensor(self, meta_inf: bool = True, pad_to: int = None):
        if not meta_inf and pad_to is not None:
            raise RuntimeError("Cannot pad trajectory without meta information")
        base_tensor = torch.tensor(self.to_list(), dtype=torch.float32)
        if meta_inf:
            eos = torch.zeros(base_tensor.size(0), dtype=torch.float32).unsqueeze(-1)
            eos[-1, 0] = 1.0
            success = torch.tensor(self.success_label, dtype=torch.float32).view(1, 1).repeat(base_tensor.size(0), 1)
            traj_unpadded = torch.cat((eos, success, base_tensor), dim=-1)
            if pad_to is None:
                return traj_unpadded
            return pad_padded_sequence(traj_unpadded, pad_to)
        return base_tensor

    def smoothen_orientations(self):
        # Set first orientation to have positive w component
        if self.poses[0].orientation.q.w < 0:
            self.poses[0].orientation = Orientation(Quaternion(-self.poses[0].orientation.q))
        # Homogenize remaining orientations to avoid jumps
        for i in range(1, len(self.poses)):
            self.poses[i].orientation.smoothen(self.poses[i-1].orientation)

    def normalize_orientations(self):
        for pose in self.poses:
            pose.orientation.normalize()

    def scale(self, from_min, from_max, to_min, to_max):
        """
        Scale both poses and forces from the given range to the given range.
        :param from_min: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param from_max: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param to_min: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :param to_max: 9D-vector (3 pose dimensions, 6 FT dimensions)
        :return:
        """
        for i in range(len(self.poses)):
            self.poses[i].position.scale(from_min[:3], from_max[:3], to_min[:3], to_max[:3])
            self.force_torques[i].scale(from_min[3:], from_max[3:], to_min[3:], to_max[3:])

    def transform(self, affine_transformation):
        """
        Transform each pose in the trajectory by the given transformation.
        Forces are not transformed.
        """
        for i in range(len(self.poses)):
            self.poses[i].transform(affine_transformation)

    def path_length(self):
        """
        Return the sum of the euclidean distances between all successive poses on the trajectory
        """
        pl = 0.0
        for i in range(len(self.poses) - 1):
            p = self.poses[i]
            q = self.poses[i+1]
            pl += np.linalg.norm(np.array(p.position.parameters()) - np.array(q.position.parameters()))
        return pl

    def __len__(self):
        return len(self.poses)
