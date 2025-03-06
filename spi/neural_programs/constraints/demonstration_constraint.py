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

import torch
from robots.robot import RobotWithGripper
from robots.robot_factory import RobotFactory
from spi.neural_templates.neural_template_utils import trajectory_length
from utils.transformations import quaternion_distance
from spi.neural_programs.constraints.constraint import ConstraintType, Constraint
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.utils.pytorch_utils import pad_padded_sequence


class DemonstrationConstraintCartesian(Constraint):
    def __init__(self, gripper_type: RobotWithGripper, demonstration: torch.Tensor, weight: float = 1.0,
                 include_pos: bool = True, include_ori: bool = True, include_gripper: bool = False):
        """
        :param demonstration: [meta|pose|ft|gripper_opening]
        """
        assert len(demonstration.size()) == 2 and demonstration.size(-1) == 2 + 7 + 6 + 1
        self.constraint_type = ConstraintType.TRAJECTORY
        self.demonstration = demonstration
        self.weight = weight
        self.target_length = trajectory_length(demonstration).float()
        self.gripper_type = gripper_type
        self.include_pos = include_pos
        self.include_ori = include_ori
        self.include_gripper = include_gripper

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        tl = trajectory_length(trajectory).float()
        max_length = int(max([self.target_length.item(), tl.item()]))
        padded_demonstration = pad_padded_sequence(self.demonstration, max_length)
        padded_trajectory = pad_padded_sequence(trajectory.squeeze(), max_length).unsqueeze(0)
        target_tensor = padded_demonstration.unsqueeze(0)

        # euclidean_dist = torch.norm(padded_trajectory[:, :, 2:5] - target_tensor[:, :, 2:5], dim=-1)
        # position_loss = euclidean_dist.mean()
        position_loss = torch.nn.MSELoss()(padded_trajectory[:, :, 2:5], target_tensor[:, :, 2:5]) if self.include_pos else torch.zeros(1)

        quaternion_dist = quaternion_distance(padded_trajectory[:, :, 5:9], target_tensor[:, :, 5:9])
        orientation_loss = 0.1 * quaternion_dist.mean() if self.include_ori else torch.zeros(1)

        current_gripper_openings = self.gripper_type.gripper_opening_from_fk(padded_trajectory[:, :, 15].unsqueeze(-1))
        gripper_dist = torch.nn.L1Loss()(current_gripper_openings, target_tensor[:, :, 15].view(current_gripper_openings.size()))
        gripper_loss = gripper_dist if self.include_gripper else torch.zeros(1)

        length_loss = torch.nn.L1Loss()(self.target_length, trajectory_length(padded_trajectory).float())

        print(f"Loss components: {position_loss.item():.5f}, {orientation_loss.item():.5f}, {gripper_loss.item():.5f}")
        return position_loss + orientation_loss + gripper_loss + 0.01 * length_loss

    @staticmethod
    def from_dict(dic):
        demonstration = torch.tensor(dic["demonstration"], dtype=torch.float32)
        gripper_type = RobotFactory().make_robot(dic["gripper_type"])
        return DemonstrationConstraintCartesian(gripper_type, demonstration, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "DemonstrationConstraintCartesian",
            "demonstration": self.demonstration.tolist(),
            "weight": self.weight,
            "gripper_type": self.gripper_type.__class__.__name__
        }
