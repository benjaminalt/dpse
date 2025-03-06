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
from spi.neural_programs.constraints.constraint import ConstraintType, Constraint


class PositionConstraintCartesian(Constraint):
    def __init__(self, constraint_type: ConstraintType, position: torch.Tensor, weight: float = 1.0):
        self.constraint_type = constraint_type
        self.position = position
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        target_tensor = self.position.reshape(1, 1, self.position.size(-1))\
            .repeat(trajectory.size(0), trajectory.size(1), 1).to(trajectory.device)
        if self.constraint_type == ConstraintType.TRAJECTORY:
            return torch.nn.L1Loss()(trajectory[:, :, 2:5], target_tensor)
        else:   # self.constraint_type == ConstraintType.GOAL
            return torch.nn.L1Loss()(trajectory[:, -1, 2:5].unsqueeze(1), target_tensor[:, -1, :].unsqueeze(1))

    @staticmethod
    def from_dict(dic):
        position = torch.tensor(dic["position"], dtype=torch.float32)
        return PositionConstraintCartesian(ConstraintType[dic["type"]], position, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "PositionConstraintCartesian",
            "type": self.constraint_type.name,
            "pose": self.position.tolist(),
            "weight": self.weight
        }


class PositionConstraintConfiguration(Constraint):
    def __init__(self, constraint_type: ConstraintType, joint_pos: torch.Tensor, weight: float = 1.0):
        self.constraint_type = constraint_type
        self.joint_pos = joint_pos
        self.weight = weight

    def loss(self, joint_trajectory: torch.Tensor) -> torch.Tensor:
        target_tensor = self.joint_pos.reshape(1, 1, self.joint_pos.size(-1))\
            .repeat(joint_trajectory.size(0), joint_trajectory.size(1), 1).to(joint_trajectory.device)
        if self.constraint_type == ConstraintType.TRAJECTORY:
            return torch.nn.L1Loss()(joint_trajectory, target_tensor)
        else:   # self.constraint_type == ConstraintType.GOAL
            return torch.nn.L1Loss()(joint_trajectory[:, -1].unsqueeze(1), target_tensor[:, -1].unsqueeze(1))

    @staticmethod
    def from_dict(dic):
        position = torch.tensor(dic["position"], dtype=torch.float32)
        return PositionConstraintCartesian(ConstraintType[dic["type"]], position, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "PoseConstraintConfiguration",
            "type": self.constraint_type.name,
            "pose": self.joint_pos.tolist(),
            "weight": self.weight
        }
