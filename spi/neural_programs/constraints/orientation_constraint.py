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
from spi.neural_templates.models.trajectory_loss import QuaternionLoss


class OrientationConstraint(Constraint):
    def __init__(self, constraint_type: ConstraintType, quaternion: torch.Tensor, weight: float = 1.0):
        self.constraint_type = constraint_type
        self.orientation = quaternion
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        target_tensor = self.orientation.reshape(1, 1, self.orientation.size(-1))\
            .repeat(trajectory.size(0), trajectory.size(1), 1).to(trajectory.device)
        if self.constraint_type == ConstraintType.TRAJECTORY:
            return QuaternionLoss()(trajectory[:, :, 5:9], target_tensor)
        else:   # self.constraint_type == ConstraintType.GOAL
            return QuaternionLoss()(trajectory[:, -1, 5:9].unsqueeze(1), target_tensor[:, -1, :].unsqueeze(1))

    @staticmethod
    def from_dict(dic):
        quaternion = torch.tensor(dic["quaternion"], dtype=torch.float32)
        return OrientationConstraint(ConstraintType[dic["type"]], quaternion, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "OrientationConstraint",
            "type": self.constraint_type.name,
            "quaternion": self.orientation.tolist(),
            "weight": self.weight
        }
