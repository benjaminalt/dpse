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
from spi.neural_programs.constraints.orientation_constraint import OrientationConstraint
from spi.neural_programs.constraints.position_constraint import PositionConstraintCartesian


class PoseConstraint(Constraint):
    def __init__(self, constraint_type: ConstraintType, pose: torch.Tensor, weight: float = 1.0):
        self.constraint_type = constraint_type
        self.position_constraint = PositionConstraintCartesian(constraint_type, pose[:3])
        self.orientation_constraint = OrientationConstraint(constraint_type, pose[3:7])
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.position_constraint.weight * self.position_constraint.loss(trajectory) \
               + self.orientation_constraint.weight * self.orientation_constraint.loss(trajectory)

    @staticmethod
    def from_dict(dic):
        pose = torch.tensor(dic["pose"], dtype=torch.float32)
        return PoseConstraint(ConstraintType[dic["type"]], pose, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "PoseConstraint",
            "type": self.constraint_type.name,
            "pose": self.position_constraint.position.tolist() + self.orientation_constraint.orientation.tolist(),
            "weight": self.weight
        }
