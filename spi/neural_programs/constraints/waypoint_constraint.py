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
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.neural_templates.neural_template_utils import trajectory_length


class WaypointConstraintCartesian(Constraint):
    def __init__(self, pose: torch.Tensor, time_idx: int, weight: float = 1.0):
        self.pose = pose
        self.weight = weight
        self.time_idx = time_idx

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.time_idx < trajectory.size(1):
            batch_size = trajectory.size(0)
            target_pos = self.pose[:3].view(1, 3).repeat(batch_size, 1)
            actual_pos = trajectory[:, self.time_idx, 2:5]
            target_ori = self.pose[3:].view((1, 1, 4)).repeat(batch_size, 1, 1)
            actual_ori = trajectory[:, self.time_idx, 5:9].unsqueeze(1)
            return torch.nn.MSELoss()(target_pos, actual_pos) + QuaternionLoss()(target_ori, actual_ori)
        return self.time_idx - trajectory_length(trajectory)

    @staticmethod
    def from_dict(dic):
        pose = torch.tensor(dic["pose"], dtype=torch.float32)
        return WaypointConstraintCartesian(pose, dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "WaypointConstraintCartesian",
            "pose": self.pose.tolist(),
            "weight": self.weight
        }
