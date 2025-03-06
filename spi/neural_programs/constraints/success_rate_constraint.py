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

from spi.neural_programs.constraints.constraint import Constraint
from spi.neural_templates.program_primitive import ProgramPrimitive
from spi.neural_templates.neural_template_utils import success_probability


class SuccessRateConstraint(Constraint):
    def __init__(self, weight=1.0):
        self.weight = weight

    def loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function for success rates: Loss high when average probability of success low
        """
        return 1 - torch.mean(success_probability(trajectory)).clamp(0, 1)

    @staticmethod
    def from_dict(dic):
        return SuccessRateConstraint(dic["weight"])

    def to_dict(self) -> dict:
        return {
            "type_id": "SuccessRateConstraint",
            "weight": self.weight
        }
