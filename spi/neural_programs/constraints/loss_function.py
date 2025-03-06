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

import torch.nn

from spi.neural_programs.constraints.constraint_parser import constraint_from_dict


class CustomLoss(torch.nn.Module):
    def __init__(self, cartesian_constraints: list, configuration_constraints: list = None):
        super(CustomLoss, self).__init__()
        self.cartesian_constraints = cartesian_constraints
        self.configuration_constraints = configuration_constraints if configuration_constraints is not None else []

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        batch_size = trajectory.size(0)
        output = torch.zeros(batch_size, device=trajectory.device)
        partial_losses = []

        for constraint in self.cartesian_constraints:
            partial_loss = constraint.loss(trajectory)
            output = torch.add(output, partial_loss, alpha=constraint.weight)
            partial_losses.append(partial_loss.item())

        if len(self.configuration_constraints) > 0:
            raise NotImplementedError()
 
        return output

    @staticmethod
    def from_dict(dic: dict):
        cartesian_constraints = [constraint_from_dict(constraint_dic) for constraint_dic in dic["cartesian_constraints"]]
        configuration_constraints = [constraint_from_dict(constraint_dic) for constraint_dic in dic["configuration_constraints"]]
        return CustomLoss(cartesian_constraints, configuration_constraints)

    def to_dict(self) -> dict:
        return {
            "cartesian_constraints": [c.to_dict() for c in self.cartesian_constraints],
            "configuration_constraints": [c.to_dict() for c in self.configuration_constraints]
    }
