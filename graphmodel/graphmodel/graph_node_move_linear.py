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

from graphmodel.graphmodel.graph_node import GraphNode, GraphNodeType
from graphmodel.graphmodel.graph_node_move_linear_relative import GraphNodeMoveLinearRelative
from utils.transformations import affine_transform, absolute_to_relative


class GraphNodeMoveLinear(GraphNode):
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: point_to|velocity|acceleration
        :param point_from:
        :return:
        """
        point_to_relative = absolute_to_relative(inputs[:7].unsqueeze(0), point_from[:7].unsqueeze(0)).squeeze()
        path = self.generate_path(torch.cat((point_to_relative, inputs[7:]), dim=-1))
        if len(point_from.size()) == 1:
            point_from = point_from.unsqueeze(0)
        path_absolute = affine_transform(point_from[:, :7], path)
        # path_absolute = torch.cat((point_from, path_absolute), dim=0)
        return path_absolute

    def generate_path(self, inputs: torch.Tensor) -> torch.Tensor:
        return GraphNodeMoveLinearRelative().generate_path(inputs)

    def type(self) -> GraphNodeType:
        return GraphNodeType.MOVE_LINEAR
