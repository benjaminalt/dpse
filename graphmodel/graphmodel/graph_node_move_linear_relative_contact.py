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


class GraphNodeMoveLinearRelativeContact(GraphNode):
    def simulate(self, inputs: torch.Tensor, point_from: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: point_to|min_force|max_force|velocity|acceleration. point_to here encodes a relative motion
        :param point_from:
        :return:
        """
        relevant_inputs = torch.cat((inputs[:7], inputs[-2:]), dim=-1)
        return GraphNodeMoveLinearRelative().simulate(relevant_inputs, point_from)

    def generate_path(self, inputs: torch.Tensor) -> torch.Tensor:
        relevant_inputs = torch.cat((inputs[:7], inputs[-2:]), dim=-1)
        return GraphNodeMoveLinearRelative().generate_path(relevant_inputs)

    def type(self) -> GraphNodeType:
        return GraphNodeType.MOVE_LINEAR_RELATIVE_CONTACT
