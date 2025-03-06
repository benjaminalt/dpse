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

from graphmodel.graphmodel.graph_node import GraphNodeType, GraphNode
from graphmodel.graphmodel.graph_node_move_linear_relative import GraphNodeMoveLinearRelative
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from graphmodel.graphmodel.graph_node_move_linear import GraphNodeMoveLinear


class GraphNodeFactory(object):
    node_for_type = {
        GraphNodeType.MOVE_LINEAR: GraphNodeMoveLinear,
        GraphNodeType.MOVE_LINEAR_RELATIVE: GraphNodeMoveLinearRelative,
        GraphNodeType.MOVE_LINEAR_RELATIVE_CONTACT: GraphNodeMoveLinearRelativeContact,
    }

    @staticmethod
    def create(graph_node_type: GraphNodeType) -> GraphNode:
        return GraphNodeFactory.node_for_type[graph_node_type]()
