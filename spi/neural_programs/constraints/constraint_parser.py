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

from spi.neural_programs.constraints.constraint import Constraint
from spi.neural_programs.constraints.constraint_cuboid import PositionConstraintCuboid, ForceConstraintCuboid
from spi.neural_programs.constraints.contact_constraint import ContactConstraint
from spi.neural_programs.constraints.demonstration_constraint import DemonstrationConstraintCartesian
from spi.neural_programs.constraints.execution_time_constraint import ExecutionTimeConstraint
from spi.neural_programs.constraints.moment_constraint import MomentConstraint
from spi.neural_programs.constraints.orientation_constraint import OrientationConstraint
from spi.neural_programs.constraints.path_length_constraint import PathLengthConstraintCartesian, \
    PathLengthConstraintConfiguration
from spi.neural_programs.constraints.position_constraint import PositionConstraintCartesian
from spi.neural_programs.constraints.reachability_constraint import ReachabilityConstraintCartesian
from spi.neural_programs.constraints.smoothness_constraint import SmoothnessConstraint
from spi.neural_programs.constraints.success_rate_constraint import SuccessRateConstraint
from spi.neural_programs.constraints.pose_constraint import PoseConstraint

string_to_type = {
    "PositionConstraintCuboid": PositionConstraintCuboid,
    "ForceConstraintCuboid": ForceConstraintCuboid,
    "ContactConstraint": ContactConstraint,
    "DemonstrationConstraintCartesian": DemonstrationConstraintCartesian,
    "ExecutionTimeConstraint": ExecutionTimeConstraint,
    "MomentConstraint": MomentConstraint,
    "OrientationConstraint": OrientationConstraint,
    "PathLengthConstraintCartesian": PathLengthConstraintCartesian,
    "PathLengthConstraintConfiguration": PathLengthConstraintConfiguration,
    "PoseConstraint": PoseConstraint,
    "PositionConstraintCartesian": PositionConstraintCartesian,
    "ReachabilityConstraintCartesian": ReachabilityConstraintCartesian,
    "SmoothnessConstraint": SmoothnessConstraint,
    "SuccessRateConstraint": SuccessRateConstraint
}


def constraint_from_dict(dic: dict) -> Constraint:
    return string_to_type[dic["type_id"]].from_dict(dic)
