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

import numpy as np

from spi.common.position import Position
from spi.common.orientation import Orientation
from spi.utils import transformations


class Pose(object):
    def __init__(self, position, orientation, reference=None):
        self.position = position
        self.orientation = orientation
        self.reference = reference

    @staticmethod
    def make_absolute_from_xyz_rxryrz(xyz_rxryrz, degrees=False):
        return Pose(Position(*xyz_rxryrz[:3]),
                    Orientation.from_euler_zyx(*xyz_rxryrz[3:6], degrees),
                    reference=None)

    @staticmethod
    def make_relative_from_xyz_rxryrz(xyz_rxryrz, reference):
        relative_position = Position(*xyz_rxryrz[:3]).to_relative(reference.position)
        relative_orientation = Orientation.from_euler_zyx(*xyz_rxryrz[3:6]).to_relative(reference.orientation)
        return Pose(relative_position, relative_orientation, reference)

    def to_absolute_xyz_rxryrz(self):
        if self.is_relative():
            absolute_position = self.position.to_absolute(self.reference.position).to_xyz()
            absolute_orientation = self.orientation.to_absolute(self.reference.orientation).to_euler_zyx()
        else:
            absolute_position = self.position.to_xyz()
            absolute_orientation = self.orientation.to_euler_zyx()
        return absolute_position + absolute_orientation

    def to_absolute_xyz_quaternion(self):
        if self.is_relative():
            absolute_position = self.position.to_absolute(self.reference.position).to_xyz()
            absolute_orientation = self.orientation.to_absolute(self.reference.orientation).parameters()
        else:
            absolute_position = self.position.to_xyz()
            absolute_orientation = self.orientation.parameters()
        return absolute_position + absolute_orientation

    @staticmethod
    def make_absolute_from_xyz_quaternion(xyz_quaternion):
        return Pose(Position(*xyz_quaternion[:3]),
                    Orientation.from_parameters(xyz_quaternion[3:7]),
                    reference=None)

    @staticmethod
    def make_relative_from_xyz_quaternion(xyz_quaternion, reference):
        return Pose(Position(*xyz_quaternion[:3]).to_relative(reference.position),
                    Orientation.from_parameters(xyz_quaternion[3:7]).to_relative(reference.orientation),
                    reference=reference)

    def is_relative(self):
        return self.reference is not None

    @staticmethod
    def from_parameters(parameters, reference=None):
        return Pose(Position.from_parameters(parameters[:3]),
                    Orientation.from_parameters(parameters[3:7]),
                    reference)

    def parameters(self):
        return self.position.parameters() + self.orientation.parameters()

    @staticmethod
    def parameter_names():
        return Position.parameter_names() + Orientation.parameter_names()
    
    def transform(self, affine_transformation, local=False):
        if local:
            result_affine = np.matmul(self.to_affine(), affine_transformation)
        else:
            result_affine = np.matmul(affine_transformation, self.to_affine())
        result = Pose.from_affine(result_affine, None)
        self.position = result.position
        self.orientation = result.orientation

    @staticmethod
    def from_affine(affine, reference=None):
        position = Position(*affine[:3,3])
        orientation = Orientation(transformations.quaternion_from_rotation_matrix(affine[:3,:3]))
        return Pose(position, orientation, reference)

    def to_affine(self):
        # Rotation matrix from quaternion
        rotation = transformations.rotation_matrix_from_quaternion(self.orientation.q)
        translation = np.array(self.position.parameters()).reshape((3,1))
        top = np.hstack([rotation, translation])
        bottom = np.expand_dims([0,0,0,1], 0)
        return np.vstack([top, bottom])

    def __str__(self):
        return str(self.parameters())

    def __repr__(self):
        return self.parameters()

    def __eq__(self, other):
        if type(other) != Pose:
            return False
        if other is self:
            return True
        return self.position == other.position and self.orientation == other.orientation

    @staticmethod
    def random(x_min, x_max, y_min, y_max, z_min, z_max):
        return Pose(Position.random(x_min, x_max, y_min, y_max, z_min, z_max),
                    Orientation.random())
