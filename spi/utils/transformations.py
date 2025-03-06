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

import math

import torch
from numpy import deg2rad
from scipy.spatial.transform import Rotation
import numpy as np
from pyquaternion.quaternion import Quaternion


def scale(x, a, b, c, d):
    """
    Transform x from the interval [a, b] to the interval [c,d]
    https://stats.stackexchange.com/a/178629
    """
    return (d - c) * ((x - a) / (b - a)) + c


def quaternion_from_euler_zyx(rx, ry, rz, degrees=False):
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    rx, ry, rz are floats (unit: rad) representing an orientation in Euler angles with ZYX rotation order (Tait-Bryan)
    :return:
    """
    z = deg2rad(rz) if degrees else rz
    y = deg2rad(ry) if degrees else ry
    x = deg2rad(rx) if degrees else rx
    rot = Rotation.from_euler("ZYX", [z, y, x])
    qx, qy, qz, qw = rot.as_quat()
    return Quaternion(qw, qx, qy, qz)


def euler_zyx_from_quaternion(q):
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    :param q: Quaternion
    :return: (rx, ry, rz) in Euler ZYX convention, in radians
    """
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
    z, y, x = rot.as_euler("ZYX")
    return np.array([x, y, z])


def rotation_matrix_from_quaternion(q):
    """
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    :param q: Quaternion
    :return:
    """
    qn = q.normalised
    return np.array(Rotation.from_quat([qn.x, qn.y, qn.z, qn.w]).as_matrix())


def quaternion_from_rotation_matrix(mat):
    """
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    :param q: 3x3 rotation matrix
    :return: Unit quaternion
    """
    q = Rotation.from_dcm(mat).as_quat()  # xyzw
    return Quaternion(w=q[3], x=q[0], y=q[1], z=q[2]).normalised  # wxyz


def affine_from_rotation_translation(rotation_matrix, translation=np.zeros((3,1))):
    """
    :param rotation_matrix: 3x3 rotation matrix
    :return: 4x4 affine transform with zero translation
    """
    if len(translation.shape) == 1:
        translation = np.expand_dims(translation, -1)
    top = np.hstack([rotation_matrix, translation])
    bottom = np.hstack([np.zeros((1,3)), np.ones((1,1))])
    return np.vstack([top,bottom])


def rotation_matrix_from_axis_angle(axis, angle):
    ax = axis / np.linalg.norm(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = ax
    return np.array([
        [c + (x ** 2) * (1 - c),    x * y * (1 - c) - z * s,    x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s,   c + (y ** 2) * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s,   z * y * (1 - c) + x * s,    c + (z ** 2) * (1 - c)]
    ])


def delta(traj_batch: torch.Tensor) -> torch.Tensor:
    metadata = traj_batch[:, :, :2]
    trajectories = traj_batch[:,:,2:]
    deltas = trajectories[:,1:] - trajectories[:,:-1]
    return torch.cat((metadata[:,:-1], deltas),  dim=-1)


def undelta(delta_batch: torch.Tensor, point_start: torch.Tensor) -> torch.Tensor:
    metadata = delta_batch[:,:,:2]
    deltas = delta_batch[:,:-1,2:]
    start_and_deltas = torch.cat((point_start.unsqueeze(1), deltas), dim=1)
    return torch.cat((metadata, start_and_deltas.cumsum(dim=1)), dim=-1)