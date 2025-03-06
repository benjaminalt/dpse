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

def is_4_by_4_matrix(t: torch.Tensor):
    return t.size(-1) == 4 and t.size(-2) == 4

def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    # Normalize quaternion
    # torch.conj() for computing the complex conjugate is only implemented for CPU (yet)
    orig_device = quat.device
    quat = quat.to(torch.device("cpu"))
    norm = torch.sqrt(torch.sum(quat ** 2, dim=-1, keepdim=True))
    q_n = quat / norm
    q_matrix = torch.stack((torch.stack((q_n[:, 0], -q_n[:, 1], -q_n[:, 2], -q_n[:, 3]), dim=-1),
                            torch.stack((q_n[:, 1],  q_n[:, 0], -q_n[:, 3],  q_n[:, 2]), dim=-1),
                            torch.stack((q_n[:, 2],  q_n[:, 3],  q_n[:, 0], -q_n[:, 1]), dim=-1),
                            torch.stack((q_n[:, 3], -q_n[:, 2],  q_n[:, 1],  q_n[:, 0]), dim=-1)), dim=1)
    q_bar_matrix = torch.stack((torch.stack((q_n[:, 0], -q_n[:, 1], -q_n[:, 2], -q_n[:, 3]), dim=-1),
                                torch.stack((q_n[:, 1],  q_n[:, 0],  q_n[:, 3], -q_n[:, 2]), dim=-1),
                                torch.stack((q_n[:, 2], -q_n[:, 3],  q_n[:, 0],  q_n[:, 1]), dim=-1),
                                torch.stack((q_n[:, 3],  q_n[:, 2], -q_n[:, 1],  q_n[:, 0]), dim=-1)), dim=1)
    product_matrix = q_matrix.matmul(q_bar_matrix.conj().transpose(-1, -2))
    return product_matrix[:, 1:, 1:].to(orig_device)


def rotation_matrix_to_quaternion(mat: torch.Tensor) -> torch.Tensor:
    """
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    :param mat: 3x3 rotation matrix
    :return: Unit quaternion
    """
    assert len(mat.size()) == 3, "Missing batch dimension"
    # torch.conj() for computing the complex conjugate is only implemented for CPU (yet)
    orig_device = mat.device
    m_batch = mat.to(torch.device("cpu")).conj().transpose(-1, -2)  # This method assumes row-vector and postmultiplication of that vector
    # Cannot process batch in parallel: Singularity checks require if
    quaternions = []
    for m in m_batch:
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = torch.stack([m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]])
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = torch.stack([m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]])
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = torch.stack([m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t])
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = torch.stack([t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]])
        q = q * 0.5 / torch.sqrt(t)
        quaternions.append(q)
    return torch.stack(quaternions).to(orig_device)


def euler_zyx_to_quaternion(euler_zyx: torch.Tensor, degrees=False) -> torch.Tensor:
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    rx, ry, rz are floats (unit: rad) representing an orientation in Euler angles with ZYX rotation order (Tait-Bryan)
    :return:
    """
    assert len(euler_zyx.size()) == 2, "Missing batch dimension"
    euler_rad = euler_zyx * math.pi / 180 if degrees else euler_zyx
    heading = euler_rad[:, 2]
    attitude = euler_rad[:, 1]
    bank = euler_rad[:, 0]
    ch = torch.cos(heading / 2)
    sh = torch.sin(heading / 2)
    ca = torch.cos(attitude / 2)
    sa = torch.sin(attitude / 2)
    cb = torch.cos(bank / 2)
    sb = torch.sin(bank / 2)
    w = cb * ca * ch + sb * sa * sh
    x = sb * ca * ch - cb * sa * sh
    y = cb * sa * ch + sb * ca * sh
    z = cb * ca * sh - sb * sa * ch
    return torch.stack((w, x, y, z), dim=1)


def quaternion_to_euler_zyx(q: torch.Tensor) -> torch.Tensor:
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion
    :param q: Quaternion
    """
    assert len(q.size()) == 2, "Missing batch dimension"
    bank = torch.atan2(2 * (q[:,0] * q[:,1] + q[:,2] * q[:,3]), 1 - 2 * (q[:,1] ** 2 + q[:,2] ** 2))
    attitude = torch.asin(2 * (q[:,0] * q[:,2] - q[:,3] * q[:,1]))
    heading = torch.atan2(2 * (q[:,0] * q[:,3] + q[:,1] * q[:,2]), 1 - 2 * (q[:,2] ** 2 + q[:,3] ** 2))
    return torch.stack((bank, attitude, heading), dim=1)


def pose_to_affine(pose: torch.Tensor) -> torch.Tensor:
    assert len(pose.size()) == 2, "Missing batch dimension"
    batch_size = pose.size(0)
    rot = quaternion_to_rotation_matrix(pose[:,3:])
    upper_block = torch.cat((rot, pose[:, :3].unsqueeze(2)), dim=2)
    lower_block = torch.cat((torch.zeros(batch_size, 1, 3, dtype=pose.dtype, device=pose.device),
                             torch.ones(batch_size, 1, 1, dtype=pose.dtype, device=pose.device)), dim=2)
    return torch.cat((upper_block, lower_block), dim=1)


def affine_to_pose(affine: torch.Tensor) -> torch.Tensor:
    assert len(affine.size()) == 3, "Missing batch dimension"
    quat = rotation_matrix_to_quaternion(affine[:, :3, :3])
    return torch.cat((affine[:, :3, -1], quat), dim=-1)


def pose_euler_zyx_to_affine(pose_euler_zyx: torch.Tensor, degrees=False) -> torch.Tensor:
    assert len(pose_euler_zyx.size()) == 2, "Missing batch dimension"
    quaternions = euler_zyx_to_quaternion(pose_euler_zyx[:,3:6], degrees)
    return pose_to_affine(torch.cat((pose_euler_zyx[:, :3], quaternions), dim=-1))


def affine_to_pose_euler_zyx(affine: torch.Tensor) -> torch.Tensor:
    assert len(affine.size()) == 3, "Missing batch dimension"
    poses = affine_to_pose(affine)
    euler_zyx = quaternion_to_euler_zyx(poses[:, 3:7])
    return torch.cat((poses[:, :3], euler_zyx), dim=-1)


def affine_transform(pose_left: torch.Tensor, pose_right: torch.Tensor):
    """
    :param pose_left: Transform in world coordinates, else pose
    :param pose_right: Transform in local coordinates, else pose
    """
    tf_left = pose_left if len(pose_left.size()) >= 3 and is_4_by_4_matrix(pose_left) else pose_to_affine(pose_left)
    tf_right = pose_right if len(pose_right.size()) >= 3 and is_4_by_4_matrix(pose_right) else pose_to_affine(pose_right)
    transformed = tf_left.matmul(tf_right)
    return affine_to_pose(transformed)


def absolute_to_relative(pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    assert len(pose.size()) > 1, "Missing batch dimension"
    pose_affine = pose_to_affine(pose)
    reference_affine = pose_to_affine(reference)
    relative_affine = reference_affine.inverse().matmul(pose_affine)
    return affine_to_pose(relative_affine)


def relative_to_absolute(relative_pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    assert len(relative_pose.size()) > 1, "Missing batch dimension"
    relative_pose_affine = pose_to_affine(relative_pose)
    reference_affine = pose_to_affine(reference)
    absolute_affine = torch.matmul(reference_affine, relative_pose_affine)
    return affine_to_pose(absolute_affine)


def rotation_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    ct = torch.cos(angle)
    st = torch.sin(angle)
    vt = torch.tensor(1) - ct
    m_vt_0 = vt * axis[0]
    m_vt_1 = vt * axis[1]
    m_vt_2 = vt * axis[2]
    m_st_0 = axis[0] * st
    m_st_1 = axis[1] * st
    m_st_2 = axis[2] * st
    m_vt_0_1 = m_vt_0 * axis[1]
    m_vt_0_2 = m_vt_0 * axis[2]
    m_vt_1_2 = m_vt_1 * axis[2]
    return torch.stack((torch.stack((ct + m_vt_0 * axis[0], -m_st_2 + m_vt_0_1, m_st_1 + m_vt_0_2)),
                        torch.stack((m_st_2 + m_vt_0_1, ct + m_vt_1 * axis[1], -m_st_0 + m_vt_1_2)),
                        torch.stack((-m_st_1 + m_vt_0_2, m_st_0 + m_vt_1_2, ct + m_vt_2 * axis[2]))))


def axis_angle_from_rotation(m: torch.Tensor, eps=0.000001) -> tuple:
    epsilon = eps  # Margin to allow for rounding errors
    epsilon2 = eps * 10  # Margin to distinguish between 0 and 180 degrees
    if abs(m[0, 1] - m[1, 0]) < epsilon and abs(m[0, 2] - m[2, 0]) < epsilon and abs(m[1, 2] - m[2, 1]) < epsilon:
        # singularity found
        # first check for identity matrix which must have +1 for all terms
        # in leading diagonal and zero in other terms
        if abs(m[0, 1] + m[1, 0]) < epsilon2 and abs(m[0, 2] + m[2, 0]) < epsilon2 \
                and abs(m[1, 2] + m[2, 1]) < epsilon2 and abs(m[0, 0] + m[1, 1] + m[2, 2] - 3) < epsilon2:
            # this singularity is identity matrix so angle = 0
            return torch.tensor([1, 0, 0], device=m.device), 0
        # otherwise this singularity is angle = 180
        angle = torch.tensor(math.pi)
        xx = (m[0, 0] + 1) / 2
        yy = (m[1, 1] + 1) / 2
        zz = (m[2, 2] + 1) / 2
        xy = (m[0, 1] + m[1, 0]) / 4
        xz = (m[0, 2] + m[2, 0]) / 4
        yz = (m[1, 2] + m[2, 1]) / 4
        if xx > yy and xx > zz:
            # m[0, 0] is the largest diagonal term
            if xx < epsilon:
                x = torch.tensor(0)
                y = torch.tensor(0.7071)
                z = torch.tensor(0.7071)
            else:
                x = torch.sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:
            # m[1, 1] is the largest diagonal term
            if yy < epsilon:
                x = torch.tensor(0.7071)
                y = torch.tensor(0.0)
                z = torch.tensor(0.7071)
            else:
                y = torch.sqrt(yy)
                x = xy / y
                z = yz / y
        else:
            # m[2, 2] is the largest diagonal term so base result on this
            if zz < epsilon:
                x = torch.tensor(0.7071)
                y = torch.tensor(0.7071)
                z = torch.tensor(0)
            else:
                z = torch.sqrt(zz)
                x = xz / z
                y = yz / z
        return torch.stack((x, y, z)), angle  # return 180 deg rotation
    # as we have reached here there are no singularities so we can handle normally
    s = torch.sqrt((m[2, 1] - m[1, 2]) * (m[2, 1] - m[1, 2])
                  + (m[0, 2] - m[2, 0]) * (m[0, 2] - m[2, 0])
                  + (m[1, 0] - m[0, 1]) * (m[1, 0] - m[0, 1]))  # used to normalise
    if abs(s) < 0.001:
        # prevent divide by zero, should not happen if matrix is orthogonal and should be caught by
        # singularity test above, but I've left it in just in case
        s = torch.tensor(1.0)

    # AcosBackward returns inf/-inf when input exactly equals 1/-1
    arg = torch.clamp((m[0, 0] + m[1, 1] + m[2, 2] - 1) / 2, -1.0 + epsilon, 1.0 - epsilon)
    angle = torch.acos(arg)
    x = (m[2, 1] - m[1, 2]) / s
    y = (m[0, 2] - m[2, 0]) / s
    z = (m[1, 0] - m[0, 1]) / s
    return torch.stack((x, y, z)).to(m.device), angle


def quaternion_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Return angle theta between quaternions q1 and q2
    https://math.stackexchange.com/a/90098
    """
    q1_normalized = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2_normalized = q2 / torch.norm(q2, dim=-1, keepdim=True)
    dot_products = torch.sum(q1_normalized * q2_normalized, -1)
    acos_arg = 2 * (dot_products ** 2) - torch.ones(dot_products.size(), device=dot_products.device)
    return torch.acos(torch.clamp(acos_arg, -1 + 1e-7, 1 - 1e-7))
