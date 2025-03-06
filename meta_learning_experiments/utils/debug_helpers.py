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

import os

import cv2
import numpy as np
import pybullet as pb
from spi.common.pose import Pose


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "urdf")


def plot_object_coordinate_system(object_id, link_id=None):
    if link_id is not None:
        pb.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=3, parentObjectUniqueId=object_id,
                            parentLinkIndex=link_id)
        pb.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=3, parentObjectUniqueId=object_id,
                            parentLinkIndex=link_id)
        pb.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=3, parentObjectUniqueId=object_id,
                            parentLinkIndex=link_id)
    else:
        pb.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=3, parentObjectUniqueId=object_id)
        pb.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=3, parentObjectUniqueId=object_id)
        pb.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=3, parentObjectUniqueId=object_id)


def plot_pose(pose: Pose, label=None):
    object_id = pb.loadURDF(os.path.join(URDF_PATH, "objects", "tiny_box.urdf"), basePosition=pose.position.parameters(),
                            baseOrientation=pose.orientation.to_qxyzw(), useFixedBase=True)
    plot_object_coordinate_system(object_id)
    if label is not None:
        pb.addUserDebugText(label, pose.position.parameters())
    return object_id


def show_img(np_rgb):
    np_bgr = cv2.cvtColor(np_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imshow("test", np_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
