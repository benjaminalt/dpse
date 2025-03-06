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
import socket

PYBULLET_SAMPLING_INTERVAL = 1/240
TRAJECTORY_SAMPLING_INTERVAL = 0.032
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "urdf")
OBJ_PATH = os.path.join(URDF_PATH, "objects")
OBJECTS = {os.path.splitext(filename)[0]: os.path.join(OBJ_PATH, filename) for filename in os.listdir(OBJ_PATH)}

dope_dirs = {
    "earth": "/home/bal/Projects/dope",
    "pc019l": "/home/lab019/alt/dope",
    "nb067": None,
    "pc008": None
}

hostname = socket.gethostname().lower()
DOPE_DIR = dope_dirs[hostname]
