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
import shutil
from typing import List


def resample(l: List, orig_sampling_interval: float, new_sampling_interval: float):
    resampled = []
    t = 0.0
    i = 0
    while i < len(l):
        while t < (i + 1) * orig_sampling_interval:
            resampled.append(l[i])
            t += new_sampling_interval
        i += 1
    return resampled


def remove_dir_if_exists_and_create(dirpath: str):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)
