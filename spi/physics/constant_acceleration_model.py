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

class ConstantAccelerationModel(object):
    def __init__(self, time_interval_seconds):
        self.time_interval_seconds = time_interval_seconds

    def simulate(self, a: float, x_0: float, v_0: float = 0):
        t = 0
        while True:
            v_current = v_0 + a * t
            x = x_0 + v_0 * t + 0.5 * a * t ** 2
            yield x, v_current
            t += self.time_interval_seconds
