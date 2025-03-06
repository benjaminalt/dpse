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

from robots.robot import UR5, UR5Robotiq, RobotiqGripper, Robot


class RobotFactory(object):
    def __init__(self):
        self.robots = {robot_class.__name__: robot_class for robot_class in [UR5, UR5Robotiq, RobotiqGripper]}

    def make_robot(self, robot_string) -> Robot:
        return self.robots[robot_string]()
