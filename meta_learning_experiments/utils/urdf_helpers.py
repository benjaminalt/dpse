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

from argparse import ArgumentParser


def inertia_cuboid(m, h, w, d):
    """
    https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    i_h = (1/12) * m * (w ** 2 + d ** 2)
    i_w = (1/12) * m * (d ** 2 + h ** 2)
    i_d = (1/12) * m * (w ** 2 + h ** 2)
    return i_w, i_h, i_d


def inertia_cylinder(m, h, r):
    """
    https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    """
    i_x = (1/12) * m * (3 * r ** 2 + h ** 2)
    i_z = (1/2) * m * r ** 2
    return i_x, i_x, i_z


def main(args):
    if args.command == "inertia_cuboid":
        i_xx, i_yy, i_zz = inertia_cuboid(*map(float, args.params))
        print(f"I_xx: {i_xx:6f}, I_yy: {i_yy:6f}, I_zz: {i_zz:6f}")
    elif args.command == "inertia_cylinder":
        i_xx, i_yy, i_zz = inertia_cylinder(*map(float, args.params))
        print(f"I_xx: {i_xx:6f}, I_yy: {i_yy:6f}, I_zz: {i_zz:6f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["inertia_cuboid", "inertia_cylinder"])
    parser.add_argument("params", nargs="+")
    main(parser.parse_args())
