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
from scipy.stats import multivariate_normal
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_hole_distribution_surface(ax, hole_cnf: dict):
    def plot_2d_normal_distribution(mean, cov, axes):
        N = 60
        X = np.linspace(mean[0] - 0.003, mean[0] + 0.003, N)
        Y = np.linspace(mean[1] - 0.003, mean[1] + 0.003, N)
        X, Y = np.meshgrid(X, Y)

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        rv = multivariate_normal(mean, cov)
        Z = rv.pdf(pos)

        axes.contourf(X, Y, Z, 10, zdir='z', cmap=cm.viridis, offset=0, alpha=0.3)

    if hole_cnf["distribution"] == "normal" and not np.any(np.array(hole_cnf["cov"])):
        # Deterministic hole
        r = np.linspace(0, hole_cnf["radius"], 100)
        u = np.linspace(0, 2 * np.pi, 100)

        x = np.outer(r, np.cos(u)) + hole_cnf["mean"][0]
        y = np.outer(r, np.sin(u)) + hole_cnf["mean"][1]

        ax.plot_surface(x, y, np.ones((len(x), len(y))) * hole_cnf["z"], color="lightblue", alpha=0.5)

    elif hole_cnf["distribution"] == "normal":
        plot_2d_normal_distribution(hole_cnf["mean"], hole_cnf["cov"], ax)

    elif hole_cnf["distribution"] == "gaussian_mixture":
        for i in range(len(hole_cnf["mean"])):
            plot_2d_normal_distribution(hole_cnf["mean"][i], hole_cnf["cov"][i], ax)


def plot_flat_surface(ax, position_x, position_y, surface_width, z_height):
    xx, yy = np.meshgrid(np.linspace(position_x - surface_width / 2,
                                     position_x + surface_width / 2, 10),
                         np.linspace(position_y - surface_width / 2,
                                     position_y + surface_width / 2, 10))
    zz = np.ones((10, 10)) * z_height
    ax.plot_surface(xx, yy, zz, alpha=0.5)


if __name__ == '__main__':
    hole_cnf = {
        "distribution": "normal",
        "mean": [0.49172, -0.0528],
        "cov": [
            [2.5e-07, 2e-07],
            [2e-07, 2.5e-07]
        ],
        "z": 0.1314,
        "radius": 0.0005
    }
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_hole_distribution_surface(ax, hole_cnf)
    plt.show()
