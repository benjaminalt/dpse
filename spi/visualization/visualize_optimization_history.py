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

import matplotlib.pyplot as plt
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from spi.visualization import poses, trajectory_line_plots
from spi.utils import matplotlib_defaults


def plot_optimization_history(param_history, reference_lines_y=None, title="Optimization History",
                              out_dir=None, xlabel=None, ylabels=None, optimal_idx=None):
    fig, axs = plt.subplots(len(param_history[0]), figsize=(8,12))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for index, ax in enumerate(axs):
        ax.grid(True)
        # Plot horizontal reference lines (if optima known)
        if reference_lines_y is not None and reference_lines_y[index] is not None:
            ax.axhline(reference_lines_y[index], dashes=[5, 2], color="green")
        # Plot vertical line for found optimal index
        if optimal_idx is not None:
            ax.axvline(optimal_idx, dashes=[2,2], color="green")
        # Plot actual data
        data = [params[index] for params in param_history]
        ax.plot(range(len(param_history)), data)
        if xlabel is not None and ylabels is not None:
            ax.set_ylabel(ylabels[index], fontsize=10)
    axs[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.subplots_adjust(left=0.13, bottom=0.05, right=0.99, top=0.94, hspace=0.7)
    if out_dir is None:
        plt.show()
    else:
        fig.savefig(os.path.join(out_dir, f"{title.lower().replace(' ', '_')}.png"))


def animate_optimization_history(optimized_param_history, fixed_poses):
    """
    TODO: This assumes that the program consists only of templates which have a point_to as first parameter
    """
    pose_series = []
    for neural_template_idx, param_history in enumerate(optimized_param_history):
        optimized_point_tos = [Pose.from_parameters(params[:7]) for params in param_history]
        pose_series.append(optimized_point_tos)
    poses.animate(pose_series, fixed_poses=fixed_poses)


def plot_intermediate_trajectories(intermediate_trajectories, output_dir=None, include_forces=False):
    for idx, trajectory in enumerate(intermediate_trajectories[::int(len(intermediate_trajectories) / 5)]):
        output_filename = None if output_dir is None else os.path.join(output_dir, "intermediate_trajectory_{}.png".format(idx + 1))
        trajectory_line_plots.plot_multiple_trajectories_horizontal([Trajectory.from_list(trajectory)],
                                                         "Test {}".format(idx + 1), output_filename, include_forces)
