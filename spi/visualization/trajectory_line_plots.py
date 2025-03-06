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

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from spi.common.trajectory import Trajectory

plt.rcParams.update({"legend.fontsize": 15,
                     "axes.titlesize": 15,
                     "axes.labelsize": 15,
                     "xtick.labelsize": 11,
                     "ytick.labelsize": 11})


def plot_1d_trajectory_xy(ax_x, ax_y, trajectories: List[torch.Tensor], colors: List[str],
                          labels: List[str], alphas: List[float] = None):
    alphas = alphas if alphas is not None else [1.0] * len(trajectories)
    for i in range(len(trajectories)):
        ax_x.plot(range(len(trajectories[i])), trajectories[i][:, 2].cpu(), color=colors[i], label=labels[i],
                  alpha=alphas[i])
        ax_y.plot(range(len(trajectories[i])), trajectories[i][:, 3].cpu(), color=colors[i], label=labels[i],
                  alpha=alphas[i])
    ax_x.set_xlabel("t")
    ax_x.set_ylabel("X")
    ax_y.set_xlabel("t")
    ax_y.set_ylabel("Y")


def plot_multiple_trajectories_vertical(trajectories: List[Trajectory], title=None, output_filename=None, show=False,
                                        include_forces=False, ms_per_sample=16, colors=None, labels=None):
    dim_names = [r"Pos. (X) [$m$]", r"Pos. (Y) [$m$]", r"Pos. (Z) [$m$]", "Ori. (qw)", "Ori. (qx)", "Ori. (qy)", "Ori. (qz)"]
    if include_forces:
        dim_names.extend([r"Force (X) [$N$]", r"Force (Y) [$N$]", r"Force (Z) [$N$]", r"Torque (X) [$Nm$]", r"Torque (Y) [$Nm$]", r"Torque (Z) [$Nm$]"])
    if colors is None:
        colors = ["black"] * len(trajectories)
    num_columns = 2 if include_forces else 1
    fig, axes = plt.subplots(7, num_columns, figsize=(4 * num_columns, 12))
    trajectories_as_list = [trajectory.to_list() for trajectory in trajectories]
    max_length = max([len(traj) for traj in trajectories])
    x = np.arange(0, max_length * ms_per_sample, ms_per_sample) / 1000
    for dim, dim_label in enumerate(dim_names):
        for idx, trajectory in enumerate(trajectories_as_list):
            # Pad all trajectories to the same length
            padded_trajectory = trajectory + [trajectory[-1] for _ in range(max_length - len(trajectory))]
            if include_forces:
                col_idx = int(dim / 7)
                row_idx = dim % 7
                axes[row_idx][col_idx].plot(x, [datapoint[dim] for datapoint in padded_trajectory], color=colors[idx])
                axes[row_idx][col_idx].set_ylabel(dim_label)
                if (col_idx == 0 and row_idx == 6) or (col_idx == 1 and row_idx == 5):
                    axes[row_idx][col_idx].set_xlabel(r"Time [$s$]")
            else:
                axes[dim].plot(x, [datapoint[dim] for datapoint in padded_trajectory], color=colors[idx])
                axes[dim].set_ylabel(dim_label)
                if dim == len(dim_names) - 1:
                    axes[dim].set_xlabel(r"Time [$s$]")
    if include_forces:
        axes[-1][-1].remove()
    if title is not None:
        fig.suptitle(title, y=1)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    if len(trajectories) > 1 and labels is not None:
        fig.legend(labels, loc="lower center", facecolor="white", framealpha=0.7, frameon=True)
        fig.subplots_adjust(bottom=0.14)
    if output_filename is not None:
        plt.savefig(output_filename)
    if show:
        plt.show()
    return fig, axes
