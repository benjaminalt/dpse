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
import torch
from matplotlib.axes import Axes

import numpy as np

from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from spi.utils.pytorch_utils import unpad_padded_sequence


def line_plot(ax: Axes, x: list, y: list, title: str, xaxis_label: str, yaxis_label: str):
    ax.plot(x, y)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_title(title)


def plot_trajectory(trajectory: Trajectory, title: str):
    n_rows = 4
    n_cols = 3  # Position, orientation, force
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    trajectory_arr = np.array(trajectory.to_list())
    trajectory_len = trajectory_arr.shape[0]
    line_plot(axes[0, 0], range(trajectory_len), trajectory_arr[:, 0], "Position (x)", "Time", "Position (x) [m]")
    line_plot(axes[1, 0], range(trajectory_len), trajectory_arr[:, 1], "Position (y)", "Time", "Position (y) [m]")
    line_plot(axes[2, 0], range(trajectory_len), trajectory_arr[:, 2], "Position (z)", "Time", "Position (z) [m]")
    line_plot(axes[0, 1], range(trajectory_len), trajectory_arr[:, 3], "Orientation (qw)", "Time", "Orientation (qw)")
    line_plot(axes[1, 1], range(trajectory_len), trajectory_arr[:, 4], "Orientation (qx)", "Time", "Orientation (qx)")
    line_plot(axes[2, 1], range(trajectory_len), trajectory_arr[:, 5], "Orientation (qy)", "Time", "Orientation (qy)")
    line_plot(axes[3, 1], range(trajectory_len), trajectory_arr[:, 6], "Orientation (qz)", "Time", "Orientation (qz)")
    line_plot(axes[0, 2], range(trajectory_len), trajectory_arr[:, 7], "Force (x)", "Time", "Force (x) [N]")
    line_plot(axes[1, 2], range(trajectory_len), trajectory_arr[:, 8], "Force (y)", "Time", "Force (y) [N]")
    line_plot(axes[2, 2], range(trajectory_len), trajectory_arr[:, 9], "Force (z)", "Time", "Force (z) [N]")
    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.97, wspace=0.35, hspace=0.42)
    plt.show()


def plot_trajectory_3d(trajectories: List[Trajectory], colors: List[str], labels: List[str], alphas: List[float] = None,
                       ax=None, points: List[Pose] = None):
    from mpl_toolkits.mplot3d import Axes3D
    show = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        show = True
    if alphas is None:
        alphas = [1.0 for _ in range(len(trajectories))]
    for i in range(len(trajectories)):
        trajectory_arr = np.array(trajectories[i].to_list())[:, :3]
        ax.plot(trajectory_arr[:, 0], trajectory_arr[:, 1], trajectory_arr[:, 2], color=colors[i], label=labels[i],
                alpha=alphas[i])
        if i == 0:
            # Create cubic bounding box to simulate equal aspect ratio
            X = trajectory_arr[:, 0]
            Y = trajectory_arr[:, 1]
            Z = trajectory_arr[:, 2]
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
            # Comment or uncomment following both lines to test the fake bounding box:
            # for xb, yb, zb in zip(Xb, Yb, Zb):
            #     ax.plot([xb], [yb], [zb], 'w')
    if points is not None:
        ax.scatter([point.position.x for point in points], [point.position.y for point in points], [point.position.z for point in points])
    ax.legend(fancybox=True, framealpha=0.5)
    if show:
        plt.show()


def plot_2d_trajectory_xy(ax, trajectories: List[torch.Tensor], colors: List[str], labels: List[str],
                          alphas: List[float] = None, zorder=None):
    alphas = alphas if alphas is not None else [1.0] * len(trajectories)
    for i in range(len(trajectories)):
        ax.plot(trajectories[i][:, 2].cpu(), trajectories[i][:, 3].cpu(), color=colors[i], label=labels[i],
                alpha=alphas[i], zorder=zorder)


def plot_2d_trajectory_distribution_xy(ax, real_world_pred_batch: torch.Tensor, sim_world_pred: torch.Tensor,
                                       hole_center_batch, hole_center, real_world_label, sim_world_label):
    # 2D spirals (multiple)
    ax.scatter(hole_center_batch[:, 0].cpu(), hole_center_batch[:, 1].cpu(), color="gray")
    ax.scatter(hole_center[0].cpu(), hole_center[1].cpu(), color="black", marker="^")
    for real_world_pred in real_world_pred_batch:
        pred_real_unpadded = unpad_padded_sequence(real_world_pred)
        ax.scatter(pred_real_unpadded[-1, 2].cpu(), pred_real_unpadded[-1, 3].cpu(), color="red", alpha=0.2)#1/len(real_world_pred_batch))
    pred_sim_unpadded = unpad_padded_sequence(sim_world_pred)
    ax.plot(pred_sim_unpadded[:, 2].cpu(), pred_sim_unpadded[:, 3].cpu(), color="lightsalmon", label="Pred (sim)")
    ax.plot(sim_world_label[:, 2].cpu(), sim_world_label[:, 3].cpu(), color="palegreen", label="Label (sim)")
    ax.plot(real_world_label[:, 2].cpu(), real_world_label[:, 3].cpu(), color="green", label="Label (real)")
    ax.legend()


def _plot_trajectory_deltas(axes, trajectory_deltas: np.array, color="red"):
    delta_x = trajectory_deltas[:, 2]
    delta_y = trajectory_deltas[:, 3]
    x = np.cumsum(delta_x)
    y = np.cumsum(delta_y)

    cuts = np.where(trajectory_deltas[:, 0] > 0.5)[0]
    start = 0

    for cut_value in cuts:
        axes[0, 0].plot(delta_x[start:cut_value], delta_y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[1, 0].plot(range(start, cut_value), delta_x[start:cut_value], 'k-', linewidth=3, color=color)
        axes[2, 0].plot(range(start, cut_value), delta_y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[0, 1].plot(x[start:cut_value], y[start:cut_value], 'k-', linewidth=3, color=color)
        axes[1, 1].plot(range(start, cut_value), x[start:cut_value], 'k-', linewidth=3, color=color)
        axes[2, 1].plot(range(start, cut_value), y[start:cut_value], 'k-', linewidth=3, color=color)
        start = cut_value + 1


def plot_trajectory_deltas(trajectory_deltas: np.array, reference_trajectory_deltas=None, save_name=None):
    f, axes = plt.subplots(3, 2)
    _plot_trajectory_deltas(axes, trajectory_deltas)
    if reference_trajectory_deltas is not None:
        _plot_trajectory_deltas(axes, reference_trajectory_deltas, color="green")

    axes[0, 0].axis('equal')
    axes[0, 1].axis('equal')

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    plt.close()
