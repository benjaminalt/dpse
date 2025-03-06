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

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from spi.visualization import common_utils
from spi.visualization import trajectory_plot
from spi.visualization.common_utils import get_movie_writer

default_axis_limits = {"x": [-2,2], "y": [-2,2], "z": [-2,2]}


def plot_trajectory_3d(trajectory, axis_limits=None, static_poses=None):
    """
    :param trajectory: A vector of affine transformations
    """
    global ax, quiver, line
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    base_quiver = ax.quiver([0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,1,0], [0,0,1], colors=common_utils.base_cs_colors, arrow_length_ratio=0.0)
    static_pose_quivers = []
    if static_poses is not None:
        for pose in static_poses:
            pose_cs = common_utils.coordinate_system_for_affine(pose)
            static_pose_quivers.append(ax.quiver(*pose_cs.transpose(), colors=common_utils.cs_colors, length=0.1, arrow_length_ratio=0.0))
    initial_cs = common_utils.coordinate_system_for_affine(trajectory[0])
    line = ax.plot([trajectory[0][0,3]], [trajectory[0][1,3]], [trajectory[0][2,3]])[0]
    quiver = ax.quiver(*initial_cs.transpose(), colors=common_utils.cs_colors, length=0.1, arrow_length_ratio=0.0)

    if axis_limits is None:
        axis_limits = default_axis_limits
    ax.set_xlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_ylim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_zlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_proj_type("persp")

    def update(num, trajectory):
        global quiver, line
        idx = num % len(trajectory)
        if idx == 0:
            line.remove()
            line = ax.plot([trajectory[idx][0, 3]], [trajectory[idx][1, 3]], [trajectory[idx][2, 3]], color="black")[0]
        line.set_data([[pose_affine[0, 3] for pose_affine in trajectory[:idx]],
                       [pose_affine[1, 3] for pose_affine in trajectory[:idx]]])
        line.set_3d_properties([pose_affine[2, 3] for pose_affine in trajectory[:idx]])
        # Redraw moving CS
        new_cs = common_utils.coordinate_system_for_affine(trajectory[idx])
        quiver.remove()
        quiver = ax.quiver(*(new_cs.transpose()), colors=common_utils.cs_colors, length=0.1, arrow_length_ratio=0.0)

    ani = FuncAnimation(fig, update, fargs=[trajectory], interval=100)
    plt.show()


def plot_trajectories_3d(traj_over_time: List[Trajectory],
                         fixed_trajectories: List[Trajectory],
                         fixed_poses: List[Pose], fixed_pose_colors: List[str] = None, fig=None, axes=None,
                         output_file=None, show=True):
    global ax, line
    if fig is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    else:
        ax = axes
    trajectory_plot.plot_trajectory_3d(fixed_trajectories, ["gray" for _ in range(len(fixed_trajectories))],
                                       ["reference" for _ in range(len(fixed_trajectories))], ax=ax)
    if fixed_pose_colors is None:
        fixed_pose_colors = ["blue" for _ in range(len(fixed_poses))]
    ax.scatter([pose.position.x for pose in fixed_poses],
               [pose.position.y for pose in fixed_poses],
               [pose.position.z for pose in fixed_poses], c=fixed_pose_colors)
    traj_over_time = [trajectory.to_tensor(meta_inf=False) for trajectory in traj_over_time]
    line = ax.plot(xs=[dp[0] for dp in traj_over_time[0]],
                   ys=[dp[1] for dp in traj_over_time[0]],
                   zs=[dp[2] for dp in traj_over_time[0]],
                   color="red")

    def update(num, trajectory_history):
        global ax, line
        idx = num % len(trajectory_history)
        line[0].set_data(np.array([dp[0].item() for dp in trajectory_history[idx]]), np.array([dp[1].item() for dp in trajectory_history[idx]]))
        line[0].set_3d_properties(np.array([dp[2].item() for dp in trajectory_history[idx]]))

        azimuth = num % 360
        ax.view_init(elev=10, azim=azimuth)
        return line


    ani = FuncAnimation(fig, update, fargs=[traj_over_time], interval=40, save_count=len(traj_over_time))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)
    if show:
        plt.show()


def plot_trajectories_2d(traj_over_time: List[Trajectory], fixed_trajectories: List[Trajectory],
                         fixed_hlines: List[float] = None, ylabels=None,  title=None, fig=None, axes=None,
                         output_file=None, show=True):
    global ax, lines
    if fig is None:
        fig, ax = plt.subplots(3, 2, sharex="col")
    else:
        ax = axes

    for fixed_traj in fixed_trajectories:
        traj_list = np.array(fixed_traj.to_list())
        for row in range(3):
            ax[row, 0].plot(range(len(traj_list)), traj_list[:, row], color="gray")       # Positions
            ax[row, 1].plot(range(len(traj_list)), traj_list[:, 7 + row], color="gray")   # Forces

    if fixed_hlines is not None:
        for row in range(3):
            ax[row, 0].axhline(fixed_hlines[row])
            ax[row, 1].axhline(fixed_hlines[3 + row])

    if ylabels is not None:
        for row in range(3):
            ax[row, 0].set_ylabel(ylabels[row])
            ax[row, 1].set_ylabel(ylabels[3 + row])

    if title is not None:
        fig.suptitle(title)

    traj_list = np.array(traj_over_time[0].to_list())
    lines = [ax[i, 0].plot(range(len(traj_list)), traj_list[:, i], color="red") for i in range(3)]
    lines.extend([ax[j, 1].plot(range(len(traj_list)), traj_list[:, 7 + j], color="red") for j in range(3)])

    def update(num, trajectory_history):
        global lines
        idx = num % len(trajectory_history)
        traj_as_list = np.array(trajectory_history[idx].to_list())
        for i in range(3):
            lines[i][0].set_data(range(len(traj_as_list)), traj_as_list[:, i])
            lines[3 + i][0].set_data(range(len(traj_as_list)), traj_as_list[:, 7 + i])
        for row in ax:
            for axs in row:
                axs.relim()
                axs.autoscale_view(True, True, True)

    ani = FuncAnimation(fig, update, fargs=[traj_over_time], interval=100, save_count=len(traj_over_time))
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)
    if show:
        plt.show()


def plot_trajectories_xy(traj_over_time: List[Trajectory], fixed_trajectories: List[Trajectory], points_over_time: List=None,
                         title=None, labels=None,
                         fig=None, axes=None, output_file=None, show=True):
    global ax, line, points, frame_number
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    for i, fixed_traj in enumerate(fixed_trajectories):
        traj_list = np.array(fixed_traj.to_list())
        if labels is not None:
            ax.plot(traj_list[:, 0], traj_list[:, 1], color="gray", label=labels[1 + i])
        else:
            ax.plot(traj_list[:, 0], traj_list[:, 1], color="gray")

    if title is not None:
        fig.suptitle(title)

    traj_list = np.array(traj_over_time[0].to_list())
    if labels is not None:
        line = ax.plot(traj_list[:, 0], traj_list[:, 1], color="red", label=labels[0])
    else:
        line = ax.plot(traj_list[:, 0], traj_list[:, 1], color="red")
    if points_over_time is not None:
        points = ax.scatter([point[0] for point in points_over_time[0]], [point[1] for point in points_over_time[0]], color="red", alpha=0.2)
    frame_number = ax.text(0.05, 0.05, "n = 0", fontsize=14, transform=ax.transAxes, horizontalalignment="left")

    def update(num, trajectory_history):
        global line, frame_number, points
        idx = num % len(trajectory_history)
        traj_as_list = np.array(trajectory_history[idx].to_list())
        line[0].set_data(traj_as_list[:, 0], traj_as_list[:, 1])
        if points_over_time is not None:
            points.set_offsets(np.array([point.numpy() for point in points_over_time[idx]]))
        frame_number.set_text(f"n = {num}")

    if labels is not None:
        ax.legend(loc="lower right")

    ani = FuncAnimation(fig, update, fargs=[traj_over_time], interval=100, save_count=len(traj_over_time))
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)
    if show:
        plt.show()


def plot_value_over_time(value_over_time: List[float], title=None, label=None, fig=None, axes=None, output_file=None, show=True):
    global ax, line, frame_number
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    if title is not None:
        fig.suptitle(title)

    line = ax.plot([0], [value_over_time[0]], color="blue", label=label)

    frame_number = ax.text(0.90, 0.80, "n = 0", fontsize=14, transform=ax.transAxes, horizontalalignment="left")

    def update(num, value_over_time):
        global line, frame_number
        line[0].set_data(range(num), value_over_time[:num])
        frame_number.set_text(f"n = {num}")

    if label is not None:
        ax.legend(loc="upper right")

    ani = FuncAnimation(fig, update, fargs=[value_over_time], interval=100, save_count=len(value_over_time))
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)


def plot_line_2d(line_over_time: List[torch.Tensor], fixed_lines: List[torch.Tensor], ylabel=None,  title=None, fig=None,
                 axes=None, output_file=None, show=True):
    global ax, line
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    for fixed_line in fixed_lines:
        ax.plot(range(len(fixed_line)), fixed_line, color="gray")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        fig.suptitle(title)

    line = ax.plot(range(len(line_over_time[0])), line_over_time[0], color="red")

    def update(num, line_history):
        global line
        idx = num % len(line_history)
        line[0].set_data(range(len(line_history[idx])), line_history[idx])

    ani = FuncAnimation(fig, update, fargs=[line_over_time], interval=40, save_count=len(line_over_time))
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)
    if show:
        plt.show()
