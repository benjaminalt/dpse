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
import pylab
from spi.common.pose import Pose
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation

from spi.visualization import common_utils
from spi.visualization.common_utils import get_movie_writer

default_axis_limits = {"x": [-2,2], "y": [-2,2], "z": [-2,2]}


def visualize(poses, reference_pose):
    global ax, quiver, line
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    reference_cs = common_utils.coordinate_system_for_affine(reference_pose.to_affine())
    reference_quiver = ax.quiver(*reference_cs.transpose(), colors=common_utils.base_cs_colors, arrow_length_ratio=0.0)
    pose_css = [common_utils.coordinate_system_for_affine(pose.to_affine()) for pose in poses]
    quivers = [ax.quiver(*cs.transpose(), colors=common_utils.cs_colors, arrow_length_ratio=0.0, length=0.05) for cs in pose_css]

    position_vectors = []
    for pose in poses:
        start_pos = reference_cs[:,:3]
        direction = pose.to_affine()[:3,3] - start_pos
        position_vectors.append(np.hstack([start_pos, direction]))
    lines = [ax.quiver(*np.transpose(position_vectors), arrow_length_ratio=0.00, linewidths=0.05)]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_proj_type("persp")

    plt.show()


def animate(pose_series, pose_series_labels=None, reference_pose=None, fixed_poses=None, fixed_labels=["start", "end"], axis_limits=None):
    global ax, quivers, fig, moving_labels, static_labels
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Reference pose
    if reference_pose is not None:
        reference_cs = common_utils.coordinate_system_for_affine(reference_pose.to_affine())
        reference_quiver = ax.quiver(*reference_cs.transpose(), colors=common_utils.base_cs_colors, arrow_length_ratio=0.0)

    # Fixed poses
    static_labels = []
    if fixed_poses is not None:
        fixed_quivers = []
        for idx, fixed_pose in enumerate(fixed_poses):
            pose_cs = common_utils.coordinate_system_for_affine(fixed_pose.to_affine())
            fixed_quivers.append(ax.quiver(*pose_cs.transpose(), colors=common_utils.fixed_cs_colors, length=0.05, arrow_length_ratio=0.0))
            x, y, _ = proj3d.proj_transform(fixed_pose.position.x, fixed_pose.position.y, fixed_pose.position.z, ax.get_proj())
            label = pylab.annotate(
                str(fixed_labels[idx]),
                xy=(x, y), xytext=(-10, 10),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            static_labels.append(label)

    # Moving poses
    pose_css = [[common_utils.coordinate_system_for_affine(pose.to_affine()) for pose in poses] for poses in pose_series]
    quivers = [ax.quiver(*pose_css[i][0].transpose(), colors=common_utils.cs_colors, length=0.05, arrow_length_ratio=0.0) for i in range(len(pose_css))]

    # Labels
    if pose_series_labels is None:
        pose_series_labels = [str(i + 1) for i in range(len(pose_series))]
    moving_labels = []
    for idx, pose in enumerate([poses[0] for poses in pose_series]):
        x, y, _ = proj3d.proj_transform(pose.position.x, pose.position.y, pose.position.z, ax.get_proj())
        label = pylab.annotate(
            pose_series_labels[idx],
            xy=(x, y), xytext=(-10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        moving_labels.append(label)

    if axis_limits is None:
        axis_limits = default_axis_limits
    ax.set_xlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_ylim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_zlim(axis_limits["x"][0], axis_limits["x"][1])
    ax.set_proj_type("persp")

    def update(num):
        global quivers, fig, moving_labels, static_labels
        if fixed_poses is not None:
            for idx in range(len(static_labels)):
                pose = fixed_poses[idx]
                pose_cs = common_utils.coordinate_system_for_affine(pose.to_affine())
                x, y, _ = proj3d.proj_transform(pose_cs[0, 0], pose_cs[0, 1], pose_cs[0, 2], ax.get_proj())
                static_labels[idx].xy = x, y
                static_labels[idx].update_positions(fig.canvas.renderer)

        for series_idx in range(len(quivers)):
            step_idx = num % len(pose_css[0])
            new_cs = pose_css[series_idx][step_idx]
            quivers[series_idx].remove()
            quivers[series_idx] = ax.quiver(*new_cs.transpose(), colors=common_utils.cs_colors, length=0.05, arrow_length_ratio=0.0)

            x, y, _ = proj3d.proj_transform(new_cs[0,0], new_cs[0,1], new_cs[0,2], ax.get_proj())
            moving_labels[series_idx].xy = x, y
            moving_labels[series_idx].update_positions(fig.canvas.renderer)
        fig.canvas.draw()

    ani = FuncAnimation(fig, update, interval=5)
    plt.show()


def animate_2d_xy(pose_series: List[List[Pose]], fixed_poses: List[Pose], output_file=None, show=False):
    """
    :param pose_series: shape (n_timesteps, n_poses)
    :param fixed_poses:
    :return:
    """
    global ax, scatter
    fig, ax = plt.subplots()
    ax.scatter([fixed_pose.position.x for fixed_pose in fixed_poses],
               [fixed_pose.position.y for fixed_pose in fixed_poses], color="gray", alpha=0.5)
    scatter = ax.scatter([pose.position.x for pose in pose_series[0]], [pose.position.y for pose in pose_series[0]],
                         color="red")

    def update(num, pose_history):
        global scatter
        idx = num % len(pose_history)
        scatter.set_offsets(np.array([[pose.position.x, pose.position.y] for pose in pose_history[idx]]))

    ani = FuncAnimation(fig, update, fargs=[pose_series], interval=200, save_count=len(pose_series))
    if output_file is not None:
        writer = get_movie_writer(output_file)
        ani.save(output_file, writer=writer)
    if show:
        plt.show()
