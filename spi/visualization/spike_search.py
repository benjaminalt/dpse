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

from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from spi.common.pose import Pose
from spi.common.trajectory import Trajectory
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from sklearn import mixture, preprocessing
from spi.utils.distributions import GaussianMixture
from spi.utils.pytorch_utils import unpad_padded_sequence
from spi.visualization.common_utils import get_movie_writer
from spi.visualization.trajectory_plot import plot_2d_trajectory_xy
from spi.visualization.trajectory_line_plots import plot_1d_trajectory_xy


def plot_spike_search_2d(ax, spike_points_world: torch.Tensor, hole_center_batch: torch.Tensor,
                         hole_center: torch.Tensor, real_world_pred: torch.Tensor, sim_world_pred: torch.Tensor,
                          real_world_label: torch.Tensor, sim_world_label: torch.Tensor):
    pred_real_unpadded = unpad_padded_sequence(real_world_pred)
    pred_sim_unpadded = unpad_padded_sequence(sim_world_pred)

    # Hole centers
    hole_radius = 0.0005
    holes_x = hole_center_batch[:, 0].cpu()
    holes_y = hole_center_batch[:, 1].cpu()
    circles = [plt.Circle((xi, yi), radius=hole_radius, linewidth=0, color=(0.9, 0.9, 0.9, 1), zorder=1) for xi, yi in zip(holes_x, holes_y)]
    for circle in circles:
        ax.add_artist(circle)
    ax.scatter(holes_x, holes_y, color="gray", marker="o", zorder=2)
    ax.scatter(hole_center[0].cpu(), hole_center[1].cpu(), color="black", marker="^", zorder=3)
    ax.scatter(spike_points_world[:, 0], spike_points_world[:, 1], color="red", alpha=0.5, zorder=4)
    ax.text(0.6, 0.95, f"Succ: Pred {pred_real_unpadded[-1, 1].cpu():.2f}, real {real_world_label[-1, 1].cpu():.2f}",
                  transform=ax.transAxes, bbox=dict(boxstyle="square", ec=(0, 0, 0), fc=(0.8, 0.8, 0.8), alpha=0.5))
    plot_2d_trajectory_xy(ax, [pred_sim_unpadded, sim_world_label], ["lightsalmon", "palegreen"],
                          ["Predicted (sim)", "Label (sim)"], alphas=[0.5, 1.0, 0.5, 1.0], zorder=5)
    ax.legend()


def plot_spike_search_2d_trajectory_distribution(ax, hole_center_batch: torch.Tensor, hole_center: torch.Tensor,
                                                  real_world_pred_batch: torch.Tensor, sim_world_pred: torch.Tensor,
                                                  real_world_label: torch.Tensor, sim_world_label: torch.Tensor):

    ax.scatter(hole_center_batch[:, 0].cpu(), hole_center_batch[:, 1].cpu(), color="gray")
    ax.scatter(hole_center[0].cpu(), hole_center[1].cpu(), color="black", marker="^")
    for real_world_pred in real_world_pred_batch:
        pred_real_unpadded = unpad_padded_sequence(real_world_pred)
        ax.scatter(pred_real_unpadded[-1, 2].cpu(), pred_real_unpadded[-1, 3].cpu(), color="red", alpha=0.2)
    pred_sim_unpadded = unpad_padded_sequence(sim_world_pred)
    plot_2d_trajectory_xy(ax, [pred_sim_unpadded, real_world_label, sim_world_label],
                          ["lightsalmon", "green", "palegreen"],
                          ["Predicted (sim)", "Label (real)", "Label (sim)"])
    ax.legend()


def plot_spike_search_1d_xy(ax_x, ax_y, real_world_pred: torch.Tensor, sim_world_pred: torch.Tensor,
                          real_world_label: torch.Tensor, sim_world_label: torch.Tensor):
    pred_real_unpadded = unpad_padded_sequence(real_world_pred)
    pred_sim_unpadded = unpad_padded_sequence(sim_world_pred)
    plot_1d_trajectory_xy(ax_x, ax_y, [pred_real_unpadded, pred_sim_unpadded, real_world_label, sim_world_label],
                          ["red", "lightsalmon", "green", "palegreen"],
                          ["Predicted (real)", "Predicted (sim)", "Label (real)", "Label (sim)"])


def plot_spike_search_1d_xy_distribution(ax_x, ax_y, real_world_pred_batch: torch.Tensor, sim_world_pred: torch.Tensor,
                                         real_world_label: torch.Tensor, sim_world_label: torch.Tensor):
    for real_world_pred in real_world_pred_batch:
        pred_real_unpadded = unpad_padded_sequence(real_world_pred)
        ax_x.scatter(range(len(pred_real_unpadded)), pred_real_unpadded[-1, 2].cpu(), color="red", alpha=0.2)
        ax_y.scatter(range(len(pred_real_unpadded)), pred_real_unpadded[-1, 3].cpu(), color="red", alpha=0.2)
    pred_sim_unpadded = unpad_padded_sequence(sim_world_pred)
    plot_1d_trajectory_xy(ax_x, ax_y, [pred_sim_unpadded, real_world_label, sim_world_label],
                                      ["lightsalmon", "green", "palegreen"],
                                      ["Predicted (sim)", "Label (real)", "Label (sim)"])


class SpikeSearchAnimation(object):
    def __init__(self, traj: Trajectory, hole_distribution: List[Pose], spike_pose_history: List[List[Pose]],
                 z_range: Tuple[float, float] = None):
        self.traj = traj
        self.hole_distribution = hole_distribution
        self.spike_pose_history = spike_pose_history
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ani = FuncAnimation(self.fig, self.update, interval=75, init_func=self.setup_plot, blit=False,
                                 save_count=len(spike_pose_history) + 45 + len(self.traj))
        self.z_range = z_range

    def setup_plot(self):
        self.ax.scatter([hole.position.x for hole in self.hole_distribution],
                        [hole.position.y for hole in self.hole_distribution],
                        [hole.position.z for hole in self.hole_distribution], color="gray")


        # Fit GMM
        gmm = mixture.GaussianMixture(6)
        scaler = preprocessing.MinMaxScaler()
        points_scaled = scaler.fit_transform(np.array([hole.parameters()[:2] for hole in self.hole_distribution]))
        gmm.fit(points_scaled)
        x = np.linspace(-0.75, 1.25)
        y = np.linspace(-0.75, 1.25)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T
        Z = np.exp(gmm.score_samples(XX))
        Z = Z.reshape(X.shape)

        mins_maxes = scaler.inverse_transform([[-0.75, -0.75], [1.25, 1.25]])
        x = np.linspace(mins_maxes[0, 0], mins_maxes[1, 0])
        y = np.linspace(mins_maxes[0, 1], mins_maxes[1, 1])
        X, Y = np.meshgrid(x, y)
        # CS = ax.plot_wireframe(X, Y, Z, alpha=0.5)
        self.ax.view_init(azim=0, elev=90)
        self.prob_dist = self.ax.contourf(X, Y, Z, 100, zdir="z", cmap=cm.get_cmap("jet"), alpha=0.25)

        self.spikes_scatter = self.ax.scatter([pose.position.x for pose in self.spike_pose_history[0]],
                                              [pose.position.y for pose in self.spike_pose_history[0]],
                                              [pose.position.z for pose in self.spike_pose_history[0]],
                                              color="red")
        self.traj_line, = self.ax.plot([], [], [], color="red")
        return (self.spikes_scatter, self.traj_line, *self.prob_dist.collections)

    def update(self, num):
        if num < len(self.spike_pose_history):
            # Animate spike point history
            xs = [pose.position.x for pose in self.spike_pose_history[num]]
            ys = [pose.position.y for pose in self.spike_pose_history[num]]
            zs = [pose.position.z for pose in self.spike_pose_history[num]]
            self.spikes_scatter._offsets3d = (xs, ys, zs)
        elif len(self.spike_pose_history) <= num < len(self.spike_pose_history) + 45:
            # Pan camera
            elev = 90 - (num - len(self.spike_pose_history))
            azim = num - len(self.spike_pose_history)
            self.ax.view_init(azim=azim, elev=elev)
            if num == len(self.spike_pose_history):
                for c in self.prob_dist.collections:
                    c.remove()
                self.ax.relim()
                self.ax.autoscale_view()
        else:
            idx = num - (len(self.spike_pose_history) + 45)
            xs = np.array([pose.position.x for pose in self.traj.poses[:idx]])
            ys = np.array([pose.position.y for pose in self.traj.poses[:idx]])
            zs = np.array([pose.position.z for pose in self.traj.poses[:idx]])
            self.traj_line.set_data(xs, ys)
            self.traj_line.set_3d_properties(zs)
            if len(zs) > 0:
                self.ax.set_zlim(min(zs), max(zs)) if self.z_range is None else self.ax.set_zlim(*self.z_range)
        return (self.spikes_scatter, self.traj_line, *self.prob_dist.collections)

    def show(self):
        plt.show()

    def save(self, filepath):
        writer = get_movie_writer(filepath)
        self.ani.save(filepath, writer=writer)


class NonstationarySpikeSearchAnimation(object):
    def __init__(self, traj: Trajectory, hole_distribution_history: List[List[Pose]], spike_pose_history: List[List[Pose]],
                 z_range: Tuple[float, float] = None):
        self.traj = traj
        self.hole_distribution_history = hole_distribution_history
        self.spike_pose_history = spike_pose_history
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        total_duration = 5000   # 5s
        interval = int(total_duration / len(hole_distribution_history))  # Adapt speed of animation to amount of data
        self.ani = FuncAnimation(self.fig, self.update, interval=interval, init_func=self.setup_plot, blit=False)
        self.z_range = z_range

    def setup_plot(self):
        self.ax.view_init(azim=0, elev=90)
        self.holes_scatter = self.ax.scatter([hole.position.x for hole in self.hole_distribution_history[0]],
                                             [hole.position.y for hole in self.hole_distribution_history[0]],
                                             [hole.position.z for hole in self.hole_distribution_history[0]],
                                             color="gray")
        self.spikes_scatter = self.ax.scatter([pose.position.x for pose in self.spike_pose_history[0]],
                                              [pose.position.y for pose in self.spike_pose_history[0]],
                                              [pose.position.z for pose in self.spike_pose_history[0]],
                                              color="red")
        max_x, max_y, min_x, min_y = -np.inf, -np.inf, np.inf, np.inf
        for i in range(len(self.hole_distribution_history)):
            holes = self.hole_distribution_history[i]
            spikes = self.spike_pose_history[i]
            for lst in (holes, spikes):
                for pose in lst:
                    max_x = pose.position.x if pose.position.x > max_x else max_x
                    max_y = pose.position.y if pose.position.y > max_y else max_y
                    min_x = pose.position.x if pose.position.x < min_x else min_x
                    min_y = pose.position.y if pose.position.y < min_y else min_y
        self.ax.set_xlim3d(min_x, max_x)
        self.ax.set_ylim3d(min_y, max_y)
        self.traj_line, = self.ax.plot([], [], [], color="red")
        return (self.holes_scatter, self.spikes_scatter, self.traj_line)

    def update(self, num):
        idx = num % len(self.spike_pose_history)
        # Animate spike point history
        spike_xs = [pose.position.x for pose in self.spike_pose_history[idx]]
        spike_ys = [pose.position.y for pose in self.spike_pose_history[idx]]
        spike_zs = [pose.position.z for pose in self.spike_pose_history[idx]]
        self.spikes_scatter._offsets3d = (spike_xs, spike_ys, spike_zs)

        # Animate hole pose history
        hole_xs = [pose.position.x for pose in self.hole_distribution_history[idx]]
        hole_ys = [pose.position.y for pose in self.hole_distribution_history[idx]]
        hole_zs = [pose.position.z for pose in self.hole_distribution_history[idx]]
        self.holes_scatter._offsets3d = (hole_xs, hole_ys, hole_zs)

        # elif len(self.spike_pose_history) <= num < len(self.spike_pose_history) + 45:
        #     # Pan camera
        #     elev = 90 - (num - len(self.spike_pose_history))
        #     azim = num - len(self.spike_pose_history)
        #     self.ax.view_init(azim=azim, elev=elev)
        # else:
        #     idx = num - (len(self.spike_pose_history) + 45)
        #     xs = np.array([pose.position.x for pose in self.traj.poses[:idx]])
        #     ys = np.array([pose.position.y for pose in self.traj.poses[:idx]])
        #     zs = np.array([pose.position.z for pose in self.traj.poses[:idx]])
        #     self.traj_line.set_data(xs, ys)
        #     self.traj_line.set_3d_properties(zs)
        #     if len(zs) > 0:
        #         self.ax.set_zlim(min(zs), max(zs)) if self.z_range is None else self.ax.set_zlim(*self.z_range)
        return (self.holes_scatter, self.spikes_scatter, self.traj_line)

    def show(self):
        plt.show()

    def save(self, filepath):
        writer = get_movie_writer(filepath)
        self.ani.save(filepath, writer=writer)


if __name__ == '__main__':
    gmm = GaussianMixture.make_random(2, 6, -0.002, 0.002, -0.002, 0.002)
    holes = gmm.sample(128)
    holes = [Pose.from_parameters(hole + [0.0, 1.0, 0.0, 0.0, 0.0]) for hole in holes]
    traj = Trajectory.from_tensor(-0.002 + 0.004 * torch.rand((32, 2+7+6)))
    spike_pose_history = []
    for i in range(25):
        spike_pose_history.append([])
        for j in range(16):
            spike_pose_history[i].append(Pose.from_parameters(-0.002 + 0.004 * torch.rand(7)))
    a = SpikeSearchAnimation(traj, holes, spike_pose_history)
    plt.show()
