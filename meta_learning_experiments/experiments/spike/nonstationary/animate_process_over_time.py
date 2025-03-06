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

import json
import os
from argparse import ArgumentParser
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams, animation

rcParams['animation.ffmpeg_path'] = "C:\\Users\\alt\\Tools\\ffmpeg-4.4-full_build\\bin\\ffmpeg.exe"
from matplotlib.animation import FuncAnimation

from natsort import natsorted
from spi.utils import matplotlib_defaults
from spi.utils.distributions import GaussianMixture

from meta_learning_experiments.experiments.common_utils import plot_hole_distribution_2d


class Animator:
    def __init__(self, hole_distributions, spike_points_over_time, sim_over_time):
        self.hole_distributions = hole_distributions
        self.spike_points_over_time = spike_points_over_time
        self.sim_over_time = sim_over_time
        self.xmax = np.max(self.spike_points_over_time[:,:,0])
        self.xmin = np.min(self.spike_points_over_time[:,:,0])
        self.ymax = np.max(self.spike_points_over_time[:,:,1])
        self.ymin = np.min(self.spike_points_over_time[:,:,1])
        self.plot_every = 1
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Pos. (X) [mm]")
        self.ax.set_ylabel("Pos. (Y) [mm]")
        self.cax = self.fig.add_axes([0.87, 0.15, 0.01, 0.75])
        self.ani = None
        self.fig.subplots_adjust(left=0.136, bottom=0.157, right=0.829)

    def init_animation(self):
        pass

    def update_animation(self, time_idx):
        if time_idx != len(self.hole_distributions) - 1 and time_idx % self.plot_every != 0:
            return
        i = time_idx
        hole_distribution = GaussianMixture.from_dict(self.hole_distributions[time_idx])
        spike_points = torch.from_numpy(self.spike_points_over_time[time_idx])
        sim = torch.from_numpy(self.sim_over_time[time_idx])

        self.ax.clear()

        contour, _ = plot_hole_distribution_2d(self.ax, self.fig, hole_distribution, plot_in_mm=True, add_colorbar=False)
        self.fig.colorbar(contour, cax=self.cax)
        self.cax.set_ylabel("$f_{H_{t}}$")
        self.cax.yaxis.set_label_position("right")

        artists = [*contour.collections]

        scat = self.ax.scatter(spike_points[:, 0] * 1000, spike_points[:, 1] * 1000, marker="s", color="blue",
                               s=0.5 * rcParams['lines.markersize'] ** 2)
        artists.append(scat)
        for j in range(len(spike_points) - 1):
            p = spike_points[j] * 1000
            q = spike_points[j + 1] * 1000
            artists.append(self.ax.plot([p[0], q[0]], [p[1], q[1]], color="blue")[0])
        # axes[i].plot(sim[:, 2] * 1000, sim[:, 3] * 1000, color="blue")
        text = self.ax.text(0.1, 0.05, f"$t={time_idx}$", transform=self.ax.transAxes)
        artists.append(text)

        # self.ax.set_xlim(self.xmin * 1000 - 1, self.xmax * 1000 + 1)
        self.ax.set_xlim(420, 430)
        self.ax.set_ylim(self.ymin * 1000 - 1, self.ymax * 1000 + 1)
        self.ax.set_xlabel("Pos. (X) [mm]")
        self.ax.set_ylabel("Pos. (Y) [mm]")

        return artists

    def animate(self):
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=len(self.hole_distributions), blit=True)

    def save(self, path):
        # writer = animation.FFMpegWriter(fps=30, codec='h264')
        self.ani.save(path, dpi=300)


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for subdir in natsorted(os.listdir(args.input_dir)):
        task_dir = os.path.join(args.input_dir, subdir)
        with open(os.path.join(task_dir, "hole_distributions_over_time.json")) as hole_distributions_file:
            hole_distributions = json.load(hole_distributions_file)
        spike_points_over_time = np.load(os.path.join(task_dir, "spike_points_over_time.npy"))
        sim_over_time = np.load(os.path.join(task_dir, "sim_over_time.npy"))
        animator = Animator(hole_distributions, spike_points_over_time, sim_over_time)
        animator.animate()
        # plt.show()
        animator.save(os.path.join(args.output_dir, f"{subdir}.gif"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dir containing subdirs for each task, i.e. output/spike/nonstationary/brownian/cdist")
    parser.add_argument("output_dir", type=str)
    main(parser.parse_args())
