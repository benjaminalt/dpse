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
from matplotlib import rcParams

from natsort import natsorted
from spi.utils import matplotlib_defaults
from spi.utils.distributions import GaussianMixture

from meta_learning_experiments.experiments.common_utils import plot_hole_distribution_2d


def main(args):
    plot_every = 25

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for subdir in natsorted(os.listdir(args.input_dir)):
        task_dir = os.path.join(args.input_dir, subdir)
        with open(os.path.join(task_dir, "hole_distributions_over_time.json")) as hole_distributions_file:
            hole_distributions = json.load(hole_distributions_file)
        spike_points_over_time = np.load(os.path.join(task_dir, "spike_points_over_time.npy"))
        sim_over_time = np.load(os.path.join(task_dir, "sim_over_time.npy"))
        num_axes = int(len(hole_distributions) / plot_every)+1
        fig, axes = plt.subplots(ncols=num_axes, figsize=(1.5*num_axes, 1.5))
        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf
        for time_idx in range(len(hole_distributions)):
            if time_idx != len(hole_distributions) - 1 and time_idx % plot_every != 0:
                continue
            i = round(time_idx / plot_every)
            hole_distribution = GaussianMixture.from_dict(hole_distributions[time_idx])
            spike_points = torch.from_numpy(spike_points_over_time[time_idx])
            sim = torch.from_numpy(sim_over_time[time_idx])

            contour, _ = plot_hole_distribution_2d(axes[i], fig, hole_distribution, plot_in_mm=True, add_colorbar=False)
            if i == num_axes - 1: # Add colorbar to the rightmost plot
                cax = fig.add_axes([0.91, 0.25, 0.01, 0.75])
                fig.colorbar(contour, cax=cax)
                cax.set_ylabel("$f_{H_{t}}$")
                cax.yaxis.set_label_position("right")

            axes[i].scatter(spike_points[:, 0] * 1000, spike_points[:, 1] * 1000, marker="s", color="blue",
                            s=0.5*rcParams['lines.markersize'] ** 2)
            for j in range(len(spike_points) - 1):
                p = spike_points[j] * 1000
                q = spike_points[j+1] * 1000
                axes[i].plot([p[0], q[0]], [p[1], q[1]], color="blue")
            # axes[i].plot(sim[:, 2] * 1000, sim[:, 3] * 1000, color="blue")
            axes[i].text(0.1, 0.05, f"$t={time_idx}$", transform=axes[i].transAxes)

            # Remove y axis labels from all but the leftmost plot
            if i > 0:
                axes[i].set_yticklabels([])

            xmin = min(xmin, axes[i].get_xlim()[0])
            ymin = min(ymin, axes[i].get_ylim()[0])
            xmax = max(xmax, axes[i].get_xlim()[1])
            ymax = max(ymax, axes[i].get_ylim()[1])
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("Pos. (X) [mm]")

        axes[0].set_ylabel("Pos. (Y) [mm]")
        fig.subplots_adjust(left=0.09, bottom=0.255, top=1.0, wspace=0.045)
        # plt.show()
        fig.savefig(os.path.join(args.output_dir, f"{subdir}.pdf"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Dir containing subdirs for each task, i.e. output/spike/nonstationary/brownian/cdist")
    parser.add_argument("output_dir", type=str)
    main(parser.parse_args())
