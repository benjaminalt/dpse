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
import random
from typing import List

import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from scipy import stats


class GaussianMixture(object):
    def __init__(self, means, covs, weights):
        self.means = means
        self.covs = covs
        self.weights = weights

    @staticmethod
    def make_random(modes_min, modes_max, x_min, x_max, y_min, y_max):
        num_gaussians = random.randint(modes_min, modes_max)
        weights = np.random.uniform(0, 1, num_gaussians)
        means = np.random.uniform([x_min, y_min], [x_max, y_max], (num_gaussians, 2))
        covs = []
        for _ in range(num_gaussians):
            P = stats.ortho_group.rvs(2)
            cov_magnitudes = np.random.uniform(1e-07, 1e-07, 2)
            diag = np.diag(cov_magnitudes)
            cov = np.matmul(np.matmul(np.transpose(P), diag), P)
            covs.append(cov)
        return GaussianMixture(means, np.array(covs), weights)

    def sample(self, num_samples) -> List[List[float]]:
        samples = []
        for i in range(num_samples):
            gaussian_idx = random.choices(range(len(self.weights)), self.weights)[0]
            mean = self.means[gaussian_idx]
            cov = self.covs[gaussian_idx]
            pos_x, pos_y = np.random.multivariate_normal(mean, cov)
            samples.append([pos_x, pos_y])
        return samples

    def plot(self, ax=None, x_min=None, x_max=None, y_min=None, y_max=None):
        samples = self.sample(500)
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            show = True
        ax.scatter([sample[0] for sample in samples], [sample[1] for sample in samples], color="gray", alpha=0.2)
        ax.scatter(self.means[:, 0], self.means[:, 1], color="purple", marker="^")
        if all(list(map(lambda x: x is not None, [x_min, x_max, y_min, y_max]))):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        ax.xaxis.set_minor_locator(MultipleLocator(0.001))
        ax.yaxis.set_minor_locator(MultipleLocator(0.001))
        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.grid(True, which="major", color="gray")
        ax.grid(True, which="minor", color="lightgray")
        if show:
            plt.show()

    @staticmethod
    def load(filepath: str):
        with open(filepath) as json_file:
            dic = json.load(json_file)
        means = np.array(dic["means"])
        covs = np.array(dic["covs"])
        weights = np.array(dic["weights"])
        return GaussianMixture(means, covs, weights)

    def save(self, filepath: str):
        with open(filepath, "w") as json_file:
            json.dump(self.to_dict(), json_file)

    def to_dict(self):
        return {
                "means": self.means.tolist(),
                "covs": self.covs.tolist(),
                "weights": self.weights.tolist()
            }

    @staticmethod
    def from_dict(dic: dict):
        return GaussianMixture(dic["means"], dic["covs"], dic["weights"])