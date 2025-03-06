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

import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt

from spi.common.pose import Pose


def load_mesh(object_file: str, origin: Pose, in_meters=False) -> trimesh.Trimesh:
    mesh = trimesh.load(object_file)
    if not in_meters:
        mesh.apply_scale(0.001)
    mesh.apply_transform(origin.to_affine())
    return mesh


def compute_distance_field(meshes: List[trimesh.Trimesh]) -> torch.Tensor:
    xs = torch.linspace(-1000, 1000, 20)
    ys = torch.linspace(-1000, 1000, 20)
    zs = torch.linspace(-1000, 1000, 20)
    coords = torch.stack(torch.meshgrid([xs, ys, zs]), dim=0).reshape(3, -1).transpose(0, 1)
    distance_field = evaluate_distance_field(meshes, coords)
    return torch.stack((coords, distance_field.unsqueeze(-1)), dim=-1)


def evaluate_distance_field(meshes: List[trimesh.Trimesh], coords: torch.Tensor) -> torch.Tensor:
    distance_field = (torch.ones(coords.size(0), device=coords.device) * np.inf).double()
    for mesh in meshes:
        closest_points, distances, triangle_id = mesh.nearest.on_surface(coords)
        distance_field = torch.min(distance_field, torch.from_numpy(distances).double().to(coords.device))
    return distance_field


def plot_models_and_distance_field_matplotlib(meshes: List[trimesh.Trimesh], origins: List[Pose],
                                              xrange: List[int], yrange: List[int], zrange: List[int]):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    num_samples = 50
    for i, row in enumerate(axes):
        if i == 0:
            x, y = torch.meshgrid([torch.linspace(xrange[0], xrange[1], num_samples),
                                   torch.linspace(yrange[0], yrange[1], num_samples)])
            coords = torch.stack((x, y), dim=0).reshape(2, -1).transpose(0, 1)
            z_increment = (zrange[1] - zrange[0]) / (len(row) - 1)
        elif i == 1:
            x, z = torch.meshgrid([torch.linspace(xrange[0], xrange[1], num_samples),
                                   torch.linspace(zrange[0], zrange[1], num_samples)])
            coords = torch.stack((x, z), dim=0).reshape(2, -1).transpose(0, 1)
            y_increment = (yrange[1] - yrange[0]) / (len(row) - 1)
        for j, ax in enumerate(row):
            if i == 0:
                z_value = zrange[0] + j * z_increment
                coords_3d = torch.cat((coords, (torch.ones(len(coords)) * z_value).unsqueeze(-1)), dim=-1)
                distance_field = evaluate_distance_field(meshes, coords_3d)
                ax.pcolormesh(x, y, distance_field.reshape(x.size()))
                for origin in origins:
                    ax.scatter(origin.position.x, origin.position.y, color="red")
                ax.set_ylabel("Y")
                ax.set_title(z_value)
            elif i == 1:
                y_value = yrange[0] + j * y_increment
                coords_3d = torch.cat((coords, (torch.ones(len(coords)) * y_value).unsqueeze(-1)), dim=-1)
                coords_3d = torch.index_select(coords_3d, 1, torch.LongTensor([0, 2, 1]))
                distance_field = evaluate_distance_field(meshes, coords_3d)
                ax.pcolormesh(x, z, distance_field.reshape(x.size()))
                for origin in origins:
                    ax.scatter(origin.position.x, origin.position.z, color="red")
                ax.set_ylabel("Z")
                ax.set_title(y_value)
            ax.set_xlabel("X")
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.05)
    plt.show()

