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

import matplotlib.animation
import numpy as np
import socket
import matplotlib.pyplot as plt

cs_colors = [[1,0,0],[0,1,0],[0,0,1], # Arrow bodies
             [1,0,0],[1,0,0], # Red arrowheads
             [0,1,0],[0,1,0], # Green arrowheads
             [0,0,1],[0,0,1]] # Blue arrowheads

base_cs_colors = [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5], # Arrow bodies
                    [0.5,0.5,0.5],[0.5,0.5,0.5], # Arrowheads
                    [0.5,0.5,0.5],[0.5,0.5,0.5], # Arrowheads
                    [0.5,0.5,0.5],[0.5,0.5,0.5]] # Arrowheads

fixed_cs_colors = [[0.2,0.2,0.2],[0.2,0.2,0.2],[0.2,0.2,0.2], # Arrow bodies
                    [0.2,0.2,0.2],[0.2,0.2,0.2], # Arrowheads
                    [0.2,0.2,0.2],[0.2,0.2,0.2], # Arrowheads
                    [0.2,0.2,0.2],[0.2,0.2,0.2]] # Arrowheads



def coordinate_system_for_affine(pose_affine):
    """
    Returns the position and direction vectors to plot a coordinate system
    See https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html?highlight=3d%20quiver%20plot#quiver
    :param pose_affine: np.array of shape (4,4)
    :return: A matrix of shape 3x6. XYZ components for the arrow positions; UVW components for the arrow directions
    """
    cs = np.empty(shape=(3, 6))
    cs[:, :3] = pose_affine[:3, 3]  # Position
    cs[:, 3:] = pose_affine[:3, :3].transpose()  # Orientation
    return cs


def get_movie_writer(filepath: str):
    if socket.gethostname() == "NB067":
        plt.rcParams['animation.ffmpeg_path'] = '/c/Program Files/ImageMagick-7.0.11-Q16-HDRI/ffmpeg'
    if filepath.endswith("mp4"):
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=23, extra_args=['-vcodec', 'libx264'])
    else:
        writer = matplotlib.animation.PillowWriter(fps=30)
    return writer