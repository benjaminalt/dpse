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
from argparse import ArgumentParser

import torch
from graphmodel.graphmodel.graph_node_move_linear_relative_contact import GraphNodeMoveLinearRelativeContact
from spi.neural_programs.train_program_component_sgd import train_sgd
from spi.neural_templates.program_primitive import ProgramPrimitive

from spi.neural_templates.models.seq2seq.model import ResidualGRU
from spi.simulation.static_simulator import StaticSimulator


def main(args):
    with open(args.model_config) as model_config_file:
        mlrc_model_config = json.load(model_config_file)
    with open(args.experiment_config) as experiment_config_file:
        experiment_config = json.load(experiment_config_file)

    simulator = StaticSimulator(GraphNodeMoveLinearRelativeContact(),
                                sampling_interval=experiment_config["sampling_interval"], multiproc=False)
    mlrc_input_size = 7+2+2
    output_size = 2+7+6
    model = ResidualGRU(mlrc_input_size, output_size, None, mlrc_model_config["hidden_size"], output_size,
                        mlrc_model_config["num_layers"], mlrc_model_config["dropout_p"])
    limits_x = torch.tensor(experiment_config["mlrc"]["limits_x"], dtype=torch.float32)
    limits_s = torch.tensor(experiment_config["mlrc"]["limits_s"], dtype=torch.float32)
    limits_Y = torch.tensor(experiment_config["mlrc"]["limits_Y"], dtype=torch.float32)
    mlrc = ProgramPrimitive("Move Linear Relative Contact", mlrc_input_size, limits_x, limits_s, limits_Y, simulator,
                            mlrc_model_config, model)
    train_sgd(mlrc, args.data_dir, args.output_dir, invert_success_label=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("model_config", type=str)
    parser.add_argument("experiment_config", type=str)
    main(parser.parse_args())
