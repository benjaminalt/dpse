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
import time
from typing import Tuple

import torch
import matplotlib.pyplot as plt

from spi.neural_programs.program_component import ProgramComponent
from spi.neural_templates.models.seq2seq.model import ResidualGRU
from spi.neural_templates.neural_template_utils import normalize, denormalize_traj, normalize_traj
from spi.simulation.differentiable_simulator import DifferentiableSimulator
from spi.simulation.static_simulator import StaticSimulator


class ProgramPrimitive(ProgramComponent):
    def __init__(self, template_type: str, input_size: int, limits_x: torch.Tensor, limits_s: torch.Tensor,
                 limits_Y: torch.Tensor, simulator: DifferentiableSimulator, model_config: dict,
                 model: torch.nn.Module = None, device=None):
        super().__init__(input_size, limits_x, limits_s, limits_Y, device=device)
        self.template_type = template_type
        self.model_config = model_config
        self.simulator = simulator
        self.model = model

    @staticmethod
    def load(input_dir: str, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        with open(os.path.join(input_dir, "model_config.json")) as config_file:
            model_config = json.load(config_file)
        with open(os.path.join(input_dir, "misc.json")) as config_file:
            misc = json.load(config_file)
            input_size = misc["input_size"]
            template_type = misc["template_type"]
        with open(os.path.join(input_dir, "limits.json")) as config_file:
            limits = json.load(config_file)
            limits_x = torch.tensor(limits["x"], dtype=torch.float32, device=device)
            limits_s = torch.tensor(limits["s"], dtype=torch.float32, device=device)
            limits_Y = torch.tensor(limits["Y"], dtype=torch.float32, device=device)
        with open(os.path.join(input_dir, "simulator_config.json")) as simulator_config_file:
            simulator_config = json.load(simulator_config_file)
        simulator = StaticSimulator.from_dict(simulator_config)

        model = ResidualGRU.load(os.path.join(input_dir, "model.pt"), device) if "model.pt" in os.listdir(input_dir) else None
        template = ProgramPrimitive(template_type, input_size, limits_x, limits_s, limits_Y,
                                    simulator, model_config, model, device)
        return template

    def save(self, output_dir: str, training_history=None):
        out_dir = os.path.join(output_dir, "{}_({})".format(type(self).__name__, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(out_dir)
        # Models
        if self.model is not None:
            self.model.save(os.path.join(out_dir, "model.pt"))
        # Model config
        with open(os.path.join(out_dir, "model_config.json"), "w") as json_file:
            json.dump(self.model_config, json_file)
        # Input and output limits
        with open(os.path.join(out_dir, "limits.json"), "w") as json_file:
            json.dump({
                "x": self.limits_x.cpu().tolist(),
                "s": self.limits_s.cpu().tolist(),
                "Y": self.limits_Y.cpu().tolist()}, json_file)
        # Misc settings
        with open(os.path.join(out_dir, "misc.json"), "w") as json_file:
            json.dump({"input_size": self.input_size, "template_type": self.template_type}, json_file)
        # Simulator
        with open(os.path.join(out_dir, "simulator_config.json"), "w") as json_file:
            json.dump(self.simulator.to_dict(), json_file)
        # Training history
        if training_history is not None:
            with open(os.path.join(out_dir, "training_history.json"), "w") as training_history_file:
                json.dump(training_history, training_history_file)
        print("Results saved under {}".format(out_dir))
        return out_dir

    def forward(self, x: torch.Tensor, s_in: torch.Tensor, Y_sim: torch.Tensor = None, denormalize_out=True,
                debug=False, skip_model=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if Y_sim is None:
            Y_sim = self.simulator.simulate(x, s_in, max_trajectory_len=self.model_config["seq_len"], cache=False)

        Y_sim = Y_sim.to(self.device)
        self.limits_Y = self.limits_Y.to(self.device)

        if self.model is not None and not skip_model:
            self.model = self.model.to(self.device)
            x = x.to(self.device)
            self.limits_x = self.limits_x.to(self.device)

            Y_sim_norm = normalize_traj(Y_sim, self.limits_Y)
            x_norm = normalize(x, self.limits_x)
            Y_norm = self.model(x_norm, Y_sim_norm)

            if debug:
                plt.plot(Y_sim_norm[0, :, 2].detach().cpu(), Y_sim_norm[0, :, 3].detach().cpu(), color="lightgreen",
                         label="sim")
                plt.plot(Y_norm[0, :, 2].detach().cpu(), Y_norm[0, :, 3].detach().cpu(), color="green", label="real")
                plt.legend()
                plt.show()
        else:
            Y_norm = normalize_traj(Y_sim, self.limits_Y)

        Y_out = denormalize_traj(Y_norm, self.limits_Y) if denormalize_out else Y_norm
        s_out = Y_out[:, -1, 2:9]
        return Y_out, s_out
