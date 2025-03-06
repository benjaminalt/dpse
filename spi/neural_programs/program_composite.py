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
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
from spi.neural_templates.models.seq2seq.model import ResidualGRU

from spi.neural_templates.program_primitive import ProgramPrimitive

from spi.neural_programs.program_component import ProgramComponent
from spi.neural_templates.neural_template_utils import normalize_traj, normalize, denormalize_traj


class ProgramComposite(ProgramComponent):
    """
    A neural program has several neural templates.
    Provides functionality to optimize the learnable parameters of the constituent templates.
    """

    def __init__(self, components: List[ProgramComponent], model_config: dict = None, model=None):
        """
        :param components: A list of ProgramComponents
        """
        self.components = components
        limits_x, limits_s, limits_Y = self._compute_limits()
        input_size = sum([c.input_size for c in self.components])
        device = torch.device("cpu") if model is None else None
        super().__init__(input_size, limits_x, limits_s, limits_Y, device=device)
        self.model = model
        self.model_config = model_config
        self.learnable_parameter_gradient_mask = []
        for component in self.components:
            if hasattr(component, "learnable_parameter_gradient_mask"):
                self.learnable_parameter_gradient_mask.append(component.learnable_parameter_gradient_mask)
            else:
                self.learnable_parameter_gradient_mask.append(torch.ones(component.input_size, dtype=torch.float32))
        self.learnable_parameter_gradient_mask = torch.cat(self.learnable_parameter_gradient_mask)

    @staticmethod
    def load(input_dir: str):
        components = []
        for component_dirname in os.listdir(os.path.join(input_dir, "components")):
            component_dir = os.path.join(input_dir, "components", component_dirname)
            if component_dirname.startswith("NeuralProgram"):
                components.append(ProgramComposite.load(component_dir))
            elif component_dirname.startswith("NeuralTemplate"):
                components.append(ProgramPrimitive.load(component_dir))
            else:
                raise NotImplementedError()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if "model_config.json" in os.listdir(input_dir):
            with open(os.path.join(input_dir, "model_config.json")) as model_config_file:
                model_config = json.load(model_config_file)
                model = ResidualGRU.load(os.path.join(input_dir, "model.pt"), device)
        else:
            model_config = None
            model = None
        return ProgramComposite(components, model_config, model)

    def save(self, output_dir: str, training_history=None):
        out_dir = os.path.join(output_dir, "{}_({})".format(type(self).__name__, time.strftime("%Y%m%d-%H%M%S")))
        os.makedirs(out_dir)
        if self.model_config is not None:
            with open(os.path.join(out_dir, "model_config.json"), "w") as json_file:
                json.dump(self.model_config, json_file)
            self.model.save(os.path.join(out_dir, "model.pt"))
        # Components
        components_dir = os.path.join(out_dir, "components")
        os.makedirs(components_dir)
        for c in self.components:
            c.save(components_dir)
        # Training history
        if training_history is not None:
            with open(os.path.join(out_dir, "training_history.json"), "w") as training_history_file:
                json.dump(training_history, training_history_file)

    def forward(self, x: torch.Tensor, s_in: torch.Tensor, Y_sim: torch.Tensor = None, denormalize_out=True,
                debug=False, skip_model=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Batch of program inputs
        :param s_in: Batch of start states
        :param Y_sim: (optional, unused) Batch of prior trajectories
        """
        Y = []
        s_current = s_in
        inputs_idx = 0
        for i, component in enumerate(self.components):
            component_inputs = x[:, inputs_idx:inputs_idx+component.input_size]
            Y_i, s_out_i = component(component_inputs, s_current, None,
                                     denormalize_out=True, skip_model=skip_model)
            Y.append(Y_i.to(self.device))
            s_current = s_out_i
            inputs_idx += component.input_size
        Y = torch.cat(Y, dim=1)

        self.limits_Y = self.limits_Y.to(self.device)

        if self.model is not None and not skip_model:
            self.model = self.model.to(self.device)
            Y = Y.to(self.device)
            x = x.to(self.device)
            self.limits_x = self.limits_x.to(self.device)

            Y_norm = normalize_traj(Y, self.limits_Y)
            x_norm = normalize(x, self.limits_x)
            Y_out_norm = self.model(x_norm, Y_norm)

            if debug:
                plt.plot(Y_norm[0, :, 2].detach().cpu(), Y_norm[0, :, 3].detach().cpu(), color="lightgreen",
                         label="sim")
                plt.plot(Y_out_norm[0, :, 2].detach().cpu(), Y_out_norm[0, :, 3].detach().cpu(), color="green", label="real")
                plt.legend()
                plt.show()
        else:
            Y_out_norm = normalize_traj(Y, self.limits_Y)

        Y_out = denormalize_traj(Y_out_norm, self.limits_Y) if denormalize_out else Y_out_norm
        s_out = Y_out[:, -1, 2:9]
        return Y_out, s_out

    def _compute_limits(self):
        limits_x = torch.cat([c.limits_x for c in self.components], dim=-1)
        limits_s_all = torch.stack([c.limits_s for c in self.components])
        limits_Y_all = torch.stack([c.limits_Y for c in self.components])
        limits_s = torch.stack((limits_s_all[:, 0, :].min(dim=0)[0], limits_s_all[:, 1, :].max(dim=0)[0]))
        limits_Y = torch.stack((limits_Y_all[:, 0, :].min(dim=0)[0], limits_Y_all[:, 1, :].max(dim=0)[0]))
        return limits_x, limits_s, limits_Y
