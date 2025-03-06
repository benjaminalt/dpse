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

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def main(args):
    results = {}
    for process_type in ["drift", "brownian_mild", "shift"]:
        results[process_type] = {}
        process_type_dir = os.path.join(args.results_dir, process_type)
        for alg in ["fixed", "cdist", "none", "grid"]:
            results_dir_for_alg = os.path.join(process_type_dir, alg)
            successes = []
            for task in os.listdir(results_dir_for_alg):
                task_dir = os.path.join(results_dir_for_alg, task)
                success_over_time = np.load(os.path.join(task_dir, "success_label_over_time.npy"))
                successes.append(success_over_time.mean())
            mean_success = sum(successes) / len(successes)
            results[process_type][alg] = mean_success
    df = pd.DataFrame.from_dict(results)
    print(df)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("results_dir", type=str)
    main(parser.parse_args())
