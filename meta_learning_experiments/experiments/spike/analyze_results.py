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
import matplotlib.pyplot as plt
import pandas as pd
from spi.utils import matplotlib_defaults
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ALG_LABELS = {
    "spi": "tSPI",
    "spi_cdist": "tSPI ($\mathcal{L}_{cdist}$)",
    "spi_grid": "tSPI ($\mathcal{L}_{init}$)",
    "heuristic_baseline": "Heuristic",
    "random_baseline": "Human",
    "ga": "NSGA"
}

METRIC_LABELS = {
    "success": "Failure rate ($\mathcal{L}_{fail}$)",
    "cycle_time": "Cycle time ($\mathcal{L}_{cycle}$)",
    "path_length": "Path length ($\mathcal{L}_{path}$)"
}

TYPE_LABELS = {
    "sim": r"\begin{center}"
           r"Sim\\"
           r"{\footnotesize"
           r"10 test tasks\\"
           r"\vspace{-3pt}"
           r"$N$ = 128}"
           r"\end{center}",
    "real": r"\begin{center}"
           r"Real\\"
           r"{\footnotesize"
           r"6 test tasks\\"
           r"\vspace{-3pt}"
           r"$N$ = 128}"
           r"\end{center}"
}

ALG_COLORS = {
    "spi_cdist": "lightblue",
    "spi_grid": "blue",
    "ga": "red",
    "heuristic_baseline": "green",
    "random_baseline": "orange"
}

def main(args):
    results = {}
    for type in ["sim", "real"]:
        results_per_alg = {}
        for alg in ["random_baseline", "ga", "heuristic_baseline", "spi_cdist", "spi_grid"]:
            dir = f"metrics_{type}_{alg}"
            aggregated_results_for_tasks = []
            if dir not in os.listdir(args.input_dir):
                continue
            for task_dir in os.listdir(os.path.join(args.input_dir, dir)):
                aggregated_results_for_task = {}
                with open(os.path.join(args.input_dir, dir, task_dir, "results.json")) as results_file:
                    all_results_for_task = json.load(results_file)
                    for metric in all_results_for_task[0].keys():
                        aggregated_result = sum([res[metric] if not np.isnan(res[metric]) else 0.0 for res in all_results_for_task]) / len(all_results_for_task)
                        if metric not in results.keys():
                            results[metric] = {}
                        if alg not in results[metric].keys():
                            results[metric][alg] = {}
                        if type not in results[metric][alg].keys():
                            results[metric][alg][type] = []
                        results[metric][alg][type].append(aggregated_result)
                        aggregated_results_for_task[metric] = aggregated_result
                aggregated_results_for_tasks.append(aggregated_results_for_task)
            results_per_alg[alg] = aggregated_results_for_tasks
        # min_number_of_tasks = min([len(alg_results) for alg_results in results_per_alg.values()])
        # keys = ["success", "cycle_time", "path_length"]
        # for task_idx in range(min_number_of_tasks):
        #     print(f"[{type}|{task_idx}]---------------------------------------------------------------------")
        #     print(f"{'Algorithm':20}" + " | ".join(f"{key:>10}" for key in keys))
        #     for alg_name in results_per_alg.keys():
        #         print(f"{alg_name:20}" + " | ".join([f"{results_per_alg[alg_name][task_idx][key]:>10.4f}" for key in keys]))

    # results: metric -> alg -> type -> task
    for alg in results["success"].keys():
        if "real" in results['success'][alg].keys():
            print(f"{alg}: {results['success'][alg]['real']}")
        else:
            results['success'][alg]['real'] = [0.0] * 6

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[3.1, 2.625])
    print(fig.get_size_inches())
    algs = list(results["success"].keys())
    types = list(results["success"][algs[0]].keys())
    data_per_alg = {alg: [1 - (sum(results["success"][alg][type]) / len(results["success"][alg][type])) for type in types] for alg in algs}
    df = pd.DataFrame({ALG_LABELS[alg]: data_per_alg[alg] for alg in algs}, index=[TYPE_LABELS[type] for type in types])
    df.plot(kind="bar", ax=ax, rot=0, color=[ALG_COLORS[alg] for alg in algs])
    ax.set_ylabel(METRIC_LABELS["success"])
    ax.get_legend().remove()
    # leg = fig.legend(labels=[ALG_LABELS[alg] for alg in algs], bbox_to_anchor=(0., 1.02, 1., .102), mode="expand",
    #                  loc='upper center', ncol=len(algs), handlelength=0.8, labelspacing=0.1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, handlelength=0.8, labelspacing=0.1)
    plt.subplots_adjust(left=0.145, bottom=0.162, right=1.0, top=0.8)
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, f"spike_results.pdf"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing the metrics_X_Y directories")
    parser.add_argument("output_dir", type=str, help="Directory to store plots")
    main(parser.parse_args())
