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

from typing import Tuple
import numpy as np

from am_control_plugin_python.data.template_data import ProgramData
from am_control_plugin_python.rps_interface.http_interface import RPSInterface
from spi.utils.rps_utils import assert_ok
from spi.utils.lar_utils import last_n_run_ids, revision_hashes_for_run_ids, trajectories_for_revision_hash, \
    template_instance_id_for_node_id

DB_CREDENTIALS = {"host": "nb067", "database": "lar_local", "user": "root", "password": "root"}


def _get_ml_and_mlrc_node_ids(rps) -> Tuple[int, int]:
    program_data = ProgramData.from_dict(assert_ok(rps.get_program_structure()))
    spike_search_template = list(filter(lambda t: t.type == "Spike Search Relative",
                                        program_data.topLevelTemplates))[0]
    spike_search_controller_ids = [dp.controllerNodeId for dp in spike_search_template.dynamicProperties]
    ml_node_id = spike_search_controller_ids[0]
    mlrc_node_id = spike_search_controller_ids[1]
    return ml_node_id, mlrc_node_id


def main():
    rps = RPSInterface()
    ml_node_id, mlrc_node_id = _get_ml_and_mlrc_node_ids(rps)
    last_run_ids = last_n_run_ids(DB_CREDENTIALS, 100)
    revision_hashes = revision_hashes_for_run_ids(DB_CREDENTIALS, last_run_ids)
    mlrcs = []
    for revision_hash in revision_hashes:
        # ml_tid = template_instance_id_for_node_id(DB_CREDENTIALS, revision_hash, ml_node_id)
        mlrc_tid = template_instance_id_for_node_id(DB_CREDENTIALS, revision_hash, mlrc_node_id)
        mlrc_trajectories = trajectories_for_revision_hash(DB_CREDENTIALS, revision_hash, [mlrc_tid])[0]
        mlrcs.extend(mlrc_trajectories)
    min_end_succ = np.inf
    min_end_fail = np.inf
    max_end_fail = -np.inf
    for traj in mlrcs:
        lowest_pose = min([pose.position.z for pose in traj.poses])
        if not traj.success_label:  # Contact failed --> in hole
            min_end_fail = min(min_end_fail, lowest_pose)
            max_end_fail = max(max_end_fail, lowest_pose)
        else:
            min_end_succ = min(min_end_succ, lowest_pose)
    print(f"min_end_succ: {min_end_succ}")
    print(f"min_end_fail: {min_end_fail}")
    print(f"max_end_fail: {max_end_fail}")


if __name__ == '__main__':
    main()