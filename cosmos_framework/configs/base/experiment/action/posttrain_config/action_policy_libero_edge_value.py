# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_libero_edge_value`` -- ``action_policy_libero_edge`` plus a Cosmos-Policy-style value target.

Identical to ``action_policy_libero_edge`` except that the LIBERO loader appends one value column to the
10-D action (``value_target=True``): the discounted return ``gamma ** steps_to_go`` of the frame each action
leads to (every demo succeeds at its last frame). The action head therefore predicts the value alongside the
actions, as Cosmos Policy (Kim et al. 2026) does; the recipe already predicts future frames (``wam`` mode).

- The value is the LAST column (index 10), so a client that keeps the first 10 dims (closed_loop_eval
  ``--action_dim 10``) still gets ``[pos(3), rot6d(6), gripper(1)]``.
- Normalization uses ``libero_native_frame_wise_relative_rot6d_value.json`` (the bundled stats + a value column
  with q01/q99 pinned to 0/1, so the normalized value is ``2V - 1``).
- The loss is masked to the real dims, which now include the value, so it gets 1/11 of the action loss.
- Serve with ``--action-stats-path libero_native_frame_wise_relative_rot6d_value.json --raw-action-dim 11``.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_edge import (
    action_policy_libero_edge,
)

cs = ConfigStore.instance()

VALUE_STATS = "libero_native_frame_wise_relative_rot6d_value.json"

action_policy_libero_edge_value = copy.deepcopy(action_policy_libero_edge)
_ds = action_policy_libero_edge_value["dataloader_train"]["dataloader"]["datasets"]["libero"]["dataset"]
_ds["value_target"] = True
_ds["value_gamma"] = 0.99
_ds["action_stats_path"] = VALUE_STATS

cs.store(group="experiment", package="_global_", name="action_policy_libero_edge_value", node=action_policy_libero_edge_value)
