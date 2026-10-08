#!/usr/bin/env bash
# LOCAL (ubr-g1): 2-GPU full Edge run of the LIBERO-10 action-policy recipe (track-B pipeline test).
TOML_FILE="examples/toml/sft_config/action_policy_libero_10_edge_2gpu.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
export LIBERO_ROOT="${LIBERO_ROOT:-}"
EXTRA_DATASET_CHECK='[[ -f "$LIBERO_ROOT/meta/info.json" ]] || { echo "ERROR: LIBERO_ROOT must contain meta/info.json (got: '\''$LIBERO_ROOT'\'')" >&2; exit 1; }'
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
