#!/usr/bin/env bash
# 10-step smoke test (proposal milestone: "baseline trains").
#
# Checks to make by hand once it finishes:
#   1. critic/score/mean is nonzero by step ~5-10 (reward extraction works)
#   2. response_length/mean is in the thousands (thinking mode is on)
#   3. no OOM at max_response_length=8192
#   4. step time x 100 fits the compute window for TWO seeds
#      -> report step time to the group TODAY; it fixes the step budget.
set -euo pipefail
STEPS=10 EXP_NAME=smoke SEED=0 bash "$(dirname "$0")/run_seed.sh"
