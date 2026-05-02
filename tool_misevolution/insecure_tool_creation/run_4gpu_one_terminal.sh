#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "legacy launcher name: run_4gpu_one_terminal.sh"
echo "using 6 GPU layout: 2,3,4,5,6,7"

exec ./run_all_methods_multigpu.sh "$@"
