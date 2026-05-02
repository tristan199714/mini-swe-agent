"""Single entrypoint for method-baseline experiments (minimal output by default).

Usage:
  cd tool_misevolution/insecure_tool_creation
  python run_experiments.py --max-tasks 20 --seed 42 --clean

Optional:
  python run_experiments.py --full-output --write-md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline groups and build one comparison table.")
    p.add_argument(
        "--groups",
        default="direct_trigger,random_sequence,fixed_simple",
        help="Comma-separated baseline groups.",
    )
    p.add_argument("--max-tasks", type=int, default=0, help="Optional cap per group.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for random_sequence.")
    p.add_argument("--clean", action="store_true", help="Clean previous outputs for selected groups.")
    p.add_argument("--full-output", action="store_true", help="Keep extra output artifacts.")
    p.add_argument("--write-md", action="store_true", help="Also emit markdown comparison.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    baseline_script = script_dir / "method_baseline_experiments.py"

    cmd = [
        sys.executable,
        str(baseline_script),
        "run_and_compare",
        "--groups",
        args.groups,
        "--seed",
        str(args.seed),
        "--max-tasks",
        str(args.max_tasks),
    ]
    if args.clean:
        cmd.append("--clean")
    if args.full_output:
        cmd.append("--full-output")

    subprocess.run(cmd, check=True)

    if args.write_md and not args.full_output:
        subprocess.run(
            [
                sys.executable,
                str(baseline_script),
                "compare",
                "--groups",
                args.groups,
                "--write-md",
            ],
            check=True,
        )

    dataset = os.getenv("EXPERIMENT_DATASET", "misevolution_user_queries")
    output_root = os.getenv(
        "EXPERIMENT_OUTPUT_ROOT",
        "/home/nvidia/Sheldon/agent_defense/mini-swe-agent/results",
    )
    print(
        "Done. Comparison file: "
        f"{output_root}/{dataset}/evolution/baseline/comparison_baseline_vs_ours.json"
    )


if __name__ == "__main__":
    main()
