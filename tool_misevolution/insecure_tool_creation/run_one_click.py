#!/usr/bin/env python3
"""One-click 6 GPU runner for self-evolution security experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"
DEFAULT_DATASET = os.getenv("EXPERIMENT_DATASET", "misevolution_user_queries")
DEFAULT_OUTPUT = Path(os.getenv("EXPERIMENT_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
GROUP_KEYS = ["ours", "direct_trigger", "random_sequence", "fixed_simple"]
PAIR_KEYS = [
    "ours_v4_vs_direct_trigger_matched_base_tasks",
    "ours_v1_vs_fixed_simple_matched_base_tasks",
    "ours_vs_random_sequence_matched_exact_task_names",
]


@dataclass
class RunningJob:
    label: str
    gpu: str
    proc: subprocess.Popen
    log_path: Path
    log_handle: object
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-evolution experiment suite on GPUs 2,3,4,5,6,7.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Experiment dataset name.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT), help="Experiment output root.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional cap per run; 0 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for direct and fixed baselines.")
    parser.add_argument("--random-seeds", default="11,22,33,44,55", help="Comma-separated seeds for random_sequence.")
    parser.add_argument("--groups", default="direct_trigger,random_sequence,fixed_simple", help="Baseline groups for final compare.")
    parser.add_argument("--full-output", action="store_true", help="Keep extra output artifacts.")
    parser.add_argument("--write-md", action="store_true", help="Also emit markdown comparison.")
    parser.add_argument("--clean", action="store_true", help="Clean run outputs before executing.")
    parser.add_argument("--skip-ours-full", action="store_true", help="Skip ours_full run.")
    parser.add_argument("--skip-ours-strong", action="store_true", help="Skip ours_strong run.")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip all baseline runs.")
    parser.add_argument("--skip-random-multiseed", action="store_true", help="Skip random_sequence multi-seed run.")
    parser.add_argument("--gpu-ours-full", default="2", help="GPU id for ours_full.")
    parser.add_argument("--gpu-ours-strong", default="3", help="GPU id for ours_strong.")
    parser.add_argument("--gpu-direct", default="4", help="GPU id for direct_trigger.")
    parser.add_argument("--gpu-fixed", default="5", help="GPU id for fixed_simple.")
    parser.add_argument("--gpu-random", default="6,7", help="Comma-separated GPU ids for random_sequence shards.")
    parser.add_argument(
        "--random-start-mode",
        choices=["wait_ours", "parallel"],
        default="wait_ours",
        help="Start random shards after ours_full finishes, or in parallel.",
    )
    return parser.parse_args()


def shell_join(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(item) for item in parse_csv_items(raw)]
    if not seeds:
        raise ValueError("No valid random seeds provided.")
    return seeds


def split_round_robin(items: list[int], parts: int) -> list[list[int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    buckets = [[] for _ in range(parts)]
    for index, item in enumerate(items):
        buckets[index % parts].append(item)
    return [bucket for bucket in buckets if bucket]


def variant_dir(output_root: Path, dataset: str, variant: str) -> Path:
    return output_root / dataset / "evolution" / variant


def clean_variant(output_root: Path, dataset: str, variant: str) -> None:
    root = variant_dir(output_root, dataset, variant)
    for relative in [
        "trajectories",
        "judge_results",
        "mcp_tools.jsonl",
        "mcp_tools_summary.json",
        "clean_mcp_tools.jsonl",
        "poison_mcp_tools.jsonl",
    ]:
        path = root / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def shard_output_root(output_root: Path, shard_index: int) -> Path:
    return output_root / "_random_shards" / f"shard_{shard_index}"


def shard_log_name(shard_index: int, gpu: str, seeds: list[int]) -> str:
    seed_suffix = "_".join(str(seed) for seed in seeds)
    return f"random_shard_{shard_index}_gpu{gpu}_seeds_{seed_suffix}.log"


def build_env(output_root: Path, dataset: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["EXPERIMENT_DATASET"] = dataset
    env["EXPERIMENT_OUTPUT_ROOT"] = str(output_root)
    if extra:
        env.update(extra)
    return env


def launch_job(label: str, cmd: list[str], env: dict[str, str], gpu: str, log_path: Path) -> RunningJob:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = dict(env)
    run_env["CUDA_VISIBLE_DEVICES"] = gpu
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(f"gpu={gpu}\n")
    log_handle.write(f"cwd={SCRIPT_DIR}\n")
    log_handle.write(f"$ {shell_join(cmd)}\n\n")
    log_handle.flush()
    print(f"[launch] {label} gpu={gpu} log={log_path}")
    proc = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        env=run_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(label=label, gpu=gpu, proc=proc, log_path=log_path, log_handle=log_handle, started_at=time.time())


def wait_for_jobs(jobs: list[RunningJob], stage_label: str) -> None:
    if not jobs:
        return
    pending = {job.proc.pid: job for job in jobs}
    labels = ", ".join(job.label for job in jobs)
    print(f"[stage] waiting for {stage_label}: {labels}")
    failures: list[str] = []
    while pending:
        for pid, job in list(pending.items()):
            result = job.proc.poll()
            if result is None:
                continue
            job.log_handle.close()
            elapsed = round(time.time() - job.started_at, 1)
            if result == 0:
                print(f"[done] {job.label} gpu={job.gpu} elapsed={elapsed}s")
            else:
                failures.append(f"{job.label} exited with code {result}; log={job.log_path}")
                print(f"[fail] {job.label} gpu={job.gpu} code={result} log={job.log_path}")
            pending.pop(pid)
        if pending:
            time.sleep(2)
    if failures:
        raise RuntimeError("; ".join(failures))


def mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(values) == 1:
        value = round(float(values[0]), 4)
        return {"mean": value, "std": 0.0, "min": value, "max": value}
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return {
        "mean": round(float(mean_value), 4),
        "std": round(float(variance ** 0.5), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }


def aggregate_seed_metrics(all_seed_metrics: list[dict]) -> dict:
    group_stats: dict[str, dict] = {}
    for group in GROUP_KEYS:
        low_rates = [
            seed_metrics["groups"][group]["low_score_rate"]
            for seed_metrics in all_seed_metrics
            if isinstance(seed_metrics["groups"][group]["low_score_rate"], (int, float))
        ]
        mean_scores = [
            seed_metrics["groups"][group]["mean_score"]
            for seed_metrics in all_seed_metrics
            if isinstance(seed_metrics["groups"][group]["mean_score"], (int, float))
        ]
        group_stats[group] = {
            "low_score_rate": mean_std([float(value) for value in low_rates]),
            "mean_score": mean_std([float(value) for value in mean_scores]),
        }

    matched_pair_stats: dict[str, dict] = {}
    for pair in PAIR_KEYS:
        deltas = [
            seed_metrics["matched_pairs"][pair]["delta_ours_minus_baseline"]
            for seed_metrics in all_seed_metrics
            if isinstance(seed_metrics["matched_pairs"][pair]["delta_ours_minus_baseline"], (int, float))
        ]
        ours_low = [
            seed_metrics["matched_pairs"][pair]["ours_low_score_rate"]
            for seed_metrics in all_seed_metrics
            if isinstance(seed_metrics["matched_pairs"][pair]["ours_low_score_rate"], (int, float))
        ]
        baseline_low = [
            seed_metrics["matched_pairs"][pair]["method_baseline_low_score_rate"]
            for seed_metrics in all_seed_metrics
            if isinstance(seed_metrics["matched_pairs"][pair]["method_baseline_low_score_rate"], (int, float))
        ]
        matched_pair_stats[pair] = {
            "delta_ours_minus_baseline": mean_std([float(value) for value in deltas]),
            "ours_low_score_rate": mean_std([float(value) for value in ours_low]),
            "method_baseline_low_score_rate": mean_std([float(value) for value in baseline_low]),
        }
    return {"group_stats": group_stats, "matched_pair_stats": matched_pair_stats}


def write_summary_csv(path: Path, all_seed_metrics: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "pair",
                "ours_n",
                "baseline_n",
                "ours_low_score_rate",
                "method_baseline_low_score_rate",
                "delta_ours_minus_baseline",
                "ours_mean_score",
                "method_baseline_mean_score",
            ]
        )
        for seed_metrics in all_seed_metrics:
            seed = seed_metrics["seed"]
            for pair, row in seed_metrics["matched_pairs"].items():
                writer.writerow(
                    [
                        seed,
                        pair,
                        row.get("ours_n"),
                        row.get("baseline_n"),
                        row.get("ours_low_score_rate"),
                        row.get("method_baseline_low_score_rate"),
                        row.get("delta_ours_minus_baseline"),
                        row.get("ours_mean_score"),
                        row.get("method_baseline_mean_score"),
                    ]
                )


def merge_random_shard_summaries(
    target_output_root: Path,
    dataset: str,
    shard_output_roots: list[Path],
    expected_seeds: list[int],
) -> None:
    if not shard_output_roots:
        return

    target_multi_seed_root = target_output_root / dataset / "evolution" / "baseline" / "multi_seed"
    target_multi_seed_root.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[Path, dict]] = []
    for root in shard_output_roots:
        summary_path = root / dataset / "evolution" / "baseline" / "multi_seed" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        summaries.append((root, json.loads(summary_path.read_text(encoding="utf-8"))))

    per_seed_map: dict[int, dict] = {}
    total_elapsed = 0.0
    multi_seed_strategy = None
    rerun_ours_each_seed = False
    ours_versions = None

    for root, summary in summaries:
        total_elapsed += float(summary.get("elapsed_seconds") or 0.0)
        multi_seed_strategy = multi_seed_strategy or summary.get("multi_seed_strategy")
        rerun_ours_each_seed = rerun_ours_each_seed or bool(summary.get("rerun_ours_each_seed"))
        ours_versions = ours_versions or summary.get("ours_versions")
        for seed_metrics in summary.get("per_seed", []):
            seed = seed_metrics.get("seed")
            if not isinstance(seed, int):
                continue
            per_seed_map[seed] = seed_metrics
            shard_seed_dir = root / dataset / "evolution" / "baseline" / "multi_seed" / f"seed_{seed}"
            target_seed_dir = target_multi_seed_root / f"seed_{seed}"
            if not shard_seed_dir.exists():
                raise FileNotFoundError(f"Missing shard seed directory for seed {seed}: {shard_seed_dir}")
            if shard_seed_dir.resolve() == target_seed_dir.resolve():
                continue
            if target_seed_dir.exists():
                shutil.rmtree(target_seed_dir)
            shutil.copytree(shard_seed_dir, target_seed_dir)

    merged_per_seed = [per_seed_map[seed] for seed in sorted(per_seed_map)]
    merged_summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": expected_seeds,
        "num_seeds": len(expected_seeds),
        "rerun_ours_each_seed": rerun_ours_each_seed,
        "multi_seed_strategy": multi_seed_strategy or "random_only_each_seed",
        "ours_versions": ours_versions,
        "elapsed_seconds": round(total_elapsed, 2),
        "per_seed": merged_per_seed,
        "aggregate": aggregate_seed_metrics(merged_per_seed),
        "shard_output_roots": [str(root) for root in shard_output_roots],
    }

    summary_json = target_multi_seed_root / "summary.json"
    summary_csv = target_multi_seed_root / "summary.csv"
    summary_json.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, merged_per_seed)
    print(f"[merge] random multi-seed summary -> {summary_json}")


def run_compare(output_root: Path, dataset: str, groups: str, full_output: bool, write_md: bool) -> None:
    env = build_env(output_root=output_root, dataset=dataset)
    cmd = [sys.executable, "method_baseline_experiments.py", "compare", "--groups", groups]
    if write_md or full_output:
        cmd.append("--write-md")
    print(f"[compare] {shell_join(cmd)}")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def build_ours_job(
    output_root: Path,
    dataset: str,
    variant: str,
    versions: str,
    gpu: str,
    max_tasks: int,
    log_root: Path,
) -> RunningJob:
    env = build_env(output_root=output_root, dataset=dataset, extra={"OURS_VARIANT": variant, "OURS_TASK_VERSIONS": versions})
    if max_tasks > 0:
        env["MAX_TASKS"] = str(max_tasks)
    command = [
        "/bin/bash",
        "-lc",
        f"{shlex.quote(sys.executable)} evaluation.py && {shlex.quote(sys.executable)} extract_mcp_tools.py && {shlex.quote(sys.executable)} judge_evaluation.py",
    ]
    label = f"{variant}_pipeline"
    return launch_job(label=label, cmd=command, env=env, gpu=gpu, log_path=log_root / f"{label}.log")


def build_random_jobs(
    output_root: Path,
    dataset: str,
    random_gpus: list[str],
    random_seed_shards: list[list[int]],
    max_tasks: int,
    clean: bool,
    full_output: bool,
    log_root: Path,
) -> tuple[list[RunningJob], list[Path]]:
    jobs: list[RunningJob] = []
    shard_roots: list[Path] = []
    for shard_index, seed_chunk in enumerate(random_seed_shards):
        gpu = random_gpus[shard_index]
        shard_root = output_root if shard_index == 0 else shard_output_root(output_root, shard_index)
        if clean and shard_index > 0 and shard_root.exists():
            shutil.rmtree(shard_root, ignore_errors=True)
        env = build_env(output_root=shard_root, dataset=dataset)
        cmd = [
            sys.executable,
            "run_multi_seed_experiments.py",
            "--seeds",
            ",".join(str(seed) for seed in seed_chunk),
            "--groups",
            "random_sequence",
            "--skip-rerun-ours",
            "--max-tasks",
            str(max_tasks),
        ]
        if clean:
            cmd.append("--clean-each")
        if full_output:
            cmd.append("--full-output")
        jobs.append(
            launch_job(
                label=f"random_sequence_shard_{shard_index}",
                cmd=cmd,
                env=env,
                gpu=gpu,
                log_path=log_root / shard_log_name(shard_index, gpu, seed_chunk),
            )
        )
        shard_roots.append(shard_root)
    return jobs, shard_roots


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    log_root = output_root / args.dataset / "evolution" / "logs" / time.strftime("%Y%m%d_%H%M%S")
    log_root.mkdir(parents=True, exist_ok=True)

    random_gpus = parse_csv_items(args.gpu_random)
    random_seeds = parse_seed_list(args.random_seeds)
    random_seed_shards = split_round_robin(random_seeds, max(1, len(random_gpus))) if random_gpus else []

    if args.clean:
        if not args.skip_ours_full:
            clean_variant(output_root, args.dataset, "ours")
        if not args.skip_ours_strong:
            clean_variant(output_root, args.dataset, "ours_strong")
        for shard_index in range(1, len(random_seed_shards)):
            extra_root = shard_output_root(output_root, shard_index)
            if extra_root.exists():
                shutil.rmtree(extra_root, ignore_errors=True)

    phase_one_jobs: list[RunningJob] = []
    if not args.skip_ours_full:
        phase_one_jobs.append(
            build_ours_job(
                output_root=output_root,
                dataset=args.dataset,
                variant="ours",
                versions="1,2,3,4",
                gpu=args.gpu_ours_full,
                max_tasks=args.max_tasks,
                log_root=log_root,
            )
        )
    if not args.skip_ours_strong:
        phase_one_jobs.append(
            build_ours_job(
                output_root=output_root,
                dataset=args.dataset,
                variant="ours_strong",
                versions="3,4",
                gpu=args.gpu_ours_strong,
                max_tasks=args.max_tasks,
                log_root=log_root,
            )
        )
    if not args.skip_baselines:
        baseline_env = build_env(output_root=output_root, dataset=args.dataset)
        for label, group, gpu in [
            ("direct_trigger", "direct_trigger", args.gpu_direct),
            ("fixed_simple", "fixed_simple", args.gpu_fixed),
        ]:
            cmd = [
                sys.executable,
                "method_baseline_experiments.py",
                "run",
                "--groups",
                group,
                "--seed",
                str(args.seed),
                "--max-tasks",
                str(args.max_tasks),
            ]
            if args.clean:
                cmd.append("--clean")
            if args.full_output:
                cmd.append("--full-output")
            phase_one_jobs.append(launch_job(label=label, cmd=cmd, env=baseline_env, gpu=gpu, log_path=log_root / f"{label}.log"))

    random_jobs: list[RunningJob] = []
    random_roots: list[Path] = []
    if not args.skip_baselines and not args.skip_random_multiseed:
        random_jobs, random_roots = build_random_jobs(
            output_root=output_root,
            dataset=args.dataset,
            random_gpus=random_gpus,
            random_seed_shards=random_seed_shards,
            max_tasks=args.max_tasks,
            clean=args.clean,
            full_output=args.full_output,
            log_root=log_root,
        )

    if args.random_start_mode == "parallel" and random_jobs:
        wait_for_jobs(phase_one_jobs + random_jobs, stage_label="phase_one_plus_random")
    else:
        wait_for_jobs(phase_one_jobs, stage_label="phase_one")
        wait_for_jobs(random_jobs, stage_label="random_shards")

    if random_roots:
        merge_random_shard_summaries(
            target_output_root=output_root,
            dataset=args.dataset,
            shard_output_roots=random_roots,
            expected_seeds=random_seeds,
        )

    run_compare(
        output_root=output_root,
        dataset=args.dataset,
        groups=args.groups,
        full_output=args.full_output,
        write_md=args.write_md,
    )

    comparison = output_root / args.dataset / "evolution" / "baseline" / "comparison_baseline_vs_ours.json"
    multiseed = output_root / args.dataset / "evolution" / "baseline" / "multi_seed" / "summary.json"
    print("\nDone")
    print(f"comparison={comparison}")
    print(f"multi_seed={multiseed}")
    print("ours_full=" + str(variant_dir(output_root, args.dataset, "ours")))
    print("ours_strong=" + str(variant_dir(output_root, args.dataset, "ours_strong")))
    print(f"logs={log_root}")


if __name__ == "__main__":
    main()
