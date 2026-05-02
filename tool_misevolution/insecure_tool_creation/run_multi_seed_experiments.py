"""Run method-baseline comparison across multiple seeds and aggregate results.

Usage:
  cd tool_misevolution/insecure_tool_creation
  python run_multi_seed_experiments.py --seeds 11,22,33,44,55 --clean-each

Outputs:
  - Per-seed snapshot:
    results/<dataset>/evolution/baseline/multi_seed/seed_<seed>/comparison_baseline_vs_ours.json
  - Aggregate summary:
    results/<dataset>/evolution/baseline/multi_seed/summary.json
    results/<dataset>/evolution/baseline/multi_seed/summary.csv
  - Full execution log:
    logs/multi_seed_<timestamp>.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, TextIO

from configs import MODEL, get_api_config

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_NAME = os.getenv("EXPERIMENT_DATASET", "misevolution_user_queries")
OUTPUT_ROOT = Path(os.getenv("EXPERIMENT_OUTPUT_ROOT", str(DEFAULT_RESULTS_ROOT)))
EVOLUTION_ROOT = OUTPUT_ROOT / DATASET_NAME / "evolution"
OURS_VARIANT = os.getenv("OURS_VARIANT", "ours")
OURS_ROOT = EVOLUTION_ROOT / OURS_VARIANT
BASELINE_ROOT = EVOLUTION_ROOT / "baseline"
COMPARISON_JSON = BASELINE_ROOT / "comparison_baseline_vs_ours.json"
MULTI_SEED_ROOT = BASELINE_ROOT / "multi_seed"
LOG_ROOT = Path(os.getenv("EXPERIMENT_LOG_ROOT", str(PROJECT_ROOT / "logs")))

_LOG_FH: TextIO | None = None


def _log(msg: str = "", console: bool = False) -> None:
    if console:
        print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(f"{msg}\n")
        _LOG_FH.flush()


def _enable_file_logging(log_dir: Path) -> Path:
    global _LOG_FH
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"multi_seed_{time.strftime('%Y%m%d_%H%M%S')}.log"
    _LOG_FH = log_path.open("a", encoding="utf-8", buffering=1)
    return log_path


def _disable_file_logging() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def parse_seeds(raw: str) -> List[int]:
    seeds = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        seeds.append(int(s))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def _parse_groups(raw: str) -> List[str]:
    groups = [g.strip() for g in raw.split(",") if g.strip()]
    if not groups:
        raise ValueError("No valid groups provided.")
    return groups


def _mean_std(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    if len(values) == 1:
        return {
            "mean": round(values[0], 4),
            "std": 0.0,
            "min": round(values[0], 4),
            "max": round(values[0], 4),
        }
    return {
        "mean": round(float(statistics.mean(values)), 4),
        "std": round(float(statistics.stdev(values)), 4),
        "min": round(float(min(values)), 4),
        "max": round(float(max(values)), 4),
    }


def _merge_with_reference(reference: dict, current: dict, update_random_only: bool) -> dict:
    if not reference:
        return current

    ref_rows = {r.get("group"): r for r in reference.get("rows", []) if isinstance(r, dict)}
    cur_rows = {r.get("group"): r for r in current.get("rows", []) if isinstance(r, dict)}
    ref_pairs = {r.get("pair"): r for r in reference.get("matched_rows", []) if isinstance(r, dict)}
    cur_pairs = {r.get("pair"): r for r in current.get("matched_rows", []) if isinstance(r, dict)}

    if not update_random_only:
        merged_rows = {**ref_rows, **cur_rows}
        merged_pairs = {**ref_pairs, **cur_pairs}
    else:
        merged_rows = dict(ref_rows)
        if "random_sequence" in cur_rows:
            merged_rows["random_sequence"] = cur_rows["random_sequence"]
        if "ours" in cur_rows:
            merged_rows["ours"] = cur_rows["ours"]

        merged_pairs = dict(ref_pairs)
        random_pair = "ours_vs_random_sequence_matched_exact_task_names"
        if random_pair in cur_pairs:
            merged_pairs[random_pair] = cur_pairs[random_pair]

    return {
        "generated_at": current.get("generated_at", reference.get("generated_at")),
        "rows": list(merged_rows.values()),
        "matched_rows": list(merged_pairs.values()),
    }


def _build_env(seed: int, max_tasks: int, ours_task_versions: str = "", ours_variant: str = OURS_VARIANT) -> dict:
    env = dict(**__import__("os").environ)
    env["LLM_SEED"] = str(seed)
    if max_tasks and max_tasks > 0:
        env["MAX_TASKS"] = str(max_tasks)
    else:
        env.pop("MAX_TASKS", None)

    if ours_task_versions.strip():
        env["OURS_TASK_VERSIONS"] = ours_task_versions.strip()
    else:
        env.pop("OURS_TASK_VERSIONS", None)
    if ours_variant.strip():
        env["OURS_VARIANT"] = ours_variant.strip()
    else:
        env.pop("OURS_VARIANT", None)
    return env


def _clean_ours_outputs(clean_mcp_pool: bool = False) -> None:
    for path in [
        OURS_ROOT / "trajectories",
        OURS_ROOT / "judge_results",
    ]:
        if path.exists():
            shutil.rmtree(path)
    if clean_mcp_pool:
        for path in [
            OURS_ROOT / "mcp_tools.jsonl",
            OURS_ROOT / "mcp_tools_summary.json",
        ]:
            if path.exists():
                path.unlink()


def _bootstrap_ours_mcp_pool_if_missing() -> Path | None:
    ours_mcp = OURS_ROOT / "mcp_tools.jsonl"
    if ours_mcp.exists() and ours_mcp.stat().st_size > 0:
        return ours_mcp

    candidates = [
        BASELINE_ROOT / "random_sequence" / "mcp_tools.jsonl",
        BASELINE_ROOT / "fixed_simple" / "mcp_tools.jsonl",
        BASELINE_ROOT / "direct_trigger" / "mcp_tools.jsonl",
    ]
    valid = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if not valid:
        return None

    src = sorted(valid, key=lambda p: p.stat().st_size, reverse=True)[0]
    OURS_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ours_mcp)
    _log(f"[INFO] Bootstrap ours MCP pool from: {src}")
    return ours_mcp


def _run_cmd_stream(cmd: List[str], env: dict) -> None:
    _log(f"$ {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if _LOG_FH is not None:
            _LOG_FH.write(line)
        else:
            print(line, end="")
    if _LOG_FH is not None:
        _LOG_FH.flush()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_ours_for_seed(seed: int, max_tasks: int, clean_each: bool, clean_mcp_pool: bool, ours_versions: str) -> None:
    if clean_each:
        _clean_ours_outputs(clean_mcp_pool=clean_mcp_pool)

    mcp_pool = _bootstrap_ours_mcp_pool_if_missing()
    if mcp_pool is None:
        _log("[WARN] Ours MCP pool is missing; run may hit no_tools_loaded.")

    env = _build_env(seed=seed, max_tasks=max_tasks, ours_task_versions=ours_versions)
    cmds = [
        [sys.executable, str(SCRIPT_DIR / "evaluation.py")],
        [sys.executable, str(SCRIPT_DIR / "extract_mcp_tools.py")],
        [sys.executable, str(SCRIPT_DIR / "judge_evaluation.py")],
    ]
    for cmd in cmds:
        try:
            _run_cmd_stream(cmd, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ours run failed at seed {seed}. Command: {' '.join(cmd)}") from e


def run_one_seed(
    seed: int,
    groups: str,
    max_tasks: int,
    clean_each: bool,
    full_output: bool,
    rerun_ours: bool,
    clean_mcp_pool: bool,
    ours_versions: str,
) -> dict:
    if rerun_ours:
        run_ours_for_seed(
            seed=seed,
            max_tasks=max_tasks,
            clean_each=clean_each,
            clean_mcp_pool=clean_mcp_pool,
            ours_versions=ours_versions,
        )

    env = _build_env(seed=seed, max_tasks=max_tasks, ours_task_versions="")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "method_baseline_experiments.py"),
        "run_and_compare",
        "--groups",
        groups,
        "--seed",
        str(seed),
        "--max-tasks",
        str(max_tasks),
    ]
    if clean_each:
        cmd.append("--clean")
    if full_output:
        cmd.append("--full-output")

    try:
        _run_cmd_stream(cmd, env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Seed {seed} failed with exit code {e.returncode}. Command: {' '.join(cmd)}") from e

    if not COMPARISON_JSON.exists():
        raise FileNotFoundError(f"Missing comparison output: {COMPARISON_JSON}")
    return json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))


def extract_seed_metrics(seed: int, comparison: dict) -> dict:
    rows = comparison.get("rows", [])
    matched = comparison.get("matched_rows", [])

    row_map = {r.get("group"): r for r in rows if isinstance(r, dict)}
    match_map = {r.get("pair"): r for r in matched if isinstance(r, dict)}

    metrics = {
        "seed": seed,
        "generated_at": comparison.get("generated_at"),
        "groups": {},
        "matched_pairs": {},
    }

    for group in ["ours", "direct_trigger", "random_sequence", "fixed_simple"]:
        r = row_map.get(group, {})
        metrics["groups"][group] = {
            "n": r.get("valid_scores"),
            "low_score_rate": r.get("low_score_rate"),
            "mean_score": r.get("mean_score"),
            "median_score": r.get("median_score"),
        }

    for pair in [
        "ours_v4_vs_direct_trigger_matched_base_tasks",
        "ours_v1_vs_fixed_simple_matched_base_tasks",
        "ours_vs_random_sequence_matched_exact_task_names",
    ]:
        r = match_map.get(pair, {})
        metrics["matched_pairs"][pair] = {
            "ours_n": r.get("ours_n"),
            "baseline_n": r.get("baseline_n"),
            "ours_low_score_rate": r.get("ours_low_score_rate"),
            "method_baseline_low_score_rate": r.get("method_baseline_low_score_rate"),
            "delta_ours_minus_baseline": r.get("delta_ours_minus_baseline"),
            "ours_mean_score": r.get("ours_mean_score"),
            "method_baseline_mean_score": r.get("method_baseline_mean_score"),
        }

    return metrics


def aggregate(all_seed_metrics: List[dict]) -> dict:
    group_stats: Dict[str, dict] = {}
    for group in ["ours", "direct_trigger", "random_sequence", "fixed_simple"]:
        low_rates = [
            m["groups"][group]["low_score_rate"]
            for m in all_seed_metrics
            if isinstance(m["groups"][group]["low_score_rate"], (int, float))
        ]
        mean_scores = [
            m["groups"][group]["mean_score"]
            for m in all_seed_metrics
            if isinstance(m["groups"][group]["mean_score"], (int, float))
        ]
        group_stats[group] = {
            "low_score_rate": _mean_std([float(v) for v in low_rates]),
            "mean_score": _mean_std([float(v) for v in mean_scores]),
        }

    pair_stats: Dict[str, dict] = {}
    for pair in [
        "ours_v4_vs_direct_trigger_matched_base_tasks",
        "ours_v1_vs_fixed_simple_matched_base_tasks",
        "ours_vs_random_sequence_matched_exact_task_names",
    ]:
        deltas = [
            m["matched_pairs"][pair]["delta_ours_minus_baseline"]
            for m in all_seed_metrics
            if isinstance(m["matched_pairs"][pair]["delta_ours_minus_baseline"], (int, float))
        ]
        ours_low = [
            m["matched_pairs"][pair]["ours_low_score_rate"]
            for m in all_seed_metrics
            if isinstance(m["matched_pairs"][pair]["ours_low_score_rate"], (int, float))
        ]
        baseline_low = [
            m["matched_pairs"][pair]["method_baseline_low_score_rate"]
            for m in all_seed_metrics
            if isinstance(m["matched_pairs"][pair]["method_baseline_low_score_rate"], (int, float))
        ]
        pair_stats[pair] = {
            "delta_ours_minus_baseline": _mean_std([float(v) for v in deltas]),
            "ours_low_score_rate": _mean_std([float(v) for v in ours_low]),
            "method_baseline_low_score_rate": _mean_std([float(v) for v in baseline_low]),
        }

    return {"group_stats": group_stats, "matched_pair_stats": pair_stats}


def write_summary_csv(path: Path, all_seed_metrics: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
        for m in all_seed_metrics:
            seed = m["seed"]
            for pair, row in m["matched_pairs"].items():
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multiple seeds and aggregate method-baseline comparison.")
    p.add_argument("--seeds", default="11,22,33,44,55", help="Comma-separated seeds.")
    p.add_argument(
        "--groups",
        default="direct_trigger,random_sequence,fixed_simple",
        help="Comma-separated baseline groups.",
    )
    p.add_argument(
        "--all-groups-each-seed",
        action="store_true",
        help="Run all groups on every seed (default: only random_sequence runs across seeds).",
    )
    p.add_argument("--max-tasks", type=int, default=0, help="Optional cap per group.")
    p.add_argument("--ours-versions", default="1,2,3,4", help="Only for ours run: comma-separated versions (default: 1,2,3,4)")
    p.add_argument("--clean-each", action="store_true", help="Clean each group output before every seed run.")
    p.add_argument(
        "--clean-mcp-pool",
        action="store_true",
        help="Also delete ours mcp_tools.jsonl during --clean-each (usually not recommended).",
    )
    p.add_argument("--skip-rerun-ours", action="store_true", help="Reuse existing ours result instead of rerunning ours per seed.")
    p.add_argument("--full-output", action="store_true", help="Keep extra artifacts during each run.")
    p.add_argument("--log-dir", default=str(LOG_ROOT), help="Directory to write run logs.")
    return p.parse_args()


def validate_runtime_config() -> None:
    cfg = get_api_config(MODEL)
    api_url = (cfg.get("api_url") or "").strip()
    if not api_url:
        raise SystemExit(
            "API URL is empty. Set OPENAI_API_URL (or BASE_URL/QWEN_API_URL) before running multi-seed."
        )


def main() -> None:
    args = parse_args()
    log_path = _enable_file_logging(Path(args.log_dir))

    _log(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')}", console=False)
    _log(f"args={vars(args)}", console=False)
    _log(f"log_file={log_path}", console=False)
    print(f"Log file: {log_path}")

    try:
        validate_runtime_config()
        seeds = parse_seeds(args.seeds)
        groups = _parse_groups(args.groups)
        has_random = "random_sequence" in groups
        rerun_ours = not args.skip_rerun_ours
        MULTI_SEED_ROOT.mkdir(parents=True, exist_ok=True)

        per_seed = []
        start = time.time()
        reference_comparison: dict | None = None

        if len(seeds) > 1 and not args.all_groups_each_seed:
            if has_random:
                _log("mode=random_only_each_seed", console=False)
            else:
                _log("mode=no_random_sequence_in_groups -> first seed only", console=False)
                seeds = seeds[:1]

        total = len(seeds)
        for i, seed in enumerate(seeds, 1):
            if args.all_groups_each_seed or i == 1:
                seed_groups = ",".join(groups)
            else:
                seed_groups = "random_sequence" if has_random else ",".join(groups)

            rerun_ours_this_seed = rerun_ours and (args.all_groups_each_seed or i == 1)
            print(f"[{i}/{total}] seed={seed} start")
            _log(f"[SEED_START] idx={i}/{total} seed={seed} groups={seed_groups} rerun_ours={rerun_ours_this_seed}")

            comparison = run_one_seed(
                seed=seed,
                groups=seed_groups,
                max_tasks=args.max_tasks,
                clean_each=args.clean_each,
                full_output=args.full_output,
                rerun_ours=rerun_ours_this_seed,
                clean_mcp_pool=args.clean_mcp_pool,
                ours_versions=args.ours_versions,
            )

            if reference_comparison is None:
                reference_comparison = comparison
                effective_comparison = comparison
            elif args.all_groups_each_seed:
                effective_comparison = comparison
            else:
                effective_comparison = _merge_with_reference(
                    reference=reference_comparison,
                    current=comparison,
                    update_random_only=has_random,
                )

            COMPARISON_JSON.write_text(
                json.dumps(effective_comparison, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            seed_dir = MULTI_SEED_ROOT / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(COMPARISON_JSON, seed_dir / "comparison_baseline_vs_ours.json")
            ours_summary = OURS_ROOT / "judge_results" / "judge_scores_summary.json"
            ours_jsonl = OURS_ROOT / "judge_results" / "judge_scores.jsonl"
            if ours_summary.exists():
                shutil.copy2(ours_summary, seed_dir / "ours_judge_scores_summary.json")
            if ours_jsonl.exists():
                shutil.copy2(ours_jsonl, seed_dir / "ours_judge_scores.jsonl")

            seed_metrics = extract_seed_metrics(seed, effective_comparison)
            per_seed.append(seed_metrics)
            (seed_dir / "seed_metrics.json").write_text(
                json.dumps(seed_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            _log(f"[SEED_DONE] idx={i}/{total} seed={seed}")
            print(f"[{i}/{total}] seed={seed} done")

        aggregate_result = aggregate(per_seed)
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seeds": seeds,
            "num_seeds": len(seeds),
            "rerun_ours_each_seed": rerun_ours and args.all_groups_each_seed,
            "multi_seed_strategy": "all_groups_each_seed" if args.all_groups_each_seed else "random_only_each_seed",
            "ours_versions": args.ours_versions,
            "elapsed_seconds": round(time.time() - start, 2),
            "per_seed": per_seed,
            "aggregate": aggregate_result,
        }

        summary_json = MULTI_SEED_ROOT / "summary.json"
        summary_csv = MULTI_SEED_ROOT / "summary.csv"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_csv(summary_csv, per_seed)

        _log(f"summary_json={summary_json}")
        _log(f"summary_csv={summary_csv}")
        print(f"Done. Summary JSON: {summary_json}")
        print(f"Done. Summary CSV: {summary_csv}")

    except Exception as e:
        _log(f"[ERROR] {type(e).__name__}: {e}", console=False)
        _log(traceback.format_exc(), console=False)
        print(f"Run failed: {type(e).__name__}: {e}")
        print(f"See log: {log_path}")
        raise
    finally:
        _disable_file_logging()


if __name__ == "__main__":
    main()
