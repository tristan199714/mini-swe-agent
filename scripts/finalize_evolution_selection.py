#!/usr/bin/env python3
"""Finalize the reviewed evolution dataset.

Usage:
  python scripts/finalize_evolution_selection.py
  python scripts/finalize_evolution_selection.py --seed 42

Reads keep flags from `task_review_sheet.csv` and writes `final/` outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-dir", default="results/swebench_verified_mini/evolution")
    parser.add_argument("--review-sheet", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tasks", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=30)
    return parser.parse_args()


def is_keep(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_keep_ids(review_sheet: Path) -> list[str]:
    keep_ids = []
    with review_sheet.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_keep(row.get("keep", "")):
                keep_ids.append(row["task_id"])
    return keep_ids


def build_run_groups(task_ids: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    guided = [{"task_id": tid, "version": v} for tid in task_ids for v in [1, 2, 3, 4]]
    randomized = guided.copy()
    rng.shuffle(randomized)
    simple_only = [{"task_id": tid, "version": 1} for tid in task_ids]
    direct_trigger = [{"task_id": tid, "version": 4} for tid in task_ids]
    return {
        "group_A_guided": {
            "description": "渐进式引导（v1->v4）",
            "trust_building_rounds": max(1, int(len(guided) * 0.8)),
            "sequence": guided,
        },
        "group_B_random": {
            "description": "随机版本顺序",
            "trust_building_rounds": max(1, int(len(randomized) * 0.8)),
            "sequence": randomized,
        },
        "group_C_simple_only": {
            "description": "固定简单任务（仅v1）",
            "trust_building_rounds": len(simple_only),
            "sequence": simple_only,
        },
        "group_D_direct_trigger": {
            "description": "无信任建立，直接触发（仅v4）",
            "trust_building_rounds": 0,
            "sequence": direct_trigger,
        },
    }


def main() -> None:
    args = parse_args()
    evolution_dir = Path(args.evolution_dir).resolve()
    review_sheet = Path(args.review_sheet).resolve() if args.review_sheet else evolution_dir / "task_review_sheet.csv"

    templates_path = evolution_dir / "evolution_templates.json"
    tasks_dir = evolution_dir / "misevolution_tasks"
    out_dir = evolution_dir / "final"
    out_tasks_dir = out_dir / "misevolution_tasks"

    if not templates_path.exists():
        raise FileNotFoundError(f"Missing templates file: {templates_path}")
    if not review_sheet.exists():
        raise FileNotFoundError(f"Missing review sheet: {review_sheet}")
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Missing task directory: {tasks_dir}")

    keep_ids = load_keep_ids(review_sheet)
    if len(keep_ids) < args.min_tasks:
        raise ValueError(f"Only {len(keep_ids)} kept tasks, less than --min-tasks={args.min_tasks}")
    if len(keep_ids) > args.max_tasks:
        print(
            f"WARNING: {len(keep_ids)} kept tasks exceed --max-tasks={args.max_tasks}. "
            "Proceeding with all kept tasks."
        )

    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    keep_set = set(keep_ids)
    final_items = [item for item in templates if item["task_id"] in keep_set]
    final_ids = [item["task_id"] for item in final_items]

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_tasks_dir.exists():
        shutil.rmtree(out_tasks_dir)
    out_tasks_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "final_selected_task_ids.txt").write_text("\n".join(final_ids) + "\n", encoding="utf-8")
    (out_dir / "evolution_templates_final.json").write_text(
        json.dumps(final_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat = []
    for item in final_items:
        for seq in item["evolution_sequence"]:
            flat.append(
                {
                    "task_id": item["task_id"],
                    "repo": item["repo"],
                    "version": seq["version"],
                    "description": seq["description"],
                    "benign": seq["benign"],
                    "target_feature": item["attack_metadata"]["target_feature"],
                    "trigger_type": item["attack_metadata"]["trigger_type"],
                    "severity": item["attack_metadata"]["severity"],
                }
            )
    (out_dir / "evolution_instances_final.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in flat) + "\n",
        encoding="utf-8",
    )

    manifest = build_run_groups(final_ids, args.seed)
    (out_dir / "experiment_manifest_final.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_tasks": len(final_ids),
                "num_instances_total": len(flat),
                "groups": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for src in tasks_dir.glob("*.json"):
        stem = src.stem
        parts = stem.rsplit("_v", 1)
        if len(parts) != 2:
            continue
        task_slug = parts[0]
        candidate_id = task_slug.replace("_", "-")
        # Use metadata to avoid fragile slug reverse mapping.
        payload = json.loads(src.read_text(encoding="utf-8"))
        task_id = payload.get("metadata", {}).get("task_id")
        if task_id in keep_set:
            shutil.copy2(src, out_tasks_dir / src.name)

    (out_dir / "metadata_final.json").write_text(
        json.dumps(
            {
                "source_review_sheet": str(review_sheet),
                "num_kept_tasks": len(final_ids),
                "num_task_files": len(list(out_tasks_dir.glob("*.json"))),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Final dataset written to: {out_dir}")
    print(f"Kept tasks: {len(final_ids)}")
    print(f"Task files: {len(list(out_tasks_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
