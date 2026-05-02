#!/usr/bin/env python3
"""Download SWE-bench Verified Mini and prepare task candidates.

Usage:
  python scripts/download_verified_mini.py
  python scripts/download_verified_mini.py --split test --max-select 30

Outputs are written under `results/swebench_verified_mini` by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET = "MariusHobbhahn/swe-bench-verified-mini"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_DIR = "results/swebench_verified_mini"

REQUIRED_FIELDS = [
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "patch",
    "test_patch",
]

INCLUDE_KEYWORDS = {
    "create",
    "implement",
    "add",
    "write",
    "build",
    "construct",
    "function",
    "class",
    "method",
    "api",
    "interface",
    "utility",
    "helper",
    "tool",
    "script",
    "module",
    "cache",
    "logger",
    "parser",
    "handler",
    "formatter",
    "validator",
}

SCENARIO_KEYWORDS = {
    "file_io": ["file", "path", "read", "write", "open", "directory"],
    "cache": ["cache", "ttl", "expire", "memo", "persist"],
    "config": ["config", "option", "parameter", "env", "settings"],
    "network": ["request", "http", "https", "socket", "url", "api"],
    "serialization": ["serialize", "deserialize", "json", "yaml", "pickle", "marshal"],
}


@dataclass
class Candidate:
    instance_id: str
    repo: str
    score: int
    matched_include: list[str]
    matched_scenarios: list[str]
    problem_statement: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split name")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--min-select", type=int, default=20, help="Minimum selected candidates")
    parser.add_argument("--max-select", type=int, default=30, help="Maximum selected candidates")
    return parser.parse_args()


def _tokenize(text: str) -> set[str]:
    lowered = text.lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return {tok for tok in cleaned.split() if tok}


def validate_fields(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Dataset split is empty.")
    first = rows[0]
    missing = [field for field in REQUIRED_FIELDS if field not in first]
    if missing:
        raise ValueError(f"Dataset missing required fields: {missing}")


def score_candidate(row: dict) -> Candidate | None:
    text = row["problem_statement"]
    tokens = _tokenize(text)

    matched_include = sorted(kw for kw in INCLUDE_KEYWORDS if kw in tokens)
    if not matched_include:
        return None

    matched_scenarios = []
    for scenario, kws in SCENARIO_KEYWORDS.items():
        if any(kw in tokens for kw in kws):
            matched_scenarios.append(scenario)

    score = len(matched_include) * 2 + len(matched_scenarios) * 3
    return Candidate(
        instance_id=row["instance_id"],
        repo=row["repo"],
        score=score,
        matched_include=matched_include,
        matched_scenarios=matched_scenarios,
        problem_statement=text,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.min_select > args.max_select:
        raise ValueError("--min-select must be <= --max-select")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install with: pip install datasets huggingface_hub"
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading dataset: {args.dataset} (split={args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(item) for item in dataset]

    print(f"[2/4] Validating fields and exporting local copy ({len(rows)} rows)")
    validate_fields(rows)
    write_jsonl(output_dir / f"{args.split}.jsonl", rows)
    (output_dir / "instance_ids.txt").write_text(
        "\n".join(row["instance_id"] for row in rows) + "\n",
        encoding="utf-8",
    )

    print("[3/4] Selecting attack-friendly task candidates by keyword rules")
    candidates = [c for c in (score_candidate(row) for row in rows) if c is not None]
    candidates.sort(key=lambda x: (-x.score, x.instance_id))

    selected = candidates[: args.max_select]
    if len(selected) < args.min_select:
        print(
            f"WARNING: only {len(selected)} candidates found, less than --min-select={args.min_select}. "
            "You may need to relax keyword constraints."
        )

    selected_ids = [c.instance_id for c in selected]
    (output_dir / "selected_task_ids.txt").write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    (output_dir / "selected_tasks.json").write_text(
        json.dumps(
            [
                {
                    "task_id": c.instance_id,
                    "repo": c.repo,
                    "selection_score": c.score,
                    "matched_include_keywords": c.matched_include,
                    "matched_scenarios": c.matched_scenarios,
                    "base_description": c.problem_statement,
                }
                for c in selected
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[4/4] Writing metadata")
    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(rows),
        "required_fields": REQUIRED_FIELDS,
        "selected_count": len(selected),
        "output_files": [
            f"{args.split}.jsonl",
            "instance_ids.txt",
            "selected_task_ids.txt",
            "selected_tasks.json",
            "metadata.json",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done. Output directory: {output_dir}")
    print("Next: use selected_task_ids.txt to build your v1->v4 evolution sequences.")


if __name__ == "__main__":
    main()
