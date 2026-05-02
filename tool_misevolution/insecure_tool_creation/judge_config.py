import os
from pathlib import Path

# Default judge configuration. CLI env vars override these values.
# Usage:
#   cd tool_misevolution/insecure_tool_creation
#   python judge_evaluation.py
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.5-flash-02-23")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not JUDGE_API_KEY:
    raise RuntimeError(
        "Judge API key not set. Export OPENROUTER_API_KEY (or JUDGE_API_KEY) in your environment."
    )
JUDGE_API_PATH = os.getenv("JUDGE_API_PATH", "api/v1/chat/completions")
JUDGE_TIMEOUT = int(os.getenv("JUDGE_TIMEOUT", "60"))

# Evaluation settings
ALL_STEPS = os.getenv("ALL_STEPS", "0") == "1"  # env override; True to judge all steps
JUDGE_VERSIONS = os.getenv("JUDGE_VERSIONS", "").strip()  # e.g. "3,4" to only judge v3/v4 tools
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_NAME = os.getenv("EXPERIMENT_DATASET", "misevolution_user_queries")
OUTPUT_ROOT = Path(os.getenv("EXPERIMENT_OUTPUT_ROOT", str(DEFAULT_RESULTS_ROOT)))
OURS_VARIANT = os.getenv("OURS_VARIANT", "ours")
OURS_DIR = OUTPUT_ROOT / DATASET_NAME / "evolution" / OURS_VARIANT
DEFAULT_TASKS_DIR = DEFAULT_RESULTS_ROOT / DATASET_NAME / "evolution" / "misevolution_tasks"
if not DEFAULT_TASKS_DIR.exists():
    alt_tasks = PROJECT_ROOT / "data" / DATASET_NAME / "evolution" / "misevolution_tasks"
    if alt_tasks.exists():
        DEFAULT_TASKS_DIR = alt_tasks
TRAJS_DIR = os.getenv("TRAJS_DIR", str(OURS_DIR / "trajectories"))
TASKS_DIR = os.getenv("TASKS_DIR", str(DEFAULT_TASKS_DIR))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(OURS_DIR / "judge_results"))
