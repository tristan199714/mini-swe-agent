"""Model/API configuration for workflow and evaluator runs.

Set environment variables before running:
  OPENAI_API_URL / BASE_URL
  MODEL
  API_KEY
"""

import os

# Generic API defaults. Override with environment variables before running.
BASE_URL = os.getenv("BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL = os.getenv("MODEL", "qwen/qwen3.5-flash-02-23")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "API key not set. Export OPENROUTER_API_KEY (or API_KEY) in your environment."
    )


def get_api_config(model_name: str) -> dict:
    """Return API config for the selected model."""
    api_url = os.getenv("OPENAI_API_URL", BASE_URL)
    # OpenRouter or any non-Qwen-direct API: always use bearer
    is_openrouter = "openrouter.ai" in api_url
    lower = model_name.lower()
    if lower.startswith("qwen") and not is_openrouter:
        return {
            "api_url": os.getenv("QWEN_API_URL", BASE_URL),
            "auth_type": "basic",
            "api_ak": os.getenv("API_AK", ""),
            "api_sk": os.getenv("API_SK", ""),
            "api_key": API_KEY,
        }
    return {
        "api_url": api_url,
        "auth_type": "bearer",
        "api_key": API_KEY,
        "api_ak": "",
        "api_sk": "",
    }
