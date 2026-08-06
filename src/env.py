from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs) -> bool:  # type: ignore
        return False


def load_env(env_path: str | Path = ".env") -> None:
    load_dotenv(env_path)


_load_env_path = Path(os.getcwd()) / ".env"
if not _load_env_path.exists():
    _load_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env_path)


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")