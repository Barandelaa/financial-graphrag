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


def get_finnhub_key() -> str | None:
    return os.getenv("FINNHUB_API_KEY")


def refresh_env() -> None:
    """Relee .env de disco sin sobreescribir variables ya exportadas.

    Cubre el caso habitual: el .env se crea/pega la key DESPUÉS de arrancar
    el servidor o el CLI (el load inicial ya pasó y no la vería nunca).
    """
    load_dotenv(_load_env_path)


def clean_key(value: str | None) -> str | None:
    """Limpia comillas/espacios accidentales al pegar la key en el .env."""
    if value is None:
        return None
    cleaned = value.strip().strip("\"'").strip()
    return cleaned or None