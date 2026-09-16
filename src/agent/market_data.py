from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import requests

from src.env import get_finnhub_key

logger = logging.getLogger(__name__)

BASE_URL = "https://finnhub.io/api/v1"
CACHE_TTL_S = 60  # caché en memoria de la sesión: re-uso dentro del turno sin HTTP repetido

_cache: dict[tuple, tuple[float, Any]] = {}

MISSING_KEY_MSG = (
    "Esta función necesita una API key gratuita de Finnhub: regístrate en "
    "https://finnhub.io/register, copia tu key y pégala como FINNHUB_API_KEY=... "
    "en el fichero .env de la raíz del proyecto (junto a GROQ_API_KEY/HF_TOKEN), "
    "y reinicia el chat."
)


def _cached(key: tuple, loader):
    now = time.time()
    if key in _cache:
        ts, value = _cache[key]
        if now - ts < CACHE_TTL_S:
            return value
    value = loader()
    _cache[key] = (now, value)
    return value


def _get(path: str, params: dict) -> dict:
    key = get_finnhub_key()
    if not key:
        raise RuntimeError(MISSING_KEY_MSG)
    try:
        resp = requests.get(
            f"{BASE_URL}{path}",
            params={**params, "token": key},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Error llamando a Finnhub {path}: {exc}")
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Finnhub devolvió error: {data['error']}")
    return data if isinstance(data, dict) else {"result": data}


def _finnhub_symbol(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    # Finnhub usa BRK.B tal cual; variantes sin punto se normalizan
    if t.replace(".", "") == "BRKB":
        return "BRK.B"
    return t


def get_quote(ticker: str) -> dict:
    """Cotización actual: precio, cambio, % cambio, máximo/mínimo del día y cierre previo."""
    symbol = _finnhub_symbol(ticker)

    def _load():
        return _get("/quote", {"symbol": symbol})

    data = _cached(("quote", symbol), _load)
    return {
        "ticker": symbol,
        "current": data.get("c"),
        "change": data.get("d"),
        "change_pct": data.get("dp"),
        "day_high": data.get("h"),
        "day_low": data.get("l"),
        "prev_close": data.get("pc"),
    }


def get_company_news(ticker: str, days: int = 7, max_items: int = 10) -> list[dict]:
    """Noticias recientes: titular, fecha, fuente, URL y resumen."""
    symbol = _finnhub_symbol(ticker)
    days = max(1, min(int(days or 7), 30))
    today = date.today()
    frm = (today - timedelta(days=days)).isoformat()

    def _load():
        data = _get("/company-news", {"symbol": symbol, "from": frm, "to": today.isoformat()})
        items = data.get("result", data if isinstance(data, list) else [])
        return items if isinstance(items, list) else []

    items = _cached(("news", symbol, days), _load)
    out = []
    for n in items[:max_items]:
        out.append({
            "headline": n.get("headline", ""),
            "date": n.get("datetime") and time.strftime("%Y-%m-%d", time.gmtime(n["datetime"])) or "",
            "source": n.get("source", ""),
            "url": n.get("url", ""),
            "summary": (n.get("summary") or "")[:400],
        })
    return out
