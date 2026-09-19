from __future__ import annotations

import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


class MissingFinnhubKeyError(RuntimeError):
    """Sin FINNHUB_API_KEY ni en entorno ni en .env. Distinta de errores de red
    para que las tools respondan el mensaje correcto en cada caso."""


def _redact_token(text: str, token: str | None) -> str:
    """Evita que la key aparezca en claro en logs y mensajes de error."""
    if token and text:
        return text.replace(token, "***")
    return text


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
    from src.env import clean_key, refresh_env

    key = clean_key(get_finnhub_key())
    if not key:
        # El .env pudo crearse tras arrancar: reintenta leyéndolo de disco.
        refresh_env()
        key = clean_key(get_finnhub_key())
    if not key:
        raise MissingFinnhubKeyError(MISSING_KEY_MSG)
    try:
        resp = requests.get(
            f"{BASE_URL}{path}",
            params={**params, "token": key},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"Error llamando a Finnhub {path}: {_redact_token(str(exc), key)}"
        )
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


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _resolve_direct_url(url: str) -> str:
    """Los enlaces de Finnhub (/api/news?id=...) son redirecciones que devuelven
    403 sin cabeceras de navegador. Se resuelven una vez aquí para entregar
    URLs directas clicables; si falla, se conserva la original."""
    if not url or "finnhub.io" not in url:
        return url
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=10) as resp:
            final = resp.geturl()
            return final if final and final != url else url
    except Exception:
        return url


def _company_keywords(ticker: str) -> set[str]:
    """Palabras para el fallback por mención: ticker + nombre legal del universo SEC."""
    words = {ticker.upper(), ticker.upper().replace(".", "")}
    try:
        from src.agent.company_registry import load_universe

        info = load_universe().get(ticker.upper(), {})
        for token in (info.get("name") or "").replace(".", " ").split():
            if len(token) > 2:
                words.add(token.upper())
    except Exception:
        pass
    return words


def _news_relevance(item: dict, symbol: str, keywords: set[str]) -> int:
    """Finnhub etiqueta cada noticia con `related` (tickers implicados): es el
    filtro preciso. Si viene vacío, fallback por mención en titular/resumen."""
    related = {(r or "").strip().upper() for r in str(item.get("related", "")).split(",")}
    if symbol in related or symbol.replace(".", "") in {r.replace(".", "") for r in related}:
        return 3
    text = f"{item.get('headline', '')} {item.get('summary', '')}".upper()
    if any(k in text for k in keywords if len(k) > 1):
        return 1
    return 0


def get_company_news(ticker: str, days: int = 7, max_items: int = 6) -> list[dict]:
    """Noticias recientes relevantes: titular, fecha, fuente, URL y resumen corto.

    Filtra por campo `related` de Finnhub (fallback por mención) y ordena por
    fecha: solo entran noticias que implican a la empresa, máximo `max_items`
    con resúmenes cortos para no agotar el contexto del modelo.
    """
    symbol = _finnhub_symbol(ticker)
    days = max(1, min(int(days or 7), 30))
    today = date.today()
    frm = (today - timedelta(days=days)).isoformat()

    def _load():
        data = _get("/company-news", {"symbol": symbol, "from": frm, "to": today.isoformat()})
        items = data.get("result", data if isinstance(data, list) else [])
        return items if isinstance(items, list) else []

    items = _cached(("news", symbol, days), _load)
    keywords = _company_keywords(symbol)
    scored = [(_news_relevance(n, symbol, keywords), n.get("datetime") or 0, n) for n in items]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    top = [n for _, _, n in scored[:max_items]]
    # Resuelve redirecciones Finnhub en paralelo (tolerante a fallos)
    urls = [n.get("url", "") for n in top]
    with ThreadPoolExecutor(max_workers=4) as pool:
        resolved = list(pool.map(_resolve_direct_url, urls))
    out = []
    for n, url in zip(top, resolved):
        out.append({
            "headline": n.get("headline", ""),
            "date": n.get("datetime") and time.strftime("%Y-%m-%d", time.gmtime(n["datetime"])) or "",
            "source": n.get("source", ""),
            "url": url,
            "summary": (n.get("summary") or "")[:220],
        })
    return out
