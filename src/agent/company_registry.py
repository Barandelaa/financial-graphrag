from __future__ import annotations

import difflib
import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEC_UNIVERSE_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CACHE_PATH = Path("data/sec/company_tickers.json")
SEC_CACHE_TTL_DAYS = 30
RESOLUTIONS_PATH = Path("data/sec/resolutions.json")
COMPANIES_CONFIG = Path("data/companies.json")

UA_NAME = "financial_graphrag"
UA_EMAIL = "user@example.com"


def _cleanup(text: str) -> str:
    """Limpieza mínima no semántica: minúsculas, espacios y puntuación. No decide nada."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.&-]", " ", (text or "").lower())).strip()


def _ticker_key(ticker: str) -> str:
    """Clave de comparación exacta: mayúsculas sin espacios ni puntos (BRK.B == BRKB)."""
    return re.sub(r"[\s.]", "", (ticker or "").upper())


def load_universe(force_refresh: bool = False) -> dict:
    """Universo oficial SEC {TICKER: {cik, name}}. Cache local con TTL; sin red usa caché."""
    if SEC_CACHE_PATH.exists() and not force_refresh:
        try:
            age_days = (time.time() - SEC_CACHE_PATH.stat().st_mtime) / 86400
            if age_days < SEC_CACHE_TTL_DAYS:
                with open(SEC_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("No se pudo leer caché SEC %s: %s", SEC_CACHE_PATH, exc)
    req = urllib.request.Request(
        SEC_UNIVERSE_URL,
        headers={"User-Agent": f"{UA_NAME} {UA_EMAIL}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.load(resp)
    except Exception as exc:
        logger.warning("No se pudo descargar universo SEC, usando caché si existe: %s", exc)
        if SEC_CACHE_PATH.exists():
            with open(SEC_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        raise RuntimeError(f"Sin universo SEC (ni red ni caché): {exc}")
    universe: dict = {}
    for entry in raw.values() if isinstance(raw, dict) else raw:
        try:
            universe[str(entry["ticker"]).upper()] = {
                "cik": str(entry["cik_str"]).zfill(10),
                "name": str(entry["title"]),
            }
        except (KeyError, TypeError):
            continue
    SEC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEC_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(universe, f, indent=2, ensure_ascii=False)
    logger.info("Universo SEC actualizado: %d tickers", len(universe))
    return universe


def get_config_tickers(config_path: str | Path = COMPANIES_CONFIG) -> list[str]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return [str(t).upper() for t in cfg.get("companies", [])]
    except Exception:
        return []


def get_config_years(config_path: str | Path = COMPANIES_CONFIG) -> list[int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return [int(y) for y in cfg.get("default_years", [])]
    except Exception:
        return []


def resolve_company(text: str, universe: Optional[dict] = None, top_n: int = 5) -> dict:
    """Resuelve texto libre contra el universo SEC.

    - Coincidencia exacta de ticker (case-insensitive, con/sin punto) -> vía rápida, sin pregunta.
    - Resto -> candidatos difusos rankeados; el agente DEBE preguntar al usuario.
    """
    universe = universe if universe is not None else load_universe()
    cleaned = _cleanup(text)
    nospace = re.sub(r"[\s.]", "", cleaned).upper()

    by_key = {_ticker_key(t): t for t in universe}
    if nospace in by_key:
        t = by_key[nospace]
        return {
            "exact": True,
            "ticker": t,
            "cik": universe[t]["cik"],
            "name": universe[t]["name"],
            "candidates": [],
        }

    # El usuario escribió el ticker literal ("AAPL", "añade AAPL 2026"): vía rápida.
    # Si el ticker aparece junto a mucho más texto, se devuelve como candidato
    # para preguntar (puede ser ambiguo: "Dow" en "Dow Jones").
    tokens = re.findall(r"[a-z0-9.]{1,10}", cleaned)
    found = [by_key[_ticker_key(tok)] for tok in tokens if _ticker_key(tok) in by_key]
    if len(set(found)) == 1:
        tick = found[0]
        rest = cleaned
        # quita solo el ticker encontrado (con y sin punto)
        rest = re.sub(r"\b" + re.escape(tick.lower()) + r"\b", " ", rest)
        rest = re.sub(r"\b" + re.escape(tick.lower().replace(".", "")) + r"\b", " ", rest)
        rest = re.sub(r"\b20(?:2[0-9]|19)\b", " ", rest)  # quita años
        rest = re.sub(
            r"\b(añade|añadir|ingesta|ingestar|agrega|agregar|add|ingest|de|la|el|en|para|por|con|y|a|the|to|for|of|in|on)\b",
            " ",
            rest,
        )
        if len(rest.split()) <= 2:
            t = found[0]
            return {
                "exact": True,
                "ticker": t,
                "cik": universe[t]["cik"],
                "name": universe[t]["name"],
                "candidates": [],
            }

    names = [f"{t} {info['name']}" for t, info in universe.items()]
    lowered = {n.lower(): n for n in names}
    matches = difflib.get_close_matches(cleaned, list(lowered.keys()), n=top_n, cutoff=0.5)
    candidates = []
    # tickers literales encontrados en el texto van primeros (score 1.0) pero exigen pregunta
    for t in dict.fromkeys(found):
        info = universe.get(t, {})
        candidates.append({"ticker": t, "cik": info.get("cik", ""), "name": info.get("name", t), "score": 1.0})
    for m in matches:
        full = lowered[m]
        ticker = full.split(" ", 1)[0]
        info = universe.get(ticker, {})
        candidates.append({
            "ticker": ticker,
            "cik": info.get("cik", ""),
            "name": info.get("name", full),
            "score": round(difflib.SequenceMatcher(None, cleaned, m).ratio(), 3),
        })
    return {"exact": False, "ticker": None, "cik": None, "name": None, "candidates": candidates}


def verify_10k(cik: str, ticker: str = "") -> dict:
    """Verifica en EDGAR que el CIK tiene 10-K recientes. Verdad absoluta antes de proponer ingesta."""
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    req = urllib.request.Request(
        url, headers={"User-Agent": f"{UA_NAME} {UA_EMAIL}", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except Exception as exc:
        return {"ok": False, "reason": f"No se pudo consultar EDGAR para {ticker or cik}: {exc}"}
    try:
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        tens = [d for f, d in zip(forms, dates) if f == "10-K"][:3]
        if tens:
            return {"ok": True, "recent_10k": tens, "cik": cik_padded}
        return {"ok": False, "reason": f"{ticker or cik} no tiene 10-K recientes en EDGAR (puede ser un índice o emisor extranjero con 20-F)."}
    except Exception as exc:
        return {"ok": False, "reason": f"Respuesta EDGAR inesperada: {exc}"}


def add_company(ticker: str, years: Optional[list[int]] = None) -> dict:
    """Añade ticker a companies.json con backup .bak. Devuelve diff antes→después."""
    t = _ticker_key(ticker)
    # recupera ticker canónico (con punto si aplica)
    universe = load_universe()
    canon = {_ticker_key(k): k for k in universe}.get(t, ticker.strip().upper())
    cfg_path = COMPANIES_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    before = list(cfg.get("companies", []))
    if canon not in before:
        backup = cfg_path.with_suffix(".json.bak")
        backup.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        cfg["companies"] = sorted(set(before) | {canon})
        if years:
            cfg["default_years"] = sorted(set(cfg.get("default_years", [])) | {int(y) for y in years})
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        # revalida
        with open(cfg_path, "r", encoding="utf-8") as f:
            json.load(f)
    return {"ticker": canon, "before": before, "after": cfg["companies"], "years": cfg.get("default_years", [])}


def record_resolution(user_text: str, ticker: str) -> None:
    """Caché de resoluciones confirmadas por el usuario (aprendida, no curada)."""
    try:
        RESOLUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if RESOLUTIONS_PATH.exists():
            with open(RESOLUTIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[_cleanup(user_text)] = ticker.upper()
        with open(RESOLUTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.debug("No se pudo guardar resolución: %s", exc)
