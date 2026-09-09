from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.tools import tool

from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)

_VALID_TICKERS = frozenset({"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "BRK.B"})
_TICKER_RE = re.compile(r"\b(AAPL|MSFT|AMZN|GOOGL|NVDA|META|TSLA|BRK\.B)\b", re.IGNORECASE)


def _make_ingest_tool(pipeline: FinancialGraphRAGPipeline):
    @tool
    def ingest_10k(ticker: str, year: int, use_cache: bool = False) -> str:
        """Ingest SEC 10-K para ticker/año y indexa en grafo+vector. ticker: AAPL|MSFT|AMZN|GOOGL|NVDA|META|TSLA|BRK.B, year: 2020-2026."""
        t = ticker.strip().upper()
        if t == "BRKB":
            t = "BRK.B"
        if t not in _VALID_TICKERS:
            return f"Error: ticker '{t}' no soportado. Soportados: {', '.join(sorted(_VALID_TICKERS))}"
        try:
            y = int(year)
        except Exception:
            return f"Error: year '{year}' invalido"
        if not (2020 <= y <= 2026):
            return f"Error: year {y} fuera de rango 2020-2026"
        try:
            chunks = pipeline.ingest_and_index(ticker=t, year=y, use_cache=use_cache)
            return f"OK: Ingested {len(chunks)} chunks for {t}/{y} -> data/processed_chunks/{t}_{y}/chunks.json + triplets.json (use_cache={use_cache})"
        except Exception as exc:
            logger.exception("ingest_10k failed %s/%s: %s", t, y, exc)
            return f"Error ingesting {t}/{y}: {exc}"

    return ingest_10k


def extract_ticker_year_fallback(question: str) -> tuple[Optional[str], Optional[int]]:
    """Fallback regex si la clasificación LLM no da ticker/year."""
    m = _TICKER_RE.search(question or "")
    ticker = m.group(1).upper() if m else None
    if ticker == "BRKB":
        ticker = "BRK.B"
    y = None
    ym = re.search(r"\b(20(?:2[0-9]|19))\b", question or "")
    if ym:
        try:
            y = int(ym.group(1))
        except Exception:
            pass
    return ticker, y
