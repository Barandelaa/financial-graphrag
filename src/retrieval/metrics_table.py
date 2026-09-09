from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import kuzu

from src.graph.schema import GraphSchema

logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r"\b(20(?:2[0-9]|19))\b")
_TICKER_RE = re.compile(r"\b(AAPL|MSFT|AMZN|GOOGL|NVDA|META|TSLA|BRK\.B)\b", re.IGNORECASE)
# métricas comunes para filtrar la tabla si la pregunta es específica
_METRIC_KEYWORDS = [
    "revenue", "sales", "net income", "net earnings", "eps", "operating income",
    "gross margin", "total assets", "cash", "debt", "liabilities", "equity",
]

@dataclass
class MetricRow:
    ticker: str
    year: int
    metric: str
    value: Optional[str]
    unit: Optional[str]
    metric_id: str
    chunk_id: Optional[str] = None

    def to_table_row(self) -> str:
        val = f"{self.value} {self.unit or ''}".strip()
        if not val:
            val = "(no value extracted)"
        return f"| {self.ticker} | {self.year} | {self.metric} | {val} | {self.metric_id} | {self.chunk_id or ''} |"

class MetricsTableRetriever:
    """Recupera métricas scoping por ticker/año para evitar la ambigüedad del grafo anterior.

    Antes FinancialMetric.id = 'revenue' colisionaba entre empresas. Ahora
    id = 'TICKER_YEAR_slug' y cada fila trae ticker/year/value con chunk_id para citar.
    """

    def __init__(self, schema: GraphSchema, max_rows: int = 20):
        self.schema = schema
        self.max_rows = max_rows

    def search(self, question: str) -> List[MetricRow]:
        tickers = self._extract_tickers(question)
        years = self._extract_years(question)
        # Si la pregunta no menciona ticker/año, no devuelvas tabla gigante
        if not tickers:
            return []
        conn = self.schema.connection
        rows: List[MetricRow] = []
        try:
            # Métricas directamente reportadas por la compañía scoping
            query = """
            MATCH (c:Company)-[:REPORTED_METRIC]->(m:FinancialMetric)
            WHERE c.ticker IN $tickers
            AND m.id CONTAINS '_'
            """
            params: dict = {"tickers": tickers}
            if years:
                query += " AND m.fiscal_year IN $years"
                params["years"] = years
            # Nota: no filtramos por keyword aquí para evitar falsos negativos
            # (ej. "revenue" vs "total net sales"). La tabla scoping ya limita por ticker/año.
            query += " RETURN c.ticker, m.fiscal_year, m.name, m.value, m.unit, m.id LIMIT $limit"
            params["limit"] = self.max_rows
            result = conn.execute(query, params)
            while result.has_next():
                row = result.get_next()
                ticker, fy, name, value, unit, mid = row[0], row[1], row[2], row[3], row[4], row[5]
                # Busca un chunk que mencione esta métrica para citar
                chunk_id = self._find_chunk_for_metric(conn, str(mid))
                rows.append(MetricRow(
                    ticker=str(ticker), year=int(fy) if fy else 0,
                    metric=str(name), value=str(value) if value else None,
                    unit=str(unit) if unit else None,
                    metric_id=str(mid), chunk_id=chunk_id
                ))
        except Exception as exc:
            logger.debug("Metrics query failed: %s", exc)
        return rows

    def _find_chunk_for_metric(self, conn: kuzu.Connection, metric_id: str) -> Optional[str]:
        try:
            r = conn.execute(
                "MATCH (d:DocumentChunk)-[:MENTIONS_METRIC]->(m:FinancialMetric) WHERE m.id = $mid RETURN d.chunk_id LIMIT 1",
                {"mid": metric_id}
            )
            if r.has_next():
                return str(r.get_next()[0])
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_tickers(question: str) -> List[str]:
        hits = _TICKER_RE.findall(question or "")
        # normaliza a mayúsculas y deduplica manteniendo orden
        seen = []
        for h in hits:
            up = h.upper().replace("BRK.B", "BRK.B")
            if up == "BRKB":
                up = "BRK.B"
            if up not in seen:
                seen.append(up)
        return seen

    @staticmethod
    def _extract_years(question: str) -> List[int]:
        years = [int(y) for y in _YEAR_RE.findall(question or "")]
        # filtra a años del corpus (2024-2025) si no hay otros, pero permite otros años sin filtrar
        return sorted(set(years))

    def format_as_block(self, rows: List[MetricRow]) -> str:
        if not rows:
            return ""
        header = "| ticker | year | metric | value | metric_id | chunk_id |"
        sep = "|---|---|---|---|---|---|"
        lines = [header, sep] + [r.to_table_row() for r in rows]
        return "\n".join(lines)
