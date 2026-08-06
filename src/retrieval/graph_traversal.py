from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import kuzu

from src.graph.schema import GraphSchema

logger = logging.getLogger(__name__)


@dataclass
class GraphSearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    traversal_path: str


_ENTITY_TABLES: List[str] = [
    "Company",
    "FinancialMetric",
    "RiskFactor",
    "BusinessSegment",
    "MacroEvent",
]

_ENTITY_PK: Dict[str, str] = {
    "Company": "ticker",
    "FinancialMetric": "id",
    "RiskFactor": "id",
    "BusinessSegment": "id",
    "MacroEvent": "id",
}

_MENTIONS_RELS: Dict[str, str] = {
    "Company": "MENTIONS_COMPANY",
    "FinancialMetric": "MENTIONS_METRIC",
    "RiskFactor": "MENTIONS_RISK",
    "BusinessSegment": "MENTIONS_SEGMENT",
    "MacroEvent": "MENTIONS_EVENT",
}

_REL_TARGETS: Dict[str, tuple[str, str]] = {
    "OPERATES_IN": ("BusinessSegment", "id"),
    "REPORTED_METRIC": ("FinancialMetric", "id"),
    "IMPACTS_REVENUE": ("FinancialMetric", "id"),
    "MITIGATES_RISK": ("RiskFactor", "id"),
    "COMPETES_WITH": ("Company", "ticker"),
}

# ticker -> nombres/alias con los que los usuarios se refieren a cada empresa
_COMPANY_ALIASES: Dict[str, List[str]] = {
    "AAPL": ["Apple", "Apple Inc", "Apple Inc."],
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "AMZN": ["Amazon", "Amazon.com", "Amazon Inc"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc", "Google Inc"],
    "META": ["Meta", "Meta Platforms", "Facebook"],
    "TSLA": ["Tesla", "Tesla Inc"],
    "NVDA": ["NVIDIA", "Nvidia", "Nvidia Corporation"],
    "JPM": ["JPMorgan", "JPMorgan Chase", "JPMorgan Chase & Co", "JPMorgan Chase "],
    "BA": ["Boeing", "The Boeing Company"],
    "JNJ": ["Johnson & Johnson", "J&J"],
    "KO": ["Coca-Cola", "Coca Cola", "The Coca-Cola Company"],
    "PFE": ["Pfizer", "Pfizer Inc"],
    "WMT": ["Walmart", "Walmart Inc"],
    "XOM": ["ExxonMobil", "Exxon Mobil", "Exxon"],
    "BRK.B": ["Berkshire Hathaway", "Berkshire"],
}

# lower(name) -> ticker para mapear lo que el usuario escribe a la PK del grafo
_NAME_TO_TICKER: Dict[str, str] = {
    alias.lower(): ticker
    for ticker, aliases in _COMPANY_ALIASES.items()
    for alias in aliases
}

_STOPWORDS: frozenset = frozenset(
    {
        "what", "how", "was", "were", "did", "does", "its", "the", "and",
        "for", "of", "in", "to", "a", "an", "is", "are", "from", "by",
        "with", "about", "that", "this", "their", "have", "has", "as", "at",
        "do", "it", "on", "which", "per", "fiscal", "year", "total", "figure",
        "report", "reports", "operation", "operations", "business", "result",
        "results", "statement", "statements", "note", "notes", "much", "many",
        "would", "should", "could", "between", "than", "then", "more", "most",
        "due", "into", "during", "over", "under", "each", "other", "used",
        "use", "using", "based", "both", "better", "also", "still",
    }
)


class GraphTraversalRetriever:
    def __init__(
        self,
        schema: GraphSchema,
        max_hops: int = 3,
    ) -> None:
        self.schema = schema
        self.max_hops = max_hops

    def extract_entities(self, query: str) -> List[str]:
        candidates: set[str] = set()

        # Tickers en mayúsculas (AAPL, MSFT...) y palabras en CamelCase
        tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
        candidates.update(tickers)

        # Palabras Capitalizadas (Apple, Boeing, RiskFactor...) y mapear a ticker
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*\b", query)
        for word in capitalized:
            if word.lower() in _STOPWORDS:
                continue
            candidates.add(word)
            alias = _NAME_TO_TICKER.get(word.lower())
            if alias:
                candidates.add(alias)

        # Palabras en minúscula (net income, supply chain, revenue...)
        lower_words = re.findall(r"\b[a-z][a-z]+(?:\s[a-z][a-z]+)*\b", query)
        for word in lower_words:
            token = word.lower()
            if token in _STOPWORDS:
                continue
            candidates.add(word.lower())
            alias = _NAME_TO_TICKER.get(token)
            if alias:
                candidates.add(alias)

        # Expande n-gramas: para query "net income 2023", añade "net income"
        compact = re.sub(r"[\s'-]+", " ", query).lower()
        clean = re.sub(r"\b(?:the|and|of|in|for|net|gross|total|increased|decreased)\b", " ", compact)
        words2 = [w for w in clean.split() if w and w not in _STOPWORDS]
        if len(words2) >= 2:
            joined = " ".join(words2)
            candidates.add(joined)
        candidates = {c.strip() for c in candidates if c and len(c.strip()) >= 2}

        logger.debug("Extracted entity candidates from query: %s", candidates)
        return sorted(candidates)

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[GraphSearchResult]:
        entity_candidates = self.extract_entities(query)
        if not entity_candidates:
            return []

        conn = self.schema.connection
        matched_entities = self._match_entities(conn, entity_candidates)
        if not matched_entities:
            return []

        chunk_scores: Dict[str, float] = {}
        for entity_name, entity_table in matched_entities:
            self._score_entity_hops(
                conn, entity_name, entity_table, depth=1, chunk_scores=chunk_scores
            )

        if not chunk_scores:
            return []

        ranked_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
        return self._fetch_chunks(conn, ranked_ids, chunk_scores, top_k)

    def _score_entity_hops(
        self,
        conn: kuzu.Connection,
        entity_name: str,
        entity_table: str,
        depth: int,
        chunk_scores: Dict[str, float],
    ) -> None:
        if depth > self.max_hops:
            return

        weight = self._depth_weight(depth)
        pk = _ENTITY_PK[entity_table]
        mentions_rel = _MENTIONS_RELS.get(entity_table)

        if mentions_rel is not None:
            query = (
                f"MATCH (c:DocumentChunk)-[:{mentions_rel}]->(e:{entity_table}) "
                f"WHERE e.{pk} = $name "
                f"RETURN c.chunk_id"
            )
            try:
                result = conn.execute(query, {"name": entity_name})
                while result.has_next():
                    row = result.get_next()
                    cid = str(row[0])
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), weight)
            except RuntimeError as exc:
                logger.debug("Mentions scan failed at depth %d: %s", depth, exc)

        if depth >= self.max_hops:
            return

        for rel, (dst_table, dst_pk) in _REL_TARGETS.items():
            try:
                result = conn.execute(
                    f"""
                    MATCH (a:{entity_table})-[r:{rel}]->(b:{dst_table})
                    WHERE a.{pk} = $name
                    RETURN DISTINCT b.{dst_pk}
                    """,
                    {"name": entity_name},
                )
                while result.has_next():
                    row = result.get_next()
                    neighbor = str(row[0])
                    self._score_entity_hops(
                        conn, neighbor, dst_table, depth + 1, chunk_scores
                    )
            except RuntimeError:
                continue

    @staticmethod
    def _depth_weight(depth: int) -> float:
        return 1.0 / depth

    def _match_entities(
        self,
        conn: kuzu.Connection,
        candidates: List[str],
    ) -> List[tuple[str, str]]:
        matched: List[tuple[str, str]] = []
        for table in _ENTITY_TABLES:
            pk = _ENTITY_PK[table]
            for name in candidates:
                found = self._lookup_entity(conn, table, pk, name)
                if found and (found, table) not in matched:
                    matched.append((found, table))
        return matched

    def _lookup_entity(
        self,
        conn: kuzu.Connection,
        table: str,
        pk: str,
        name: str,
    ) -> Optional[str]:
        # 1) Igualdad exacta sobre PK (ticker/id)
        if table == "Company":
            value = _NAME_TO_TICKER.get(name.lower(), name)
            found = self._run_lookup(conn, table, pk, f"e.{pk} = $n", value)
            if found:
                return found

        # 2) Igualdad exacta sobre el nombre (case-insensitive)
        found = self._run_lookup(
            conn, table, pk, "LOWER(e.name) = LOWER($n)", name
        )
        if found:
            return found

        # 3) Subcadena sobre el nombre (caso-insensitive)
        found = self._run_lookup(
            conn, table, pk, "LOWER(e.name) CONTAINS LOWER($n)", name
        )
        return found

    def _run_lookup(
        self,
        conn: kuzu.Connection,
        table: str,
        pk: str,
        cond: str,
        value: str,
    ) -> Optional[str]:
        query = f"MATCH (e:{table}) WHERE {cond} RETURN e.{pk} LIMIT 1"
        try:
            result = conn.execute(query, {"n": value})
            if result.has_next():
                return str(result.get_next()[0])
        except RuntimeError:
            logger.debug("Lookup failed on %s with %s: %s", table, cond, value)
        return None

    def _fetch_chunks(
        self,
        conn: kuzu.Connection,
        chunk_ids: List[str],
        chunk_scores: Dict[str, float],
        top_k: int,
    ) -> List[GraphSearchResult]:
        results: List[GraphSearchResult] = []
        for cid in chunk_ids[:top_k]:
            try:
                result = conn.execute(
                    """
                    MATCH (c:DocumentChunk)
                    WHERE c.chunk_id = $chunk_id
                    RETURN c.chunk_id, c.text, c.company_ticker,
                           c.fiscal_year, c.section_id, c.page_number
                    """,
                    {"chunk_id": cid},
                )
                if result.has_next():
                    row = result.get_next()
                    results.append(
                        GraphSearchResult(
                            chunk_id=str(row[0]),
                            text=str(row[1]),
                            score=chunk_scores.get(str(row[0]), 0.0),
                            metadata={
                                "company_ticker": str(row[2]),
                                "fiscal_year": str(row[3]),
                                "section_id": str(row[4]),
                                "page_number": str(row[5]),
                            },
                            traversal_path=f"entity_graph_{cid[:8]}",
                        )
                    )
            except RuntimeError:
                continue

        return results[:top_k]
