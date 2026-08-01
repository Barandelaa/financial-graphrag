from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

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


class GraphTraversalRetriever:
    def __init__(
        self,
        schema: GraphSchema,
        max_hops: int = 3,
    ) -> None:
        self.schema = schema
        self.max_hops = max_hops

    def extract_entities(self, query: str) -> List[str]:
        tickers = re.findall(r"\b[A-Z]{1,5}\b", query)
        words = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
        candidates = list(set(tickers + words))
        logger.debug("Extracted entity candidates from query: %s", candidates)
        return candidates

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

        chunk_ids: Set[str] = set()
        for entity_name, entity_table in matched_entities:
            hops = self._traverse_from_entity(
                conn, entity_name, entity_table
            )
            chunk_ids.update(hops)

        if not chunk_ids:
            return []

        return self._fetch_chunks(conn, list(chunk_ids), top_k)

    def _match_entities(
        self,
        conn: kuzu.Connection,
        candidates: List[str],
    ) -> List[tuple[str, str]]:
        matched: List[tuple[str, str]] = []
        for table in _ENTITY_TABLES:
            pk = _ENTITY_PK[table]
            for name in candidates:
                try:
                    result = conn.execute(
                        f"MATCH (e:{table}) WHERE e.{pk} = $name RETURN e.{pk}",
                        {"name": name},
                    )
                    if result.has_next():
                        matched.append((name, table))
                except RuntimeError:
                    continue
        return matched

    def _traverse_from_entity(
        self,
        conn: kuzu.Connection,
        entity_name: str,
        entity_table: str,
    ) -> Set[str]:
        pk = _ENTITY_PK[entity_table]
        mentions_rel = _MENTIONS_RELS.get(entity_table)
        if mentions_rel is None:
            return set()

        chunk_ids: Set[str] = set()

        query = (
            f"MATCH (c:DocumentChunk)-[:{mentions_rel}]->(e:{entity_table}) "
            f"WHERE e.{pk} = $name "
            f"RETURN c.chunk_id"
        )
        try:
            result = conn.execute(query, {"name": entity_name})
            while result.has_next():
                row = result.get_next()
                chunk_ids.add(str(row[0]))
        except RuntimeError:
            pass

        if self.max_hops >= 2:
            self._traverse_multi_hop(
                conn, entity_name, entity_table, chunk_ids, depth=2
            )

        return chunk_ids

    def _traverse_multi_hop(
        self,
        conn: kuzu.Connection,
        entity_name: str,
        entity_table: str,
        chunk_ids: Set[str],
        depth: int,
    ) -> None:
        if depth > self.max_hops:
            return

        pk = _ENTITY_PK[entity_table]

        for rel in ["OPERATES_IN", "REPORTED_METRIC", "IMPACTS_REVENUE", "MITIGATES_RISK", "COMPETES_WITH"]:
            try:
                result = conn.execute(
                    f"""
                    MATCH (a:{entity_table})-[r:{rel}]->(b)
                    WHERE a.{pk} = $name
                    RETURN b
                    """,
                    {"name": entity_name},
                )
                while result.has_next():
                    row = result.get_next()
                    neighbor = str(row[0])

                    for neighbor_table in _ENTITY_TABLES:
                        n_pk = _ENTITY_PK[neighbor_table]
                        try:
                            check = conn.execute(
                                f"MATCH (n:{neighbor_table}) WHERE n.{n_pk} = $val RETURN n.{n_pk}",
                                {"val": neighbor},
                            )
                            if check.has_next():
                                self._traverse_from_entity(
                                    conn, neighbor, neighbor_table
                                )
                        except RuntimeError:
                            continue
            except RuntimeError:
                continue

    def _fetch_chunks(
        self,
        conn: kuzu.Connection,
        chunk_ids: List[str],
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
                            score=1.0,
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
