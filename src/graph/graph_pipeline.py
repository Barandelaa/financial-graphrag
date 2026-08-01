from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import kuzu
from langchain_core.language_models import BaseChatModel

from src.graph.extractor import (
    Entity,
    EntityType,
    FinancialTriplet,
    RelationType,
    TripletExtractor,
)
from src.graph.schema import GraphConfig, GraphSchema
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_NODE_TABLE_MAP: dict[EntityType, str] = {
    EntityType.company: "Company",
    EntityType.financial_metric: "FinancialMetric",
    EntityType.risk_factor: "RiskFactor",
    EntityType.business_segment: "BusinessSegment",
    EntityType.macro_event: "MacroEvent",
}

_REL_TABLE_MAP: dict[RelationType, str] = {
    RelationType.operates_in: "OPERATES_IN",
    RelationType.reported_metric: "REPORTED_METRIC",
    RelationType.impacts_revenue: "IMPACTS_REVENUE",
    RelationType.mitigates_risk: "MITIGATES_RISK",
    RelationType.competes_with: "COMPETES_WITH",
}

_MENTIONS_TABLE_MAP: dict[EntityType, str] = {
    EntityType.company: "MENTIONS_COMPANY",
    EntityType.financial_metric: "MENTIONS_METRIC",
    EntityType.risk_factor: "MENTIONS_RISK",
    EntityType.business_segment: "MENTIONS_SEGMENT",
    EntityType.macro_event: "MENTIONS_EVENT",
}


class GraphPipeline:
    def __init__(
        self,
        llm: BaseChatModel,
        graph_config: Optional[GraphConfig] = None,
    ) -> None:
        self.llm = llm
        self.config = graph_config or GraphConfig()
        self.schema = GraphSchema(self.config)
        self.extractor = TripletExtractor(llm=llm)

    def process_chunks(self, chunks: List[Chunk]) -> List[FinancialTriplet]:
        table_prefix = _ensure_tables_string
        all_triplets: List[FinancialTriplet] = []

        conn = self.schema.connection

        for chunk in chunks:
            triplets = self.extractor.extract_from_chunk(
                chunk_text=chunk.text,
                ticker=chunk.company_ticker,
                year=chunk.fiscal_year,
                section_id=chunk.section_id,
                chunk_id=chunk.chunk_id,
            )

            self._upsert_chunk(conn, chunk)
            self._upsert_triplets(conn, chunk, triplets)
            all_triplets.extend(triplets)

        logger.info(
            "Graph pipeline: inserted %d triplets from %d chunks",
            len(all_triplets),
            len(chunks),
        )
        return all_triplets

    def _upsert_chunk(self, conn: kuzu.Connection, chunk: Chunk) -> None:
        conn.execute(
            """
            MERGE (c:DocumentChunk {chunk_id: $chunk_id})
            SET c.text = $text,
                c.company_ticker = $ticker,
                c.fiscal_year = $year,
                c.section_id = $section,
                c.page_number = $page
            """,
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "ticker": chunk.company_ticker,
                "year": chunk.fiscal_year,
                "section": chunk.section_id,
                "page": chunk.page_number,
            },
        )

    def _upsert_triplets(
        self,
        conn: kuzu.Connection,
        chunk: Chunk,
        triplets: List[FinancialTriplet],
    ) -> None:
        for ft in triplets:
            self._upsert_entity(conn, ft.triplet.source_entity)
            self._upsert_entity(conn, ft.triplet.target_entity)
            self._upsert_relationship(conn, ft)
            self._upsert_mentions(conn, chunk.chunk_id, ft)

    def _upsert_entity(
        self,
        conn: kuzu.Connection,
        entity: Entity,
    ) -> None:
        table = _NODE_TABLE_MAP.get(entity.entity_type)
        if table is None:
            logger.warning("Unknown entity type: %s", entity.entity_type)
            return

        pk_field = self._pk_field(table)
        props = dict(entity.properties or {})
        props["name"] = entity.name

        set_parts = [f"e.{k} = ${k}" for k in props]
        set_clause = "SET " + ", ".join(set_parts) if set_parts else ""

        query = (
            f"MERGE (e:{table} {{{pk_field}: $pk_val}}) {set_clause}"
        )
        params = {"pk_val": entity.name, **props}
        try:
            conn.execute(query, params)
        except RuntimeError as exc:
            logger.debug("Entity upsert skipped: %s", exc)

    def _upsert_relationship(
        self,
        conn: kuzu.Connection,
        ft: FinancialTriplet,
    ) -> None:
        rel_table = _REL_TABLE_MAP.get(ft.triplet.relation)
        if rel_table is None:
            return

        src_table = _NODE_TABLE_MAP.get(ft.triplet.source_entity.entity_type)
        dst_table = _NODE_TABLE_MAP.get(ft.triplet.target_entity.entity_type)
        if not src_table or not dst_table:
            return

        src_pk = self._pk_field(src_table)
        dst_pk = self._pk_field(dst_table)

        query = (
            f"MATCH (s:{src_table}), (t:{dst_table}) "
            f"WHERE s.{src_pk} = $src AND t.{dst_pk} = $dst "
            f"MERGE (s)-[:{rel_table}]->(t)"
        )
        try:
            conn.execute(
                query,
                {"src": ft.source_name, "dst": ft.target_name},
            )
        except RuntimeError as exc:
            logger.debug("Rel upsert skipped: %s", exc)

    def _upsert_mentions(
        self,
        conn: kuzu.Connection,
        chunk_id: str,
        ft: FinancialTriplet,
    ) -> None:
        rel_table = _MENTIONS_TABLE_MAP.get(
            ft.triplet.target_entity.entity_type
        )
        if rel_table is None:
            return

        dst_table = _NODE_TABLE_MAP.get(
            ft.triplet.target_entity.entity_type
        )
        if dst_table is None:
            return

        dst_pk = self._pk_field(dst_table)

        query = (
            f"MATCH (c:DocumentChunk), (e:{dst_table}) "
            f"WHERE c.chunk_id = $chunk_id AND e.{dst_pk} = $entity_name "
            f"MERGE (c)-[:{rel_table}]->(e)"
        )
        try:
            conn.execute(
                query,
                {"chunk_id": chunk_id, "entity_name": ft.target_name},
            )
        except RuntimeError as exc:
            logger.debug("Mentions rel skipped: %s", exc)

    def close(self) -> None:
        self.schema.close()

    @staticmethod
    def _pk_field(table: str) -> str:
        if table == "Company":
            return "ticker"
        return "id"
