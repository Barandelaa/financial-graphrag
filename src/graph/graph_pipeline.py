from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import kuzu
from langchain_core.language_models import BaseChatModel

from src.graph.extractor import (
    Entity,
    EntityType,
    FinancialTriplet,
    RelationType,
    Triplet,
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
        triplets_cache_dir: str | Path = "data/processed_chunks",
    ) -> None:
        self.llm = llm
        self.config = graph_config or GraphConfig()
        self.schema = GraphSchema(self.config)
        self.extractor = TripletExtractor(llm=llm)
        self.triplets_cache_dir = Path(triplets_cache_dir)

    def process_chunks(self, chunks: List[Chunk]) -> List[FinancialTriplet]:
        all_triplets: List[FinancialTriplet] = []

        conn = self.schema.connection

        groups: Dict[tuple[str, int], List[Chunk]] = {}
        for chunk in chunks:
            groups.setdefault(
                (chunk.company_ticker, chunk.fiscal_year), []
            ).append(chunk)

        for (ticker, year), group_chunks in groups.items():
            cache = self._load_triplets_cache(ticker, year)
            new_entries: Dict[str, List[FinancialTriplet]] = {}

            for chunk in group_chunks:
                if chunk.chunk_id in cache:
                    triplets = cache[chunk.chunk_id]
                    logger.debug(
                        "Reusing %d cached triplets for chunk %s",
                        len(triplets),
                        chunk.chunk_id,
                    )
                else:
                    triplets = self.extractor.extract_from_chunk(
                        chunk_text=chunk.text,
                        ticker=chunk.company_ticker,
                        year=chunk.fiscal_year,
                        section_id=chunk.section_id,
                        chunk_id=chunk.chunk_id,
                    )
                    new_entries[chunk.chunk_id] = triplets

                self._upsert_chunk(conn, chunk)
                self._upsert_triplets(conn, chunk, triplets)
                all_triplets.extend(triplets)

            if new_entries:
                cache.update(new_entries)
                self._save_triplets_cache(ticker, year, cache)

            try:
                conn.execute("CHECKPOINT")
                logger.info("Graph pipeline: checkpointed Kùzu after %s / %s", ticker, year)
            except RuntimeError as exc:
                logger.debug("Kùzu checkpoint skipped: %s", exc)

        logger.info(
            "Graph pipeline: processed %d chunks for %d group(s)",
            len(chunks),
            len(groups),
        )
        return all_triplets

    def _cache_path(self, ticker: str, year: int) -> Path:
        return self.triplets_cache_dir / f"{ticker}_{year}" / "triplets.json"

    def _load_triplets_cache(
        self, ticker: str, year: int
    ) -> Dict[str, List[FinancialTriplet]]:
        path = self._cache_path(ticker, year)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception as exc:
            logger.warning(
                "Could not read triplets cache %s: %s", path, exc
            )
            return {}
        return {
            rec["chunk_id"]: self._triplets_from_dict(rec)
            for rec in records
        }

    def _save_triplets_cache(
        self,
        ticker: str,
        year: int,
        cache: Dict[str, List[FinancialTriplet]],
    ) -> None:
        path = self._cache_path(ticker, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            self._triplet_to_dict(chunk_id, triplets)
            for chunk_id, triplets in sorted(cache.items())
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(
            "Persisted %d chunk triplets to %s", len(records), path
        )

    @staticmethod
    def _triplet_to_dict(
        chunk_id: str,
        triplets: List[FinancialTriplet],
    ) -> dict:
        meta = triplets[0] if triplets else None
        return {
            "chunk_id": chunk_id,
            "company_ticker": getattr(meta, "company_ticker", ""),
            "fiscal_year": getattr(meta, "fiscal_year", 0),
            "section_id": getattr(meta, "section_id", ""),
            "triplets": [
                ft.triplet.model_dump(mode="json") for ft in triplets
            ],
        }

    @staticmethod
    def _triplets_from_dict(rec: dict) -> List[FinancialTriplet]:
        triplets: List[FinancialTriplet] = []
        for item in rec.get("triplets", []):
            try:
                triplets.append(
                    FinancialTriplet(
                        triplet=Triplet.model_validate(item),
                        chunk_id=rec["chunk_id"],
                        company_ticker=rec.get("company_ticker", ""),
                        fiscal_year=rec.get("fiscal_year", 0),
                        section_id=rec.get("section_id", ""),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Skipping invalid cached triplet for chunk %s: %s",
                    rec.get("chunk_id"),
                    exc,
                )
        return triplets

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
