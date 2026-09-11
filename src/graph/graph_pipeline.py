from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
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

# Alias de compañía (minúsculas) -> ticker canónico. Mantiene la tabla
# Company limpia: 'Apple'/'Microsoft'/typos OCR ('AM,ZN','MS,FT','NV. DA')
# colapsan al ticker en vez de crear nodos fantasma.
_COMPANY_ALIAS_TO_TICKER: dict[str, str] = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "amazon inc": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "berkshire hathaway": "BRK.B",
    "berkshire": "BRK.B",
}

_VALID_TICKERS = frozenset(
    {"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "BRK.B"}
)


def _known_tickers() -> set[str]:
    """Tickers base + dados de alta en companies.json (dinámico, sin curado)."""
    tickers = set(_VALID_TICKERS)
    try:
        from src.agent.company_registry import get_config_tickers

        tickers |= {t.upper() for t in get_config_tickers()}
    except Exception:
        pass
    return tickers

# Nombres genéricos que el extractor cuela como Company y deben descartarse.
_NOISY_COMPANY_EXACT = frozenset(
    {
        "services",
        "products",
        "research and development",
        "selling, general and administrative",
        "businesssegment",
        "riskfactor",
        "financialmetric",
        "company",
        "macroevent",
        "other companies",
        "others",
        "unspecified competitors",
        "unspecified companies",
        "unspecified",
        "our competitors",
        "our peers",
        "competitors",
        "competitors in technology sector",
        "competitors developing ai technologies",
        "company x",
        "none",
        "n/a",
        "various competitors",
        "other bet companies",
        "other housing companies",
        "major automotive companies",
        "mature and prosperous companies",
        "start-ups and emerging companies",
        # Segmentos colados como Company por el extractor
        "data center",
        "graphics",
        "compute & networking",
        "building products group",
        "consumer products group",
        "industrial products group",
        "industrial products",
        "railroad",
    }
)

_NOISY_COMPANY_PREFIXES = (
    "companies that",
    "companies with",
    "other companies",
    "other freight",
    "other participants",
    "third-party ",
    "unspecified ",
    "unknown ",
    "platform-based ecosystem competitors",
    "manufacturers and suppliers",
    "national manufacturers",
    "smaller regional manufacturers",
    "large global manufacturers",
)


def _normalize_company_name(raw: str) -> Optional[str]:
    """Devuelve el ticker canónico o el nombre subsidiario, o None si vacío."""
    s = (raw or "").strip()
    if not s:
        return None
    alias = _COMPANY_ALIAS_TO_TICKER.get(s.lower())
    if alias:
        return alias
    upper = s.upper()
    if upper in _known_tickers():
        return upper
    # Variantes con sufijos legales: 'Amazon, Inc.' -> 'AMZN', 'Microsoft
    # Corporation, or Microsoft' -> 'MSFT'. Substring insensible a puntuación.
    cleaned = re.sub(r"[^a-z]", "", s.lower())
    for key, ticker in _COMPANY_ALIAS_TO_TICKER.items():
        if re.sub(r"[^a-z]", "", key) and re.sub(r"[^a-z]", "", key) in cleaned:
            return ticker
    # Typos OCR: 'AM,ZN' -> 'AMZN', 'MS,FT' -> 'MSFT', 'NV. DA' -> 'NVDA',
    # 'BR. B'/'BRK, B'/'BRK3.B'/'BR.0' -> 'BRK.B'.
    letters = re.sub(r"[^A-Z]", "", upper)
    if letters in {"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA"}:
        return letters
    if letters in {"BRKB", "BRB", "BRO", "BR", "BRK"}:
        return "BRK.B"
    return s


def _is_noisy_company(name: str) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return True
    if lowered in _NOISY_COMPANY_EXACT:
        return True
    if lowered.startswith(_NOISY_COMPANY_PREFIXES):
        return True
    if len(name.strip()) > 60:
        return True
    return False


def _metric_composite_id(ticker: str, year: int, metric_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", metric_name.strip().lower()).strip("_")[:80]
    if not slug:
        slug = "metric"
    return f"{ticker.upper()}_{year}_{slug}"


def _metric_year_from_entity(entity: Entity, fallback_year: Optional[int]) -> Optional[int]:
    for key in ("year", "fiscal_year"):
        if key in (entity.properties or {}):
            try:
                y = int(str(entity.properties[key]).strip()[:4])
                if 2015 <= y <= 2030:
                    return y
            except Exception:
                pass
    return fallback_year


def _entity_pk_value(entity: Entity, ticker: Optional[str] = None, year: Optional[int] = None) -> Optional[str]:
    """PK normalizada para un Entity, o None si debe descartarse.
    Para FinancialMetric genera id compuesto ticker_year_metric para evitar colisión entre empresas.
    Usa year de properties si el extractor lo dio (año de la cifra), si no fallback al año del filing.
    """
    raw = (entity.name or "").strip()
    if not raw:
        return None
    if entity.entity_type == EntityType.company:
        norm = _normalize_company_name(raw)
        if norm is None or _is_noisy_company(norm):
            return None
        return norm
    if entity.entity_type == EntityType.financial_metric:
        # Si tenemos contexto ticker/year, usa PK compuesto con año de la métrica si existe
        if ticker:
            eff_year = _metric_year_from_entity(entity, year)
            if eff_year:
                return _metric_composite_id(ticker, eff_year, raw)
        # Fallback legacy (solo nombre) — para compatibilidad con datos viejos sin contexto
        if len(raw) > 300:
            return None
        return raw
    # Tickers/alias colados en otras tablas (p.ej. MacroEvent 'AAPL' o
    # BusinessSegment 'Apple') se descartan: pertenecen a Company.
    if _normalize_company_name(raw) in _known_tickers():
        return None
    # Resto de tipos: recorta y descarta vacíos o absurdamente largos.
    if len(raw) > 300:
        return None
    return raw


class GraphPipeline:
    def __init__(
        self,
        llm: BaseChatModel,
        graph_config: Optional[GraphConfig] = None,
        triplets_cache_dir: str | Path = "data/processed_chunks",
        max_workers: int = 4,
        batch_size: int = 1,
        save_every: int = 1,
    ) -> None:
        self.llm = llm
        self.config = graph_config or GraphConfig()
        self.schema = GraphSchema(self.config)
        self.extractor = TripletExtractor(llm=llm)
        self.triplets_cache_dir = Path(triplets_cache_dir)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.save_every = max(1, save_every)

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
            pending: List[Chunk] = []
            # 1) upsert inmediatemente los cacheados (rápido, sin LLM)
            for chunk in group_chunks:
                if chunk.chunk_id in cache:
                    triplets = cache[chunk.chunk_id]
                    logger.debug(
                        "Reusing %d cached triplets for chunk %s",
                        len(triplets),
                        chunk.chunk_id,
                    )
                    self._upsert_chunk(conn, chunk)
                    self._upsert_triplets(conn, chunk, triplets)
                    all_triplets.extend(triplets)
                else:
                    pending.append(chunk)

            if not pending:
                logger.info("Graph pipeline: all %d chunks cached for %s / %s", len(group_chunks), ticker, year)
                try:
                    conn.execute("CHECKPOINT")
                except RuntimeError as exc:
                    logger.debug("Kùzu checkpoint skipped: %s", exc)
                continue

            logger.info(
                "Graph pipeline: extracting %d pending chunks for %s / %s (batch=%d, workers=%d)",
                len(pending),
                ticker,
                year,
                self.batch_size,
                self.max_workers,
            )
            t0 = time.time()
            # 2) Agrupa pendientes en lotes
            batches: List[List[Chunk]] = [
                pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)
            ]
            new_entries: Dict[str, List[FinancialTriplet]] = {}
            completed_batches = 0
            failed_chunks = 0

            # Mapa chunk_id -> Chunk para upsert ordenado
            chunk_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in pending}

            if self.max_workers > 1 and len(batches) > 1:
                # Paralelo: cada worker procesa un batch (varias llamadas LLM en paralelo)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {}
                    for batch in batches:
                        # Convierte batch a formato dict para extractor
                        batch_dicts = [
                            {"chunk_id": c.chunk_id, "text": c.text, "section_id": c.section_id}
                            for c in batch
                        ]
                        fut = executor.submit(self.extractor.extract_from_batch, batch_dicts, ticker, year)
                        future_to_batch[fut] = batch

                    for fut in concurrent.futures.as_completed(future_to_batch):
                        batch = future_to_batch[fut]
                        try:
                            result_map = fut.result()
                        except Exception as exc:
                            logger.error("Batch failed unexpectedly (%d chunks): %s", len(batch), exc)
                            result_map = {}
                            for c in batch:
                                # fallback individual ya está dentro de extract_from_batch, pero por si acaso
                                result_map[c.chunk_id] = []

                        # Upsert serializado (Kùzu no es thread-safe)
                        for chunk in batch:
                            cid = chunk.chunk_id
                            triplets = result_map.get(cid, [])
                            # Si batch devolvió vacío pero chunk no era skippable, cuenta como posible fallo
                            if not triplets and cid not in result_map:
                                failed_chunks += 1
                            self._upsert_chunk(conn, chunk)
                            self._upsert_triplets(conn, chunk, triplets)
                            all_triplets.extend(triplets)
                            new_entries[cid] = triplets
                            # si no estaba en result_map y era skippable, extractor ya lo maneja

                        completed_batches += 1
                        # Guardado incremental cada save_every batches + siempre al final
                        if completed_batches % self.save_every == 0:
                            cache.update(new_entries)
                            self._save_triplets_cache(ticker, year, cache)
                            # no limpiar new_entries para no perder referencia; se sigue acumulando
                            logger.info(
                                "Progress %s/%s: %d/%d batches (%d chunks) — elapsed %.1fs",
                                ticker,
                                year,
                                completed_batches,
                                len(batches),
                                completed_batches * self.batch_size,
                                time.time() - t0,
                            )
            else:
                # Secuencial (batch_size puede ser >1 aun sin workers)
                for batch in batches:
                    batch_dicts = [
                        {"chunk_id": c.chunk_id, "text": c.text, "section_id": c.section_id}
                        for c in batch
                    ]
                    result_map = self.extractor.extract_from_batch(batch_dicts, ticker, year)
                    for chunk in batch:
                        cid = chunk.chunk_id
                        triplets = result_map.get(cid, [])
                        self._upsert_chunk(conn, chunk)
                        self._upsert_triplets(conn, chunk, triplets)
                        all_triplets.extend(triplets)
                        new_entries[cid] = triplets
                    completed_batches += 1
                    if completed_batches % self.save_every == 0:
                        cache.update(new_entries)
                        self._save_triplets_cache(ticker, year, cache)

            # Guardado final del grupo
            if new_entries:
                cache.update(new_entries)
                self._save_triplets_cache(ticker, year, cache)

            elapsed = time.time() - t0
            avg_ms = (elapsed / max(1, len(pending))) * 1000
            logger.info(
                "Graph pipeline: extracted %d chunks for %s/%s in %.1fs (avg %.0f ms/chunk, failed~%d)",
                len(pending),
                ticker,
                year,
                elapsed,
                avg_ms,
                failed_chunks,
            )

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
        # Escritura atómica: escribe a tmp y renombra
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
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
        seen: set[tuple[str, str, str, str, str]] = set()
        for ft in triplets:
            # Para FinancialMetric usa PK compuesto ticker_year_metric
            ticker = ft.company_ticker
            year = ft.fiscal_year
            src_val = _entity_pk_value(ft.triplet.source_entity, ticker, year)
            dst_val = _entity_pk_value(ft.triplet.target_entity, ticker, year)
            if src_val is None or dst_val is None:
                continue
            if src_val.lower() == dst_val.lower():
                continue
            key = (
                ft.triplet.source_entity.entity_type.value,
                src_val.lower(),
                ft.triplet.relation.value,
                ft.triplet.target_entity.entity_type.value,
                dst_val.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            self._upsert_entity(conn, ft.triplet.source_entity, ticker, year)
            self._upsert_entity(conn, ft.triplet.target_entity, ticker, year)
            self._upsert_relationship(conn, ft)
            self._upsert_mentions(conn, chunk.chunk_id, ft)

    def _upsert_entity(
        self,
        conn: kuzu.Connection,
        entity: Entity,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
    ) -> None:
        table = _NODE_TABLE_MAP.get(entity.entity_type)
        if table is None:
            logger.warning("Unknown entity type: %s", entity.entity_type)
            return

        pk_val = _entity_pk_value(entity, ticker, year)
        if pk_val is None:
            logger.debug("Skipping noisy/empty entity: %r", entity.name)
            return

        pk_field = self._pk_field(table)
        props = dict(entity.properties or {})
        # Para FinancialMetric el PK es compuesto pero name debe ser el nombre legible
        if entity.entity_type == EntityType.financial_metric:
            props["name"] = (entity.name or "").strip()
            # Usa año de la métrica si el extractor lo dio, si no filing year
            eff_year = _metric_year_from_entity(entity, year)
            if eff_year is not None:
                props["fiscal_year"] = eff_year
            # Limpia year auxiliar, Kuzu espera fiscal_year
            props.pop("year", None)
            # value/unit ya vienen en props desde el extractor; respeta si existen
        else:
            props["name"] = pk_val

        set_parts = [f"e.{k} = ${k}" for k in props]
        set_clause = "SET " + ", ".join(set_parts) if set_parts else ""

        query = (
            f"MERGE (e:{table} {{{pk_field}: $pk_val}}) {set_clause}"
        )
        params = {"pk_val": pk_val, **props}
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

        ticker = ft.company_ticker
        year = ft.fiscal_year
        src_val = _entity_pk_value(ft.triplet.source_entity, ticker, year)
        dst_val = _entity_pk_value(ft.triplet.target_entity, ticker, year)
        if src_val is None or dst_val is None:
            return

        query = (
            f"MATCH (s:{src_table}), (t:{dst_table}) "
            f"WHERE s.{src_pk} = $src AND t.{dst_pk} = $dst "
            f"MERGE (s)-[:{rel_table}]->(t)"
        )
        try:
            conn.execute(
                query,
                {"src": src_val, "dst": dst_val},
            )
        except RuntimeError as exc:
            logger.debug("Rel upsert skipped: %s", exc)

    def _upsert_mentions(
        self,
        conn: kuzu.Connection,
        chunk_id: str,
        ft: FinancialTriplet,
    ) -> None:
        # Indexa tanto source como target para que las preguntas por
        # compañía/evento encuentren sus chunks (antes solo el target,
        # lo que dejaba MENTIONS_COMPANY casi vacío y MENTIONS_EVENT en 0).
        for entity in (
            ft.triplet.source_entity,
            ft.triplet.target_entity,
        ):
            self._upsert_single_mention(conn, chunk_id, entity, ft.company_ticker, ft.fiscal_year)

    def _upsert_single_mention(
        self,
        conn: kuzu.Connection,
        chunk_id: str,
        entity: Entity,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
    ) -> None:
        rel_table = _MENTIONS_TABLE_MAP.get(entity.entity_type)
        if rel_table is None:
            return

        dst_table = _NODE_TABLE_MAP.get(entity.entity_type)
        if dst_table is None:
            return

        entity_name = _entity_pk_value(entity, ticker, year)
        if entity_name is None:
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
                {"chunk_id": chunk_id, "entity_name": entity_name},
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
