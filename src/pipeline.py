from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from src.graph.communities import CommunityDetector, CommunitySummary
from src.graph.graph_pipeline import GraphPipeline
from src.graph.schema import GraphConfig
from src.ingestion.chunker import Chunk
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_COMPANIES_CONFIG = "data/companies.json"


def load_companies_config(
    config_path: str | Path = DEFAULT_COMPANIES_CONFIG,
) -> Dict[str, object]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Companies config not found at {config_path}. "
            "Create a data/companies.json with {'companies': [...], 'default_years': [...]}."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class FinancialGraphRAGPipeline:
    def __init__(
        self,
        llm: BaseChatModel,
        raw_data_dir: str | Path = "data/raw_10k",
        processed_dir: str | Path = "data/processed_chunks",
        graph_db_path: str | Path = "data/graph/kuzu_db",
        vector_db_uri: str | Path = "data/vector_store/lancedb",
        chunk_size: int = 600,
        chunk_overlap: int = 90,
    ) -> None:
        self.llm = llm
        self.ingestion = IngestionPipeline(
            raw_data_dir=raw_data_dir,
            processed_dir=processed_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        graph_config = GraphConfig(db_path=graph_db_path)
        self.graph = GraphPipeline(llm=llm, graph_config=graph_config)
        self.retrieval = RetrievalPipeline(
            llm=llm,
            graph_schema=self.graph.schema,
            dense_db_uri=vector_db_uri,
        )

    def ingest_and_index(
        self,
        ticker: str,
        year: int,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Chunk]:
        chunks = self.ingestion.run(
            ticker, year, after_date, before_date, use_cache=use_cache
        )
        if chunks:
            self.graph.process_chunks(chunks)
            chunk_dicts = [c.to_dict() for c in chunks]
            self.retrieval.index_chunks(chunk_dicts)
        return chunks

    def ingest_companies(
        self,
        config_path: str | Path = DEFAULT_COMPANIES_CONFIG,
        years: Optional[List[int]] = None,
        use_cache: bool = True,
        max_retries: int = 1,
    ) -> Dict[str, List[Chunk]]:
        config = load_companies_config(config_path)
        companies = config.get("companies", [])
        resolved_years = years or config.get("default_years", [])

        def _ingest(
            ticker: str, year: int
        ) -> List[Chunk]:
            return self.ingest_and_index(ticker, year, use_cache=use_cache)

        all_chunks: Dict[str, List[Chunk]] = {}
        failures: List[tuple[str, int]] = []

        for ticker in companies:
            for year in resolved_years:
                logger.info("Ingesting %s / %s", ticker, year)
                try:
                    chunks = _ingest(ticker, year)
                except Exception as exc:
                    logger.exception(
                        "Failed to ingest %s / %s: %s", ticker, year, exc
                    )
                    failures.append((ticker, year))
                    continue
                all_chunks.setdefault(ticker, []).extend(chunks)

        if failures:
            logger.warning(
                "%d company/year failed on first pass; retrying: %s",
                len(failures),
                failures,
            )
            for attempt in range(max_retries):
                still_failing: List[tuple[str, int]] = []
                for ticker, year in failures:
                    logger.info(
                        "Retrying %s / %s (attempt %d/%d)",
                        ticker,
                        year,
                        attempt + 1,
                        max_retries,
                    )
                    try:
                        chunks = _ingest(ticker, year)
                    except Exception as exc:
                        logger.exception(
                            "Retry failed for %s / %s: %s", ticker, year, exc
                        )
                        still_failing.append((ticker, year))
                        continue
                    all_chunks.setdefault(ticker, []).extend(chunks)
                failures = still_failing
                if not failures:
                    break

        if failures:
            logger.error(
                "Batch ingest finished with %d unresolved failures: %s",
                len(failures),
                failures,
            )

        logger.info(
            "Batch ingest complete: %d companies x %d years",
            len(companies),
            len(resolved_years),
        )
        return all_chunks

    def query(self, question: str) -> RetrievalResult:
        return self.retrieval.query(question)

    def detect_communities(
        self,
        max_summaries: int = 20,
    ) -> List[CommunitySummary]:
        detector = CommunityDetector(
            schema=self.graph.schema,
            llm=self.llm,
        )
        communities = detector.detect_communities()
        summaries = detector.generate_summaries(
            communities, max_communities=max_summaries
        )
        return summaries

    def close(self) -> None:
        self.graph.close()
