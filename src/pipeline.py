from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from src.graph.communities import CommunityDetector, CommunitySummary
from src.graph.graph_pipeline import GraphPipeline
from src.graph.schema import GraphConfig
from src.ingestion.chunker import Chunk
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline, RetrievalResult

logger = logging.getLogger(__name__)


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
    ) -> List[Chunk]:
        chunks = self.ingestion.run(ticker, year, after_date, before_date)
        if chunks:
            self.graph.process_chunks(chunks)
            chunk_dicts = [c.to_dict() for c in chunks]
            self.retrieval.index_chunks(chunk_dicts)
        return chunks

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
