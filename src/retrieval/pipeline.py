from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from src.graph.schema import GraphSchema
from src.retrieval.dense import DenseRetriever
from src.retrieval.generator import GenerationResponse, ResponseGenerator
from src.retrieval.graph_facts import GraphFactRetriever
from src.retrieval.graph_traversal import GraphTraversalRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankedResult
from src.retrieval.rrf import FusedResult, ReciprocalRankFusion
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    question: str
    answer: str
    citations: List[dict] = field(default_factory=list)
    dense_results: int = 0
    sparse_results: int = 0
    graph_results: int = 0
    graph_facts: List[str] = field(default_factory=list)
    final_context: List[RerankedResult] = field(default_factory=list)


class RetrievalPipeline:
    def __init__(
        self,
        llm: BaseChatModel,
        graph_schema: GraphSchema,
        dense_db_uri: str | Path = "data/vector_store/lancedb",
        dense_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_k: int = 60,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        top_k_graph: int = 20,
        top_k_rrf: int = 50,
        top_k_final: int = 5,
        max_facts: int = 20,
        rerank_diversity: float = 0.7,
        rerank_min_score: float = 0.2,
    ) -> None:
        self.llm = llm
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_graph = top_k_graph
        self.top_k_rrf = top_k_rrf
        self.top_k_final = top_k_final
        self.max_facts = max_facts
        self.rerank_diversity = rerank_diversity
        self.rerank_min_score = rerank_min_score

        self.dense = DenseRetriever(
            db_uri=dense_db_uri,
            model_name=dense_model,
        )
        self.sparse = SparseRetriever()
        self._rebuild_sparse_from_lance()
        self.graph = GraphTraversalRetriever(schema=graph_schema)
        self.graph_facts_retriever = GraphFactRetriever(
            schema=graph_schema, traversal=self.graph
        )
        self.fusion = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.generator = ResponseGenerator(llm=llm)

    def index_chunks(self, chunks: List[dict]) -> None:
        self.dense.index_chunks(chunks)
        self.sparse.index_chunks(chunks)
        logger.info("Retrieval pipeline: indexed %d chunks", len(chunks))

    def _rebuild_sparse_from_lance(self) -> None:
        try:
            records = self.dense.load_all_chunks()
        except Exception as exc:
            logger.warning("Could not rebuild sparse index: %s", exc)
            return
        if not records:
            logger.info("No chunks in LanceDB to rebuild the sparse index")
            return
        self.sparse.index_chunks(records)
        logger.info(
            "Rebuilt BM25 sparse index from LanceDB: %d chunks",
            len(records),
        )

    def query(self, question: str) -> RetrievalResult:
        dense_results = self.dense.search(question, top_k=self.top_k_dense)
        sparse_results = self.sparse.search(question, top_k=self.top_k_sparse)
        graph_results = self.graph.search(question, top_k=self.top_k_graph)
        graph_facts = self.graph_facts_retriever.search(
            question, top_k=self.max_facts
        )

        logger.debug(
            "Retrieval counts — dense=%d sparse=%d graph=%d facts=%d",
            len(dense_results),
            len(sparse_results),
            len(graph_results),
            len(graph_facts),
        )

        fused = self.fusion.fuse(
            dense=dense_results,
            sparse=sparse_results,
            graph=graph_results,
            top_k=self.top_k_rrf,
        )

        fused = self._ground_to_query(fused, question)

        candidates = self._dedupe_candidates(fused)

        reranked = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=self.top_k_final,
            diversity_lambda=self.rerank_diversity,
            min_score=self.rerank_min_score,
        )

        final_context = reranked

        generation = self.generator.generate(
            question=question,
            context=final_context,
            graph_facts=graph_facts,
        )

        return RetrievalResult(
            question=question,
            answer=generation.answer,
            citations=generation.citations,
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
            graph_results=len(graph_results),
            graph_facts=graph_facts,
            final_context=final_context,
        )

    def _query_tickers(self, question: str) -> set[str]:
        try:
            entities = self.graph.match_query_entities(question)
        except Exception:
            return set()
        return {
            name for name, entity_type in entities if entity_type == "Company"
        }

    def _ground_to_query(
        self,
        candidates: List[FusedResult],
        question: str,
    ) -> List[FusedResult]:
        tickers = self._query_tickers(question)
        if not tickers:
            return candidates
        return [
            c for c in candidates
            if c.metadata.get("company_ticker", "") in tickers
        ]

    @staticmethod
    def _dedupe_candidates(
        candidates: List[FusedResult],
    ) -> List[FusedResult]:
        seen: set[tuple[str, str]] = set()
        deduped: List[FusedResult] = []
        for c in candidates:
            key = (
                c.metadata.get("company_ticker", ""),
                c.metadata.get("section_id", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped
