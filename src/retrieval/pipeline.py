from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from src.graph.schema import GraphSchema
from src.retrieval.dense import DenseRetriever
from src.retrieval.generator import GenerationResponse, ResponseGenerator
from src.retrieval.graph_traversal import GraphTraversalRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankedResult
from src.retrieval.rrf import ReciprocalRankFusion
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
        top_k_final: int = 10,
    ) -> None:
        self.llm = llm
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.top_k_graph = top_k_graph
        self.top_k_rrf = top_k_rrf
        self.top_k_final = top_k_final

        self.dense = DenseRetriever(
            db_uri=dense_db_uri,
            model_name=dense_model,
        )
        self.sparse = SparseRetriever()
        self.graph = GraphTraversalRetriever(schema=graph_schema)
        self.fusion = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)
        self.generator = ResponseGenerator(llm=llm)

    def index_chunks(self, chunks: List[dict]) -> None:
        self.dense.index_chunks(chunks)
        self.sparse.index_chunks(chunks)
        logger.info("Retrieval pipeline: indexed %d chunks", len(chunks))

    def query(self, question: str) -> RetrievalResult:
        dense_results = self.dense.search(question, top_k=self.top_k_dense)
        sparse_results = self.sparse.search(question, top_k=self.top_k_sparse)
        graph_results = self.graph.search(question, top_k=self.top_k_graph)

        logger.debug(
            "Retrieval counts — dense=%d sparse=%d graph=%d",
            len(dense_results),
            len(sparse_results),
            len(graph_results),
        )

        fused = self.fusion.fuse(
            dense=dense_results,
            sparse=sparse_results,
            graph=graph_results,
            top_k=self.top_k_rrf,
        )

        reranked = self.reranker.rerank(
            query=question,
            candidates=fused,
            top_k=self.top_k_final,
        )

        generation = self.generator.generate(
            question=question,
            context=reranked,
        )

        return RetrievalResult(
            question=question,
            answer=generation.answer,
            citations=generation.citations,
            dense_results=len(dense_results),
            sparse_results=len(sparse_results),
            graph_results=len(graph_results),
            final_context=reranked,
        )
