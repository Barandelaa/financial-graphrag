from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class DenseSearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class DenseRetriever:
    def __init__(
        self,
        db_uri: str | Path = "data/vector_store/lancedb",
        model_name: str = "BAAI/bge-m3",
        table_name: str = "chunks",
        device: str = "cpu",
    ) -> None:
        self.db_uri = str(db_uri)
        self.model_name = model_name
        self.table_name = table_name
        self._embedder: Optional[SentenceTransformer] = None
        self._device = device

        Path(db_uri).parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.db_uri)

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.model_name, device=self._device
            )
        return self._embedder

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    def index_chunks(
        self,
        chunks: List[dict],
        persist: bool = True,
    ) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        records = []
        for chunk, emb in zip(chunks, embeddings):
            records.append({
                "vector": emb.tolist(),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "company_ticker": chunk["company_ticker"],
                "fiscal_year": chunk["fiscal_year"],
                "section_id": chunk["section_id"],
                "page_number": chunk["page_number"],
                **chunk.get("metadata", {}),
            })

        table = self._db.create_table(
            self.table_name, data=records, mode="overwrite"
        )
        logger.info(
            "Indexed %d chunks in LanceDB table '%s'", len(records), self.table_name
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[DenseSearchResult]:
        query_vector = self.embed_query(query)

        try:
            table = self._db.open_table(self.table_name)
        except Exception as exc:
            logger.error("Cannot open table '%s': %s", self.table_name, exc)
            return []

        results = (
            table.search(query_vector.tolist())
            .limit(top_k)
            .to_list()
        )

        return [
            DenseSearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r.get("_distance", 0.0),
                metadata={
                    "company_ticker": r.get("company_ticker", ""),
                    "fiscal_year": r.get("fiscal_year", ""),
                    "section_id": r.get("section_id", ""),
                    "page_number": r.get("page_number", ""),
                },
            )
            for r in results
        ]

    def close(self) -> None:
        self._db.close()
