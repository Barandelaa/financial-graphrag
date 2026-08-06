from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class SparseSearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class SparseRetriever:
    def __init__(self) -> None:
        self._corpus: List[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    def index_chunks(self, chunks: List[dict]) -> None:
        known_ids = {c["chunk_id"] for c in self._corpus}
        new_chunks = [
            c for c in chunks if c["chunk_id"] not in known_ids
        ]

        if new_chunks:
            self._corpus.extend(new_chunks)
            tokenized = [
                self._tokenize(c["text"]) for c in self._corpus
            ]
            self._bm25 = BM25Okapi(tokenized)
            logger.info(
                "BM25 added %d new chunks (total corpus: %d)",
                len(new_chunks),
                len(self._corpus),
            )
        else:
            logger.info("No new chunks for BM25; corpus unchanged")

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[SparseSearchResult]:
        if self._bm25 is None:
            logger.warning("BM25 index not built; call index_chunks first")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[SparseSearchResult] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self._corpus[idx]
            results.append(
                SparseSearchResult(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    score=float(score),
                    metadata={
                        "company_ticker": chunk.get("company_ticker", ""),
                        "fiscal_year": chunk.get("fiscal_year", ""),
                        "section_id": chunk.get("section_id", ""),
                    },
                )
            )
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return cleaned.split()
