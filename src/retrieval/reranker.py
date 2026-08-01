from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.retrieval.rrf import FusedResult

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        self._device = device

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name, device=self._device
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[FusedResult],
        top_k: int = 10,
    ) -> List[RerankedResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        scored = [
            (c, float(scores[i]))
            for i, c in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RerankedResult(
                chunk_id=c.chunk_id,
                text=c.text,
                score=s,
                metadata=c.metadata,
            )
            for c, s in scored[:top_k]
        ]
