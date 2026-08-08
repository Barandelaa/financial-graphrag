from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import CrossEncoder

from src.retrieval.rrf import FusedResult

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _detect_device(preferred: str) -> str:
    if preferred and preferred != "cpu":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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
        self._device = _detect_device(device)

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from src.env import get_hf_token

            kwargs = {"device": self._device}
            token = get_hf_token()
            if token:
                kwargs["token"] = token
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[FusedResult],
        top_k: int = 10,
        diversity_lambda: float = 0.7,
        min_score: float = 0.2,
    ) -> List[RerankedResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        scored = [
            (c, _sigmoid(float(scores[i])))
            for i, c in enumerate(candidates)
        ]

        if min_score > 0.0:
            scored = [
                (c, s) for c, s in scored if s >= min_score
            ]

        scored.sort(key=lambda x: x[1], reverse=True)

        if len(scored) <= top_k:
            return [self._to_result(c, s) for c, s in scored]

        token_sets = [_tokenize(c.text) for c, _ in scored]
        remaining_idx = list(range(len(scored)))
        selected_idx: List[int] = []

        while len(selected_idx) < top_k:
            best_pos = 0
            best_value = float("-inf")
            for pos, i in enumerate(remaining_idx):
                value = diversity_lambda * scored[i][1]
                if selected_idx:
                    max_sim = 0.0
                    for j in selected_idx:
                        sim = _jaccard(token_sets[i], token_sets[j])
                        if sim > max_sim:
                            max_sim = sim
                    value -= (1.0 - diversity_lambda) * max_sim
                if value > best_value:
                    best_value = value
                    best_pos = pos
            selected_idx.append(remaining_idx.pop(best_pos))

        return [self._to_result(*scored[i]) for i in selected_idx]

    @staticmethod
    def _to_result(candidate: FusedResult, score: float) -> RerankedResult:
        return RerankedResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=score,
            metadata=candidate.metadata,
        )
