from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class FusedResult:
    chunk_id: str
    text: str
    rrf_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    graph_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class ReciprocalRankFusion:
    def __init__(self, k: int = RRF_K) -> None:
        self.k = k

    def fuse(
        self,
        dense: List,
        sparse: List,
        graph: List,
        top_k: int = 50,
    ) -> List[FusedResult]:
        scores: Dict[str, dict] = {}

        self._accumulate(scores, dense, "dense_score")
        self._accumulate(scores, sparse, "sparse_score")
        self._accumulate(scores, graph, "graph_score")

        fused = [
            FusedResult(
                chunk_id=cid,
                text=v["text"],
                rrf_score=v["rrf"],
                dense_score=v.get("dense_score"),
                sparse_score=v.get("sparse_score"),
                graph_score=v.get("graph_score"),
                metadata=v.get("metadata", {}),
            )
            for cid, v in scores.items()
        ]

        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused[:top_k]

    def _accumulate(
        self,
        scores: Dict[str, dict],
        results: List,
        score_key: str,
    ) -> None:
        for rank, r in enumerate(results):
            cid = r.chunk_id
            if cid not in scores:
                scores[cid] = {
                    "text": r.text,
                    "rrf": 0.0,
                    "metadata": r.metadata,
                }
            scores[cid][score_key] = r.score
            scores[cid]["rrf"] += 1.0 / (self.k + rank + 1)
