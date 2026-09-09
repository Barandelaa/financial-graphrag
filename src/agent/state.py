from __future__ import annotations

from typing import Annotated, List, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    question: str
    intent: str  # "ingest" | "retrieve"
    ticker: Optional[str]
    year: Optional[int]
    confidence: float
    expanded_query: str
    # retrieval (no se guarda en checkpoint para ahorrar VRAM)
    dense: List[dict]
    sparse: List[dict]
    graph: List[dict]
    graph_facts: List[str]
    metrics_rows: List[dict]
    metrics_block: str
    fused: List[dict]
    reranked: List[dict]
    answer: str
    citations: List[dict]
    # ingest HITL
    ingest_request: Optional[dict]
    ingest_result: Optional[str]
    # memoria conversacional: solo Q/A, no chunks/triplets (ver impacto en docs)
    messages: Annotated[List[dict], add_messages]
    # historial ligero para resolver "¿y en 2023?" sin re-preguntar ticker
    history_ticker: Optional[str]
    history_year: Optional[int]
