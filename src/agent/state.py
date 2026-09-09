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
    # retrieval
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
    # para langgraph messages (no usado en single-turn pero útil para debug)
    messages: Annotated[List[dict], add_messages]
