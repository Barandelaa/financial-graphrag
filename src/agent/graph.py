from __future__ import annotations

import logging
from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.nodes import (
    classify_intent_node,
    fuse_rerank_node,
    generate_node,
    parallel_retrieve_node,
)
from src.agent.state import AgentState
from src.agent.tools import _make_ingest_tool
from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)


def build_agent_graph(pipeline: FinancialGraphRAGPipeline):
    """Grafo determinista single-turn: classify → (ingest con HITL | retrieve→fuse→rerank→generate)."""

    ingest_tool = _make_ingest_tool(pipeline)

    # Wrapper nodos con closure pipeline/llm
    def classify(state: AgentState) -> Dict:
        return classify_intent_node(pipeline.llm, state)

    def parallel_retrieve(state: AgentState) -> Dict:
        return parallel_retrieve_node(pipeline, state)

    def fuse_rerank(state: AgentState) -> Dict:
        return fuse_rerank_node(pipeline, state)

    def generate(state: AgentState) -> Dict:
        return generate_node(pipeline, state)

    def route_intent(state: AgentState) -> str:
        return state.get("intent", "retrieve")

    # HITL: el tool node se pausa antes de ejecutar
    tool_node = ToolNode([ingest_tool])

    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify)
    g.add_node("parallel_retrieve", parallel_retrieve)
    g.add_node("fuse_rerank", fuse_rerank)
    g.add_node("generate", generate)
    g.add_node("ingest_tool", tool_node)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_intent, {"ingest": "ingest_tool", "retrieve": "parallel_retrieve"})
    g.add_edge("parallel_retrieve", "fuse_rerank")
    g.add_edge("fuse_rerank", "generate")
    g.add_edge("generate", END)
    g.add_edge("ingest_tool", END)

    checkpointer = MemorySaver()
    app = g.compile(checkpointer=checkpointer, interrupt_before=["ingest_tool"])
    # Para CLI sin API, también expone versión sin interrupt para uso directo si se pre-confirma
    app_no_interrupt = g.compile(checkpointer=checkpointer)
    app_no_interrupt.ingest_tool = ingest_tool  # útil para tests
    return app


def build_agent_graph_no_interrupt(pipeline: FinancialGraphRAGPipeline):
    """Variante sin HITL para tests/batch (ejecuta ingest directo)."""
    from langgraph.graph import StateGraph

    ingest_tool = _make_ingest_tool(pipeline)

    def classify(state: AgentState) -> Dict:
        from src.agent.nodes import classify_intent_node
        return classify_intent_node(pipeline.llm, state)

    def parallel_retrieve(state: AgentState) -> Dict:
        from src.agent.nodes import parallel_retrieve_node
        return parallel_retrieve_node(pipeline, state)

    def fuse_rerank(state: AgentState) -> Dict:
        from src.agent.nodes import fuse_rerank_node
        return fuse_rerank_node(pipeline, state)

    def generate(state: AgentState) -> Dict:
        from src.agent.nodes import generate_node
        return generate_node(pipeline, state)

    def route_intent(state: AgentState) -> str:
        return state.get("intent", "retrieve")

    tool_node = ToolNode([ingest_tool])
    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify)
    g.add_node("parallel_retrieve", parallel_retrieve)
    g.add_node("fuse_rerank", fuse_rerank)
    g.add_node("generate", generate)
    g.add_node("ingest_tool", tool_node)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_intent, {"ingest": "ingest_tool", "retrieve": "parallel_retrieve"})
    g.add_edge("parallel_retrieve", "fuse_rerank")
    g.add_edge("fuse_rerank", "generate")
    g.add_edge("generate", END)
    g.add_edge("ingest_tool", END)
    return g.compile(checkpointer=MemorySaver())
