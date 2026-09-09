from __future__ import annotations

import gc
import logging
import re
from enum import Enum
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agent.tools import extract_ticker_year_fallback

logger = logging.getLogger(__name__)


class IntentEnum(str, Enum):
    ingest = "ingest"
    retrieve = "retrieve"


class IntentOutput(BaseModel):
    intent: IntentEnum = Field(description="ingest solo si usuario PIDE explícitamente añadir/ingerir/ingestar/nuevo ticker/año; retrieve en caso contrario incluso si menciona ticker")
    ticker: str | None = Field(default=None, description="Ticker normalizado AAPL|MSFT|AMZN|GOOGL|NVDA|META|TSLA|BRK.B si intent=ingest")
    year: int | None = Field(default=None, description="Año 2020-2026 si intent=ingest")
    confidence: float = Field(description="0-1 confianza de la clasificación")


INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "/no_think Clasifica la intención del usuario para un sistema RAG financiero local.\n"
     "Responde SOLO JSON con {{\"intent\": \"ingest\"|\"retrieve\", \"ticker\": \"...\", \"year\": 2024, \"confidence\": 0.9}}\n"
     "Reglas:\n"
     "- intent=ingest SOLO si el usuario pide explícitamente AÑADIR/INGERIR/INGESTAR/AGREGAR un ticker/año nuevo (ej: 'añade AAPL 2026', 'ingesta MSFT 2025', 'agrega BRK.B 2024').\n"
     "- Si el usuario dice 'No quiero que añadas...', 'solo dame su revenue', 'cuál es el revenue de AAPL' → intent=retrieve (pregunta, no ingesta).\n"
     "- ticker normalizado: AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, BRK.B (facebook→META, google→GOOGL).\n"
     "- year debe existir explícitamente en el texto para ingest; si no, null.\n"
     "- confidence alta solo si intención es clara."),
    ("human", "/no_think Texto: {question}"),
])


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def classify_intent_node(llm, state: dict) -> dict:
    question = state.get("question", "")
    # LLM gating 100% local
    try:
        structured = llm.with_structured_output(IntentOutput, method="json_mode")
        msgs = INTENT_PROMPT.format_messages(question=question)
        # intenta pasar reasoning=False para qwen3
        try:
            resp = structured.invoke(msgs, reasoning=False)  # type: ignore
        except TypeError:
            resp = structured.invoke(msgs)
        if isinstance(resp, IntentOutput):
            out = resp
        elif isinstance(resp, dict):
            out = IntentOutput.model_validate(resp)
        else:
            # AIMessage con content JSON
            content = getattr(resp, "content", str(resp))
            content = _strip_thinking(str(content))
            import json, re as _re
            m = _re.search(r"\{.*\}", content, _re.DOTALL)
            j = json.loads(m.group(0) if m else content)
            out = IntentOutput.model_validate(j)
        intent = out.intent.value if isinstance(out.intent, Enum) else str(out.intent)
        ticker = (out.ticker or "").strip().upper().replace("BRKB", "BRK.B") if out.ticker else None
        year = int(out.year) if out.year else None
        conf = float(out.confidence) if out.confidence else 0.0
        # fallback regex si LLM no dio ticker/year para ingest
        if intent == "ingest" and (not ticker or not year):
            fb_t, fb_y = extract_ticker_year_fallback(question)
            ticker = ticker or fb_t
            year = year or fb_y
        # umbral confianza
        if conf < 0.55:
            intent = "retrieve"
        return {"intent": intent, "ticker": ticker, "year": year, "confidence": conf}
    except Exception as exc:
        logger.debug("classify_intent fallback to regex: %s", exc)
        # Edge case: LLM falla → regex conservador: solo ingest si verbo explícito
        low = question.lower()
        has_ingest_verb = any(v in low for v in ["añade", "añadir", "ingesta", "ingestar", "agrega", "agregar", "add ", "ingest"])
        has_negation = "no quiero que añadas" in low or "no añadas" in low or "no agregues" in low
        if has_ingest_verb and not has_negation:
            t, y = extract_ticker_year_fallback(question)
            if t and y:
                return {"intent": "ingest", "ticker": t, "year": y, "confidence": 0.6}
        return {"intent": "retrieve", "ticker": None, "year": None, "confidence": 0.5}


def extract_entities_node(pipeline, state: dict) -> dict:
    # pipeline es FinancialGraphRAGPipeline -> retrieval está en pipeline.retrieval
    rp = getattr(pipeline, "retrieval", pipeline)
    tickers = rp._query_tickers(state.get("question", ""))
    # tickers set → list para estado serializable
    return {"tickers": list(tickers)}


def parallel_retrieve_node(pipeline, state: dict) -> dict:
    rp = getattr(pipeline, "retrieval", pipeline)
    question = state.get("question", "")
    expanded = rp._expand_query(question)
    dense = rp.dense.search(expanded, top_k=rp.top_k_dense)
    sparse = rp.sparse.search(expanded, top_k=rp.top_k_sparse)
    graph = rp.graph.search(expanded, top_k=rp.top_k_graph)
    facts = rp.graph_facts_retriever.search(question, top_k=rp.max_facts)
    metrics_rows = rp.metrics_retriever.search(question)
    metrics_block = rp.metrics_retriever.format_as_block(metrics_rows)
    # serializa para estado (dict)
    return {
        "expanded_query": expanded,
        "dense": [r.__dict__ for r in dense],
        "sparse": [r.__dict__ for r in sparse],
        "graph": [r.__dict__ for r in graph],
        "graph_facts": facts,
        "metrics_rows": [r.__dict__ for r in metrics_rows],
        "metrics_block": metrics_block,
    }


def fuse_rerank_node(pipeline, state: dict) -> dict:
    from src.retrieval.rrf import ReciprocalRankFusion
    from src.retrieval.reranker import RerankedResult

    # reconstruye FusedResult-like dicts → pipeline ya tiene helpers
    # usamos directamente pipeline.fusion y pipeline.reranker con objetos temporales
    # Para simplificar, delegamos a pipeline._ground/_dedupe helpers recreando listas
    # Convert dicts de estado a objetos mínimos para fusor
    class _Obj:
        def __init__(self, d): self.__dict__.update(d)
    # fusion necesita listas de resultados con chunk_id etc; usamos raw dicts via pipeline internals
    # Truco: llama pipeline.query sin generation reutilizando internals — más simple: usa pipeline.fusion/reranker directo sobre dicts convertidos
    # Aquí reconstruimos FusedResult manualmente
    from src.retrieval.rrf import FusedResult
    # dense/sparse/graph ya son dicts con chunk_id, text, score, metadata
    # Creamos wrappers para fuse
    def dicts_to_fused_dicts(ds, key):
        return [type("X", (), {"chunk_id": d["chunk_id"], "text": d["text"], "metadata": d["metadata"], key: d.get("score")})() for d in ds]

    # Si los resultados están vacíos, fusion vacía
    rp = getattr(pipeline, "retrieval", pipeline)
    dense_w = [type("R", (), d)() for d in state.get("dense", [])]
    sparse_w = [type("R", (), d)() for d in state.get("sparse", [])]
    graph_w = [type("R", (), d)() for d in state.get("graph", [])]
    # Usa pipeline internals que esperan objetos con chunk_id, text, metadata, score
    # Adaptamos llamando directamente al fusor con envoltorios
    fused = rp.fusion.fuse(dense_w, sparse_w, graph_w, top_k=rp.top_k_rrf)
    grounded = rp._ground_to_query(fused, state.get("question", ""))
    deduped = rp._dedupe_candidates(grounded)
    reranked = rp.reranker.rerank(state.get("question", ""), deduped, top_k=rp.top_k_final, diversity_lambda=rp.rerank_diversity, min_score=rp.rerank_min_score)
    # VRAM: cede a Ollama antes de generate
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    return {"fused": [r.__dict__ for r in fused], "reranked": [r.__dict__ for r in reranked]}


def generate_node(pipeline, state: dict) -> dict:
    from src.retrieval.reranker import RerankedResult
    rp = getattr(pipeline, "retrieval", pipeline)
    # reconstruye RerankedResult para generator
    reranked_objs = []
    for d in state.get("reranked", []):
        # d ya es dict con chunk_id, text, score, metadata
        reranked_objs.append(RerankedResult(chunk_id=d["chunk_id"], text=d["text"], score=d.get("score", 0), metadata=d.get("metadata", {})))
    gen = rp.generator.generate(
        question=state.get("question", ""),
        context=reranked_objs,
        graph_facts=state.get("graph_facts", []),
        metrics_table=state.get("metrics_block", ""),
    )
    return {"answer": gen.answer, "citations": gen.citations}
