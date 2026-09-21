from __future__ import annotations

"""API local FastAPI para el agente ReAct (streaming SSE + HITL).

Arranque: uvicorn api:app --host 127.0.0.1 --port 8000   (sin --reload)
"""

import asyncio
import collections
import json
import logging
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.env import load_env

logger = logging.getLogger(__name__)

# Un solo pipeline/agente compartidos (Kuzu = un escritor; pipeline síncrono).
GRAPH_LOCK = threading.Lock()
# thread_id -> {"event": threading.Event, "decision": str | bool | None}
RESUME: dict[str, dict] = {}


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    thread_id: Optional[str] = None


class ConfirmIn(BaseModel):
    thread_id: str
    decision: object = True  # "y"/"n", true/false...


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    load_env()
    from cli import build_pipeline, build_react

    workers = 2
    try:
        workers = int(os.getenv("GRAPH_WORKERS", "2"))
    except ValueError:
        pass
    import sys

    try:
        from datetime import datetime

        code_ts = datetime.fromtimestamp(os.path.getmtime(__file__)).strftime("%H:%M:%S")
    except Exception:
        code_ts = "?"
    logger.info(
        "API: pid=%d exe=%s api.py=%s — construyendo pipeline (workers=%d)...",
        os.getpid(),
        sys.executable,
        code_ts,
        workers,
    )
    pipeline = build_pipeline(max_workers=workers, batch_size=1)
    agent = build_react(pipeline)
    app.state.pipeline = pipeline
    app.state.react_agent = agent
    logger.info("API lista")
    yield
    try:
        pipeline.close()
    except Exception as exc:
        logger.debug("Error cerrando pipeline: %s", exc)


app = FastAPI(title="Financial GraphRAG (local)", lifespan=lifespan)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_text_chunk(msg_chunk) -> str:
    content = getattr(msg_chunk, "content", "")
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


def _is_ai_chunk(msg_chunk) -> bool:
    from langchain_core.messages import AIMessageChunk

    return isinstance(msg_chunk, AIMessageChunk) or getattr(msg_chunk, "type", "") in (
        "ai",
        "AIMessageChunk",
    )


def _ensure_memory(agent, thread_id: str) -> None:
    """Si el checkpointer en memoria está vacío pero hay transcripción en disco
    (p. ej. tras reiniciar el servidor), reinyecta el historial Q/A para que el
    modelo conserve el contexto de la conversación."""
    from langchain_core.messages import AIMessage, HumanMessage

    from src.agent import history_store

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        vals = state.values if isinstance(state.values, dict) else {}
        if vals.get("messages"):
            return
    except Exception:
        return
    conv = history_store.get_conversation(thread_id)
    if not conv or not conv.get("messages"):
        return
    msgs = []
    for m in conv["messages"][-20:]:
        if m.get("role") == "user":
            msgs.append(HumanMessage(content=m.get("text", "")))
        else:
            msgs.append(AIMessage(content=m.get("text", "")))
    if not msgs:
        return
    try:
        agent.update_state(config, {"messages": msgs})
        logger.info("Memoria reinyectada para thread %s (%d mensajes)", thread_id[:8], len(msgs))
    except Exception as exc:
        logger.warning("No se pudo reinyectar memoria: %s", exc)


def _repair_dangling_tool_calls(agent, thread_id: str) -> None:
    """Si un turno previo se canceló o falló abruptamente dejando un AIMessage con
    tool_calls sin su ToolMessage correspondiente, inyecta ToolMessages sintéticos de
    cancelación para que LangGraph valide el historial sin lanzar ValueError
    (INVALID_CHAT_HISTORY)."""
    from langchain_core.messages import ToolMessage

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = agent.get_state(config)
        msgs = state.values.get("messages", []) if isinstance(state.values, dict) else []
        if not msgs:
            return

        tool_call_ids: dict[str, str] = {}  # id -> name
        resolved_ids: set[str] = set()
        for m in msgs:
            tcs = getattr(m, "tool_calls", None) or []
            for tc in tcs:
                if isinstance(tc, dict):
                    t_id = tc.get("id")
                    t_name = tc.get("name", "tool")
                else:
                    t_id = getattr(tc, "id", None)
                    t_name = getattr(tc, "name", "tool")
                if t_id:
                    tool_call_ids[t_id] = t_name
            chunk_type = getattr(m, "type", "")
            if isinstance(m, ToolMessage) or chunk_type == "tool":
                tc_id = getattr(m, "tool_call_id", None)
                if tc_id:
                    resolved_ids.add(tc_id)

        missing_ids = {k: v for k, v in tool_call_ids.items() if k not in resolved_ids}
        if not missing_ids:
            return

        logger.info(
            "Reparando %d tool_calls huérfanos en thread %s: %s",
            len(missing_ids),
            thread_id[:8],
            list(missing_ids.keys()),
        )
        repairs = [
            ToolMessage(
                content="Operación detenida por el usuario. Progreso guardado.",
                tool_call_id=tc_id,
                name=tc_name,
            )
            for tc_id, tc_name in missing_ids.items()
        ]
        agent.update_state(config, {"messages": repairs})
        logger.info("Historial de chat reparado con éxito para thread %s", thread_id[:8])
    except Exception as exc:
        logger.warning("No se pudo revisar/reparar tool_calls huérfanos: %s", exc)


def _react_turn_events(agent, question: str, thread_id: str, emit) -> None:
    """Ejecuta un turno ReAct emitiendo eventos. Corre en worker thread."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from cli import INTERRUPT_HANDLERS

    from src.agent import history_store
    from src.agent.tools import JsonPrefaceFilter

    from src.agent import progress as _progress

    _progress.set_current_thread(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    pending_input = {"messages": [HumanMessage(content=question)]}
    announced: set[str] = set()
    full_text: list[str] = []
    preface = JsonPrefaceFilter()
    for _ in range(10):  # cota anti-loops del modelo
        for msg_chunk, _md in agent.stream(pending_input, config=config, stream_mode="messages"):
            if not _is_ai_chunk(msg_chunk):
                continue
            for tc in getattr(msg_chunk, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    tc_id, tc_name = tc.get("id"), tc.get("name")
                else:
                    tc_id, tc_name = getattr(tc, "id", None), getattr(tc, "name", "")
                key = tc_id or tc_name
                if key and key not in announced:
                    announced.add(key)
                    if tc_name:
                        emit("tool", {"name": tc_name})
            text = _extract_text_chunk(msg_chunk)
            if text:
                # Retiene un posible preámbulo JSON: solo se emite prosa.
                released = preface.feed(text)
                if released:
                    full_text.append(released)
                    emit("token", {"text": released})
        state = agent.get_state(config)
        pending = [i for t in state.tasks for i in (t.interrupts or [])]
        if not pending:
            if not state.next:
                break
            pending_input = None
            continue
        payload = pending[0].value or {}
        spec = INTERRUPT_HANDLERS.get(payload.get("action", ""))
        if spec is None:
            pending_input = Command(resume=None)
            continue
        notice = spec.notice.format_map(collections.defaultdict(str, payload))
        emit("interrupt", {"action": payload.get("action", ""), "notice": notice,
                           "question": spec.question, "payload": payload})
        box = RESUME.setdefault(thread_id, {"event": threading.Event(), "decision": None})
        box["event"].wait(timeout=600)
        decision = RESUME.pop(thread_id, {}).get("decision")
        if decision is None:
            _progress.clear(thread_id)
            emit("error", {"message": "Confirmación caducada (timeout). Turno cancelado."})
            return
        pending_input = Command(resume=decision)
    tail = preface.flush()
    if tail:
        full_text.append(tail)
        emit("token", {"text": tail})

    answer = "".join(full_text)
    history_store.append_turn(thread_id, question, answer)
    _progress.clear(thread_id)
    emit("done", {"answer": answer, "thread_id": thread_id})


@app.get("/health")
def health():
    from src.agent.company_registry import get_config_tickers

    try:
        tickers = get_config_tickers()
    except Exception:
        tickers = []
    return {"status": "ok", "model": "qwen3:8b", "tickers": tickers}


@app.post("/query")
async def query(body: QueryIn):
    thread_id = body.thread_id or str(uuid.uuid4())
    agent = app.state.react_agent
    q: queue.Queue = queue.Queue()

    def _worker():
        from src.agent import progress as _progress

        listener = lambda data: q.put(("progress", data))
        _progress.subscribe(thread_id, listener)
        with GRAPH_LOCK:
            try:
                _progress.set_current_thread(thread_id)
                _progress.clear_cancel(thread_id)
                _ensure_memory(agent, thread_id)
                _repair_dangling_tool_calls(agent, thread_id)
                _react_turn_events(agent, body.question, thread_id, lambda k, d: q.put((k, d)))
            except _progress.IngestCancelled:
                logger.info("Turno %s: ingesta detenida por el usuario", thread_id[:8])
                _progress.clear_cancel(thread_id)
                q.put(("cancelled", {"thread_id": thread_id}))
            except Exception as exc:
                logger.exception("Turno ReAct falló: %s", exc)
                q.put(("error", {"message": str(exc)}))
            finally:
                _progress.unsubscribe(thread_id, listener)
                _progress.clear(thread_id)
                _progress.clear_cancel(thread_id)
                q.put(("__end__", {}))

    threading.Thread(target=_worker, daemon=True).start()

    async def _gen():
        yield _sse("start", {"thread_id": thread_id})
        while True:
            kind, data = await asyncio.to_thread(q.get)
            if kind == "__end__":
                break
            yield _sse(kind, data)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/confirm")
def confirm(body: ConfirmIn):
    box = RESUME.get(body.thread_id)
    if box is None:
        raise HTTPException(status_code=409, detail="No hay confirmación pendiente para ese thread_id (¿ya se resolvió o caducó?).")
    box["decision"] = body.decision
    box["event"].set()
    return {"status": "resumed", "thread_id": body.thread_id}


class CancelIn(BaseModel):
    thread_id: str


@app.post("/cancel")
def cancel(body: CancelIn):
    """Detiene una ingesta en curso. Cooperativo: termina el lote actual
    (guarda lo avanzado para retomar) y cierra el turno. Si el turno está
    parado en un HITL, la confirmación se resuelve como 'n'."""
    from src.agent import progress as _progress

    _progress.request_cancel(body.thread_id)
    box = RESUME.get(body.thread_id)
    if box is not None:
        box["decision"] = "n"
        box["event"].set()
    return {"status": "cancelling", "thread_id": body.thread_id}


@app.get("/conversations")
def list_conversations():
    from src.agent import history_store

    return {"conversations": history_store.list_conversations()}


@app.get("/conversations/{thread_id}")
def get_conversation(thread_id: str):
    from src.agent import history_store

    conv = history_store.get_conversation(thread_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return conv


@app.delete("/conversations/{thread_id}")
def delete_conversation(thread_id: str):
    from src.agent import history_store

    if not history_store.delete_conversation(thread_id):
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    RESUME.pop(thread_id, None)
    return {"status": "deleted", "thread_id": thread_id}


@app.get("/", include_in_schema=False)
def index():
    # Sin caché: el frontend evoluciona a menudo y una copia vieja en el
    # navegador desincroniza la UI del backend (p. ej. botones sin endpoint).
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )
