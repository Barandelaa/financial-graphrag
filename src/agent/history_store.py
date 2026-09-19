from __future__ import annotations

"""Transcripciones de conversaciones en disco (data/conversations.json).

Persiste Q/A por thread_id para que el sidebar de la web pueda listar,
cargar y borrar conversaciones entre sesiones. No guarda chunks ni
tripletas, solo texto de usuario y respuestas finales (capado).
"""

import json
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_PATH = Path("data/conversations.json")
MAX_TEXT_CHARS = 4000
MAX_MESSAGES_PER_THREAD = 100

_lock = threading.Lock()


def _load() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("No se pudo leer %s: %s", HISTORY_PATH, exc)
        return {}


def _save(data: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_PATH)


def list_conversations() -> list[dict]:
    """Resumen ordenado por recencia: [{id, title, updated_at, turns}]."""
    with _lock:
        data = _load()
    items = [
        {
            "id": tid,
            "title": conv.get("title", "(sin título)"),
            "updated_at": conv.get("updated_at", 0),
            "turns": len(conv.get("messages", [])) // 2,
        }
        for tid, conv in data.items()
    ]
    items.sort(key=lambda c: c["updated_at"], reverse=True)
    return items


def get_conversation(thread_id: str) -> dict | None:
    with _lock:
        conv = _load().get(thread_id)
    if not conv:
        return None
    return {"id": thread_id, "title": conv.get("title", ""), "messages": conv.get("messages", [])}


def append_turn(thread_id: str, question: str, answer: str) -> None:
    """Añade un turno Q/A (crea la conversación con la primera pregunta como título)."""
    question = (question or "")[:MAX_TEXT_CHARS]
    answer = (answer or "")[:MAX_TEXT_CHARS]
    if not question and not answer:
        return
    with _lock:
        data = _load()
        conv = data.get(thread_id) or {"title": question[:60] or "(sin título)", "messages": []}
        conv["messages"].extend([
            {"role": "user", "text": question},
            {"role": "assistant", "text": answer},
        ])
        conv["messages"] = conv["messages"][-MAX_MESSAGES_PER_THREAD:]
        conv["updated_at"] = time.time()
        data[thread_id] = conv
        try:
            _save(data)
        except Exception as exc:
            logger.warning("No se pudo guardar historial: %s", exc)


def delete_conversation(thread_id: str) -> bool:
    with _lock:
        data = _load()
        if thread_id not in data:
            return False
        del data[thread_id]
        try:
            _save(data)
        except Exception as exc:
            logger.warning("No se pudo guardar historial: %s", exc)
        return True
