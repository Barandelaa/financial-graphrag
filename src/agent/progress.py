from __future__ import annotations

"""Progreso de ingesta por thread para la web (y consola).

El worker que ejecuta el turno fija su thread_id; el pipeline publica avances
(download/extract por ticker/año) y la API los reemite como eventos SSE
`progress` para pintar una barra por año. Fuera de la web (CLI, reprocess)
los hooks son no-op: el logging habitual sigue intacto.
"""

import threading
from typing import Any, Optional

_local = threading.local()
_global_thread_id: Optional[str] = None
_store: dict[str, dict[str, Any]] = {}
_subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
_store_lock = threading.Lock()


def set_current_thread(thread_id: Optional[str]) -> None:
    global _global_thread_id
    _local.thread_id = thread_id
    _global_thread_id = thread_id


def current_thread() -> Optional[str]:
    return getattr(_local, "thread_id", None) or _global_thread_id


def subscribe(thread_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
    with _store_lock:
        _subscribers.setdefault(thread_id, []).append(callback)


def unsubscribe(thread_id: str, callback: Callable[[dict[str, Any]], None]) -> None:
    with _store_lock:
        cbs = _subscribers.get(thread_id)
        if cbs and callback in cbs:
            cbs.remove(callback)
            if not cbs:
                _subscribers.pop(thread_id, None)


def report(thread_id: Optional[str], **fields: Any) -> None:
    tid = thread_id or current_thread()
    if not tid:
        return
    with _store_lock:
        entry = _store.setdefault(tid, {})
        entry.update(fields)
        cbs = list(_subscribers.get(tid, []))
    data = dict(entry)
    for cb in cbs:
        try:
            cb(data)
        except Exception:
            pass


def report_current(**fields: Any) -> None:
    report(current_thread(), **fields)


def get(thread_id: str) -> Optional[dict[str, Any]]:
    with _store_lock:
        entry = _store.get(thread_id)
        return dict(entry) if entry else None


def clear(thread_id: str) -> None:
    global _global_thread_id
    with _store_lock:
        _store.pop(thread_id, None)
        _subscribers.pop(thread_id, None)
    if _global_thread_id == thread_id:
        _global_thread_id = None
    if getattr(_local, "thread_id", None) == thread_id:
        _local.thread_id = None


class IngestCancelled(BaseException):
    """Cancelación cooperativa de una ingesta en curso (botón Detener / POST /cancel).

    Hereda de BaseException a propósito: los `except Exception` del extractor
    y del pipeline la dejan pasar para que llegue intacta al worker del turno.
    """


_cancel: dict[str, threading.Event] = {}


def request_cancel(thread_id: str) -> bool:
    """Marca un turno para cancelación. Devuelve False si no hay turno conocido."""
    with _store_lock:
        box = _cancel.setdefault(thread_id, threading.Event())
    box.set()
    return True


def cancel_requested(thread_id: Optional[str] = None) -> bool:
    tid = thread_id or current_thread()
    if tid:
        with _store_lock:
            box = _cancel.get(tid)
        if box and box.is_set():
            return True
    with _store_lock:
        return any(b.is_set() for b in _cancel.values())


def clear_cancel(thread_id: str) -> None:
    with _store_lock:
        _cancel.pop(thread_id, None)
