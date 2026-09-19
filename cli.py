from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import logging
import sys
import uuid
from typing import Callable, List, Optional

from src.env import load_env
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)

HELP_TEXT = """
Comandos disponibles:
  <pregunta>                  -> Consulta al pipeline RAG
  /ingest <ticker> <año>      -> Ingiere e indexa un 10-K (ej: /ingest AAPL 2023)
  /ingest-all                 -> Ingiere todas las empresas de data/companies.json
  /clear                      -> Limpia la pantalla
  /help                       -> Muestra esta ayuda
  /exit o /quit               -> Sale del programa
""".strip()


def build_pipeline(
    max_workers: int = 4,
    batch_size: int = 1,
) -> FinancialGraphRAGPipeline:
    llm = create_llm()
    return FinancialGraphRAGPipeline(
        llm=llm,
        graph_max_workers=max_workers,
        graph_batch_size=batch_size,
    )


def build_agent(pipeline: FinancialGraphRAGPipeline):
    from src.agent.graph import build_agent_graph

    return build_agent_graph(pipeline)


def build_react(pipeline: FinancialGraphRAGPipeline):
    from src.agent.react_graph import build_react_agent

    return build_react_agent(pipeline)


class _Action(enum.Enum):
    CONTINUE = "continue"
    ASK_MODEL = "ask_model"
    QUIT = "quit"


@dataclasses.dataclass
class _ReplCtx:
    pipeline: FinancialGraphRAGPipeline
    agent: Optional[object] = None
    react_agent: Optional[object] = None
    thread_id: str = ""


# --- Comandos REPL: tabla de despacho (añadir uno nuevo = una entrada) ---

def _cmd_quit(ctx: _ReplCtx, text: str) -> _Action:
    print("Saliendo...")
    return _Action.QUIT


def _cmd_help(ctx: _ReplCtx, text: str) -> _Action:
    print(HELP_TEXT + "\n\nModo agente: escribe 'añade TICKER AÑO' (ej: añade AAPL 2026) para ingesta con confirmación.\nMemoria: recuerda últimas Q/A para '¿y en 2023?' sin repetir ticker.")
    return _Action.CONTINUE


def _cmd_clear(ctx: _ReplCtx, text: str) -> _Action:
    print("\033c", end="")
    ctx.thread_id = str(uuid.uuid4())
    print(f"[Memoria limpiada, nuevo thread {ctx.thread_id[:8]}]")
    return _Action.CONTINUE


def _cmd_ingest_all(ctx: _ReplCtx, text: str) -> _Action:
    print("Ingiriendo todas las empresas del config...")
    ctx.pipeline.ingest_companies()
    print("Ingesta completada.")
    return _Action.CONTINUE


def _cmd_ingest(ctx: _ReplCtx, text: str) -> _Action:
    parts = text.split()
    if len(parts) < 3:
        print("Uso: /ingest <ticker> <año>")
        return _Action.CONTINUE
    try:
        year = int(parts[2])
    except ValueError:
        print(f"Año inválido: {parts[2]}")
        return _Action.CONTINUE
    print(f"Ingiriendo {parts[1].upper()} / {year}...")
    ctx.pipeline.ingest_and_index(parts[1].upper(), year)
    print("Ingesta completada.")
    return _Action.CONTINUE


_COMMANDS: dict[str, Callable[[_ReplCtx, str], _Action]] = {
    "/exit": _cmd_quit,
    "/quit": _cmd_quit,
    "/help": _cmd_help,
    "/clear": _cmd_clear,
    "/ingest-all": _cmd_ingest_all,
}


def dispatch_command(ctx: _ReplCtx, question: str) -> _Action:
    if not question:
        return _Action.CONTINUE
    if question.startswith("/ingest "):
        return _cmd_ingest(ctx, question)
    handler = _COMMANDS.get(question.split()[0])
    if handler is None:
        if question.startswith("/"):
            print(f"Comando desconocido: {question}")
            return _Action.CONTINUE
        return _Action.ASK_MODEL
    return handler(ctx, question)


# --- Interrupts HITL: registro (una futura tool con HITL = una entrada) ---

@dataclasses.dataclass
class _InterruptSpec:
    notice: str
    question: str


INTERRUPT_HANDLERS: dict[str, _InterruptSpec] = {
    "confirm_add_company": _InterruptSpec(
        "[ReAct] El modelo quiere añadir {ticker} a companies.json.",
        "¿Confirmas editar companies.json? (y/n): ",
    ),
    "confirm_ingest": _InterruptSpec(
        "[ReAct] El modelo quiere ingerir: {ticker} / {year}",
        "¿Confirmas ingesta? (y/n): ",
    ),
}


# --- Prints compartido ---

def extract_last_answer(messages) -> str:
    from src.agent.tools import strip_leading_json_block

    for m in reversed(messages or []):
        if isinstance(m, dict):
            role, content, tcs = m.get("type", ""), m.get("content", ""), m.get("tool_calls")
        else:
            role, content, tcs = getattr(m, "type", ""), getattr(m, "content", ""), getattr(m, "tool_calls", None)
        if role == "ai" and content and not tcs:
            text = content if isinstance(content, str) else str(content)
            return strip_leading_json_block(text)
    return ""


def print_answer_block(answer: str) -> None:
    print("-" * 60)
    print(answer or "(sin respuesta del modelo; revisa logs)")
    print("-" * 60)


def print_agent_result(answer: str, citations, facts, metrics, warn_if_empty: bool = True) -> None:
    print("-" * 60)
    print(answer)
    print("-" * 60)
    print(f"[Facts: {len(facts)} | Metrics: {len(metrics)} | Citations: {len(citations)}]")
    if citations:
        print("\nCitas:")
        for c in citations[:5]:
            print(f"  - {c}")
    if warn_if_empty and not answer:
        print("\n(No se generó respuesta; revisa los logs.)")


def print_pipeline_result(result) -> None:
    print("-" * 60)
    print(result.answer)
    print("-" * 60)
    print(
        f"[Dense: {result.dense_results} | Sparse: {result.sparse_results} | "
        f"Graph: {result.graph_results} | Facts: {len(result.graph_facts)}]"
    )
    if result.citations:
        print("\nCitas:")
        for c in result.citations:
            print(f"  - {c}")
    if not result.answer:
        print("\n(No se generó respuesta; revisa los logs.)")


# --- Turnos por modo (True = gestionado, False = fallback al siguiente) ---

def _print_stream_text(text: str) -> bool:
    """Imprime un fragmento de token y devuelve si el cursor queda a mitad de línea."""
    print(text, end="", flush=True)
    return not text.endswith("\n")


def _ensure_newline(line_open: bool) -> bool:
    if line_open:
        print(flush=True)
    return False


def run_react_turn(react_agent, question: str, thread_id: str) -> bool:
    print("\nConsultando (react)...\n")
    print("-" * 60)
    try:
        from langchain_core.messages import AIMessageChunk, HumanMessage
        from langgraph.types import Command
        from src.agent.tools import JsonPrefaceFilter

        config = {"configurable": {"thread_id": thread_id}}
        pending_input = {"messages": [HumanMessage(content=question)]}
        announced_tools: set[str] = set()
        streamed_any = False
        line_open = False
        preface = JsonPrefaceFilter()
        for _ in range(10):  # cota anti-loops del modelo
            for msg_chunk, _metadata in react_agent.stream(
                pending_input, config=config, stream_mode="messages"
            ):
                # En langchain_core>=1.x los chunks exponen type='AIMessageChunk'
                if not isinstance(msg_chunk, AIMessageChunk) and getattr(
                    msg_chunk, "type", ""
                ) not in ("ai", "AIMessageChunk"):
                    continue
                for tc in getattr(msg_chunk, "tool_calls", None) or []:
                    if isinstance(tc, dict):
                        tc_id, tc_name = tc.get("id"), tc.get("name")
                    else:
                        tc_id, tc_name = getattr(tc, "id", None), getattr(tc, "name", "")
                    key = tc_id or tc_name
                    if key and key not in announced_tools:
                        announced_tools.add(key)
                        if tc_name:
                            line_open = _ensure_newline(line_open)
                            print(f">> {tc_name}...", flush=True)
                            line_open = True
                content = getattr(msg_chunk, "content", "")
                if isinstance(content, list):
                    text = "".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = content if isinstance(content, str) else ""
                if text:
                    # Retiene un posible preámbulo JSON: solo se muestra prosa.
                    released = preface.feed(text)
                    if released:
                        streamed_any = True
                        line_open = _print_stream_text(released)
            state = react_agent.get_state(config)
            pending = [i for t in state.tasks for i in (t.interrupts or [])]
            if not pending:
                if not state.next:
                    break
                pending_input = None
                continue
            payload = pending[0].value or {}
            spec = INTERRUPT_HANDLERS.get(payload.get("action", ""))
            if spec is None:
                # Interrupt desconocido: reanuda sin valor
                pending_input = Command(resume=None)
                continue
            line_open = _ensure_newline(line_open)
            print(spec.notice.format_map(collections.defaultdict(str, payload)))
            confirm = input(spec.question).strip().lower()
            if confirm not in ("y", "yes", "s", "si"):
                print("Cancelado por el usuario.")
            pending_input = Command(resume=confirm)
        tail = preface.flush()
        if tail:
            streamed_any = True
            line_open = _print_stream_text(tail)
        if not streamed_any:
            # Respaldo: si no llegó ningún token (p. ej. error a mitad de stream
            # ya gestionado), muestra la última respuesta del estado.
            state = react_agent.get_state(config)
            msgs = state.values.get("messages", []) if isinstance(state.values, dict) else []
            print_answer_block(extract_last_answer(msgs))
            return True
        line_open = _ensure_newline(line_open)
        print("-" * 60)
        return True
    except Exception as exc:
        logger.warning("ReAct falló, fallback a pipeline: %s", exc)
        return False


def run_agent_turn(agent, pipeline: FinancialGraphRAGPipeline, question: str, thread_id: str) -> bool:
    print("\nConsultando (agent)...\n")
    try:
        config = {"configurable": {"thread_id": thread_id}}
        # 1º invoke hasta interrupt_before ingest_tool o END
        result = agent.invoke({"question": question}, config=config)
        state = agent.get_state(config)
        if state.next and "ingest_tool" in state.next:
            vals = state.values
            ticker = vals.get("ticker")
            year = vals.get("year")
            print(f"[Agent] Detectado intento de ingesta: {ticker} / {year}")
            print(f"¿Confirmas ingesta de {ticker} {year}? (y/n): ", end="", flush=True)
            confirm = input().strip().lower()
            if confirm in ("y", "yes", "s", "si"):
                print(f"Ingiriendo {ticker}/{year}...")
                result = agent.invoke(None, config=config)
                ingest_msg = result.get("ingest_result") or str(result)
                print(ingest_msg)
                for m in result.get("messages", [])[-2:]:
                    if isinstance(m, dict) and m.get("content"):
                        print(m["content"])
                    elif hasattr(m, "content"):
                        print(m.content)
            else:
                print("Ingesta cancelada, haciendo retrieval en su lugar...")
                from src.agent.graph import build_agent_graph_no_interrupt

                agent2 = build_agent_graph_no_interrupt(pipeline)
                result = agent2.invoke({"question": question, "intent": "retrieve", "ticker": None, "year": None}, config={"configurable": {"thread_id": thread_id}})
                print_agent_result(
                    result.get("answer", ""),
                    result.get("citations", []),
                    result.get("graph_facts", []),
                    result.get("metrics_rows", []),
                    warn_if_empty=False,
                )
            return True
        print_agent_result(
            result.get("answer", ""),
            result.get("citations", []),
            result.get("graph_facts", []),
            result.get("metrics_rows", []),
        )
        return True
    except Exception as exc:
        logger.warning("Agent falló, fallback a pipeline: %s", exc)
        return False


def run_pipeline_turn(pipeline: FinancialGraphRAGPipeline, question: str) -> None:
    print("\nConsultando (pipeline)...\n")
    try:
        result = pipeline.query(question)
    except Exception as exc:
        print(f"Error al consultar: {exc}")
        return
    print_pipeline_result(result)


def run_repl(pipeline: FinancialGraphRAGPipeline, use_agent: bool = True, use_react: bool = False) -> None:
    print("=" * 60)
    if use_react:
        print("Financial GraphRAG - Chat ReAct (tools + memoria)")
        print("El modelo decide qué tool usar. 'añade AAPL 2026' pide confirmación, o /help.")
    else:
        print("Financial GraphRAG - Chat interactivo (LangGraph con memoria Q/A)")
        print("Recuerda preguntas/respuestas previas (no chunks). Escribe 'añade AAPL 2026' para ingesta con confirmación, o /help.")
    print("=" * 60)

    agent = None
    react_agent = None
    if use_agent:
        if use_react:
            try:
                react_agent = build_react(pipeline)
                print("[Agent ReAct activo: query_financial_rag + lookup_metrics + calculator + propose_new_company + add_company_to_config/ingest_10k HITL]")
            except Exception as exc:
                logger.warning("No se pudo inicializar ReAct, fallback a pipeline directo: %s", exc)
        else:
            try:
                agent = build_agent(pipeline)
                print("[Agent LangGraph activo: memoria Q/A + intent classify + HITL ingest_tool]")
            except Exception as exc:
                logger.warning("No se pudo inicializar agente, fallback a pipeline directo: %s", exc)
                agent = None

    # thread_id persistente para memoria conversacional (no uuid por pregunta)
    ctx = _ReplCtx(pipeline=pipeline, agent=agent, react_agent=react_agent, thread_id=str(uuid.uuid4()))
    print(f"[Memoria conversacional: thread {ctx.thread_id[:8]} | /clear limpia historial]")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        action = dispatch_command(ctx, question)
        if action is _Action.QUIT:
            break
        if action is _Action.CONTINUE:
            continue

        # Turnos por modo con fallback en cadena: react → determinista → pipeline.
        if ctx.react_agent is not None and run_react_turn(ctx.react_agent, question, ctx.thread_id):
            continue
        if ctx.agent is not None and run_agent_turn(ctx.agent, pipeline, question, ctx.thread_id):
            continue
        run_pipeline_turn(pipeline, question)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Chat interactivo contra el pipeline Financial GraphRAG"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingiere data/companies.json antes de abrir el chat",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker a ingerir al arrancar (requiere --year)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Año fiscal a ingerir al arrancar (requiere --ticker)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Workers paralelos para extracción de tripletas (default 4, óptimo para 12GB VRAM)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Chunks por llamada LLM en batch (default 1=paralelo fiable con qwen3:8b; 2-3 experimental)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Desactiva LangGraph agent y usa pipeline directo",
    )
    parser.add_argument(
        "--react",
        action="store_true",
        help="Usa agente ReAct (el modelo decide tools) en vez del determinista",
    )
    args = parser.parse_args(argv)

    load_env()

    pipeline = build_pipeline(max_workers=args.workers, batch_size=args.batch_size)

    try:
        if args.ticker or args.year:
            if not args.ticker or not args.year:
                parser.error("--ticker y --year deben usarse juntos")
            pipeline.ingest_and_index(args.ticker.upper(), args.year)
            print(f"10-K de {args.ticker.upper()} / {args.year} ingerido.")
        elif args.ingest:
            print("Ingiriendo data/companies.json...")
            pipeline.ingest_companies()
            print("Ingesta completada.")

        run_repl(pipeline, use_agent=not args.no_agent, use_react=args.react)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
