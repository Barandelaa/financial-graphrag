from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

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


def run_repl(pipeline: FinancialGraphRAGPipeline, use_agent: bool = True) -> None:
    print("=" * 60)
    print("Financial GraphRAG - Chat interactivo (LangGraph determinista single-turn)")
    print("Escribe una pregunta, 'añade AAPL 2026' para ingesta con confirmación, o /help.")
    print("=" * 60)

    agent = None
    if use_agent:
        try:
            agent = build_agent(pipeline)
            print("[Agent LangGraph activo: intent classify + HITL ingest_tool]")
        except Exception as exc:
            logger.warning("No se pudo inicializar agente, fallback a pipeline directo: %s", exc)
            agent = None

    import uuid

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not question:
            continue

        if question in ("/exit", "/quit"):
            print("Saliendo...")
            break
        if question == "/help":
            print(HELP_TEXT + "\n\nModo agente: escribe 'añade TICKER AÑO' (ej: añade AAPL 2026) para ingesta con confirmación.")
            continue
        if question == "/clear":
            print("\033c", end="")
            continue
        if question == "/ingest-all":
            print("Ingiriendo todas las empresas del config...")
            pipeline.ingest_companies()
            print("Ingesta completada.")
            continue
        if question.startswith("/ingest "):
            parts = question.split()
            if len(parts) < 3:
                print("Uso: /ingest <ticker> <año>")
                continue
            try:
                year = int(parts[2])
            except ValueError:
                print(f"Año inválido: {parts[2]}")
                continue
            print(f"Ingiriendo {parts[1].upper()} / {year}...")
            pipeline.ingest_and_index(parts[1].upper(), year)
            print("Ingesta completada.")
            continue
        if question.startswith("/"):
            print(f"Comando desconocido: {question}")
            continue

        # LangGraph agent path con HITL (si disponible y pregunta no es comando)
        if agent is not None:
            print("\nConsultando (agent)...\n")
            try:
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}
                # 1º invoke hasta interrupt_before ingest_tool o END
                result = agent.invoke({"question": question}, config=config)
                # Si hay interrupt (ingest), result contiene ingest_request pendiente
                # Detecta si el grafo se pausó
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
                        # reanuda (ejecutará ingest_tool)
                        result = agent.invoke(None, config=config)
                        ingest_msg = result.get("ingest_result") or str(result)
                        print(ingest_msg)
                        # muestra tool result si es ingest
                        for m in result.get("messages", [])[-2:]:
                            if isinstance(m, dict) and m.get("content"):
                                print(m["content"])
                            elif hasattr(m, "content"):
                                print(m.content)
                    else:
                        print("Ingesta cancelada, haciendo retrieval en su lugar...")
                        # cancela ingest y fuerza retrieve: invoca grafo no-interrupt
                        from src.agent.graph import build_agent_graph_no_interrupt

                        agent2 = build_agent_graph_no_interrupt(pipeline)
                        # fuerza intent retrieve
                        result = agent2.invoke({"question": question, "intent": "retrieve", "ticker": None, "year": None}, config={"configurable": {"thread_id": str(uuid.uuid4())}})
                        answer = result.get("answer", "")
                        citations = result.get("citations", [])
                        facts = result.get("graph_facts", [])
                        metrics = result.get("metrics_rows", [])
                        print("-" * 60)
                        print(answer)
                        print("-" * 60)
                        print(f"[Facts: {len(facts)} | Metrics: {len(metrics)} | Citations: {len(citations)}]")
                        if citations:
                            for c in citations[:5]:
                                print(f"  - {c}")
                    continue
                # No fue ingest → es retrieval
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                facts = result.get("graph_facts", [])
                metrics = result.get("metrics_rows", [])
                print("-" * 60)
                print(answer)
                print("-" * 60)
                print(f"[Facts: {len(facts)} | Metrics: {len(metrics)} | Citations: {len(citations)}]")
                if citations:
                    print("\nCitas:")
                    for c in citations[:5]:
                        print(f"  - {c}")
                if not answer:
                    print("\n(No se generó respuesta; revisa los logs.)")
                continue
            except Exception as exc:
                logger.warning("Agent falló, fallback a pipeline: %s", exc)

        print("\nConsultando (pipeline)...\n")
        try:
            result = pipeline.query(question)
        except Exception as exc:
            print(f"Error al consultar: {exc}")
            continue

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

        run_repl(pipeline, use_agent=not args.no_agent)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
