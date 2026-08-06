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
  /ingest <ticker> <año>      -> Ingiere y indexa un 10-K (ej: /ingest AAPL 2023)
  /ingest-all                 -> Ingiere todas las empresas de data/companies.json
  /clear                      -> Limpia la pantalla
  /help                       -> Muestra esta ayuda
  /exit o /quit               -> Sale del programa
""".strip()


def build_pipeline() -> FinancialGraphRAGPipeline:
    llm = create_llm()
    return FinancialGraphRAGPipeline(llm=llm)


def run_repl(pipeline: FinancialGraphRAGPipeline) -> None:
    print("=" * 60)
    print("Financial GraphRAG - Chat interactivo")
    print("Escribe una pregunta o /help para ver los comandos.")
    print("=" * 60)

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
            print(HELP_TEXT)
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

        print("\nConsultando...\n")
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
            f"Graph: {result.graph_results}]"
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
    args = parser.parse_args(argv)

    load_env()

    pipeline = build_pipeline()

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

        run_repl(pipeline)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
