#!/usr/bin/env python3
"""Checks deterministas del pipeline Financial GraphRAG (sin LLM-juez).

Sustituye a la evaluación RAGAS: en vez de puntuar con un modelo juez
(poco fiable con modelos locales pequeños), verifica hechos objetivos
de cada respuesta del pipeline:

- respuesta no vacía y con citas,
- las citas apuntan al ticker / año / sección esperados,
- las tres vías de recuperación aportan resultados (densa, BM25, grafo).

Además detecta preguntas del dataset fuera del corpus ingerido
(ticker no configurado en data/companies.json o año no ingerido),
que fallarán por falta de datos y no por un bug del pipeline.

Uso:
    python evals/run_checks.py [--samples N] [--dataset ...] [--output ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import load_env
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else []


def load_corpus() -> tuple[set[str], set[str]]:
    """Devuelve (tickers configurados, años configurados) en minúsculas/str."""
    try:
        with open("data/companies.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set(), set()
    tickers = {_norm(t) for t in config.get("companies", [])}
    years = {str(y) for y in config.get("default_years", [])}
    return tickers, years


def check_sample(
    pipeline: FinancialGraphRAGPipeline,
    sample: dict,
    corpus_tickers: set[str],
    corpus_years: set[str],
) -> dict:
    question = sample.get("question", "")
    expected_ticker = _norm(sample.get("expected_ticker"))
    expected_year = str(sample.get("expected_year", ""))
    expected_section = _norm(sample.get("expected_section"))

    in_corpus = (
        (not expected_ticker or expected_ticker in corpus_tickers)
        and (not expected_year or expected_year in corpus_years)
    )

    try:
        result = pipeline.query(question)
    except Exception as exc:  # noqa: BLE001
        logger.error("Query falló para %r: %s", question[:80], exc)
        return {
            "question": question,
            "error": str(exc),
            "in_corpus": in_corpus,
            "checks": {},
        }

    citations = result.citations or []
    cited_tickers = {_norm(c.get("company_ticker")) for c in citations}
    cited_years = {str(c.get("fiscal_year", "")) for c in citations}
    cited_sections = {_norm(c.get("section_id")) for c in citations}

    checks = {
        "has_answer": bool((result.answer or "").strip()),
        "has_citations": bool(citations),
        "ticker_hit": bool(expected_ticker and expected_ticker in cited_tickers),
        "year_hit": bool(expected_year and expected_year in cited_years),
        "section_hit": bool(expected_section and expected_section in cited_sections),
        "dense_hit": result.dense_results > 0,
        "sparse_hit": result.sparse_results > 0,
        "graph_hit": result.graph_results > 0,
    }
    return {
        "question": question,
        "expected_ticker": sample.get("expected_ticker", ""),
        "expected_year": sample.get("expected_year", ""),
        "expected_section": sample.get("expected_section", ""),
        "in_corpus": in_corpus,
        "answer_preview": (result.answer or "")[:300],
        "counts": {
            "dense": result.dense_results,
            "sparse": result.sparse_results,
            "graph": result.graph_results,
            "facts": len(result.graph_facts),
            "citations": len(citations),
        },
        "checks": checks,
        "score": (
            sum(1 for v in checks.values() if v) / len(checks) if checks else 0.0
        ),
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DETERMINISTIC CHECKS SUMMARY")
    print("=" * 70)
    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print(f"  [ERROR] {r['question'][:60]} -> {r['error']}")
            continue
        flags = "".join("+" if v else "-" for v in r["checks"].values())
        ooc = "" if r["in_corpus"] else "  [FUERA DE CORPUS]"
        print(f"  [{flags}] {r['question'][:55]}{ooc}")
    print("-" * 70)
    print("  + = pasa, - = falla   (columnas: answer citas ticker año "
          "sección dense sparse graph)")
    if ok:
        avg = sum(r["score"] for r in ok) / len(ok)
        print(f"  Score medio: {avg:.2f} sobre {len(ok)} muestras OK "
              f"({len(results) - len(ok)} con error)")
    ooc = [r for r in results if not r.get("in_corpus", True)]
    if ooc:
        print(f"  Aviso: {len(ooc)} preguntas apuntan a datos no ingeridos "
              "(ticker/año fuera de data/companies.json).")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checks deterministas del pipeline (sin LLM-juez)"
    )
    parser.add_argument("--dataset", default="evals/test_dataset.json")
    parser.add_argument("--output", default="evals/results/checks_report.json")
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    load_env()

    samples = load_dataset(Path(args.dataset))
    if args.samples is not None:
        samples = samples[: args.samples]
    if not samples:
        print("Dataset vacío o no encontrado.")
        return

    corpus_tickers, corpus_years = load_corpus()
    pipeline = FinancialGraphRAGPipeline(llm=create_llm())
    try:
        results = [
            check_sample(pipeline, s, corpus_tickers, corpus_years)
            for s in samples
        ]
    finally:
        pipeline.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nInforme guardado en {out}")
    print_summary(results)


if __name__ == "__main__":
    main()
