from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.env import load_env
from src.ingestion.chunker import Chunk
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed_chunks"


def _find_missing(include_partial: bool = True):
    # 1) Pares esperados según data/companies.json
    expected: list[tuple[str, int]] = []
    try:
        with open(ROOT / "data" / "companies.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        companies = cfg.get("companies", [])
        years = cfg.get("default_years", [])
        for t in companies:
            for y in years:
                expected.append((t, int(y)))
    except Exception:
        expected = []

    missing: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    # 2) Revisa cada par esperado: falta si no hay chunks, chunks vacío, o sin triplets/parcial
    for ticker, year in expected:
        pair_dir = PROCESSED / f"{ticker}_{year}"
        chunks_f = pair_dir / "chunks.json"
        triplets_f = pair_dir / "triplets.json"
        needs = False
        reason = ""
        if not chunks_f.exists():
            needs = True
            reason = "no chunks.json"
        else:
            try:
                with open(chunks_f, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if not chunks or len(chunks) == 0:
                    needs = True
                    reason = "chunks vacío"
                elif not triplets_f.exists():
                    needs = True
                    reason = "no triplets.json"
                elif include_partial:
                    with open(triplets_f, "r", encoding="utf-8") as f:
                        triplets = json.load(f)
                    cached_ids = {r.get("chunk_id") for r in triplets}
                    if len(cached_ids) < len(chunks):
                        needs = True
                        reason = f"triplets parcial {len(cached_ids)}/{len(chunks)}"
            except Exception as exc:
                needs = True
                reason = f"error lectura: {exc}"
        if needs:
            logging.info("Missing detected %s/%s: %s", ticker, year, reason)
            missing.append((ticker, year))
            seen.add((ticker, year))

    # 3) Además revisa cualquier directorio huérfano (por si hay configs antiguas)
    if PROCESSED.exists():
        for d in sorted(PROCESSED.iterdir()):
            if not d.is_dir():
                continue
            try:
                name = d.name.rsplit("_", 1)
                key = (name[0], int(name[1]))
            except Exception:
                continue
            if key in seen or key in expected:
                continue
            chunks_f = d / "chunks.json"
            triplets_f = d / "triplets.json"
            if chunks_f.exists() and not triplets_f.exists():
                missing.append(key)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocesa pares sin triplets o con triplets parciales")
    parser.add_argument("--workers", type=int, default=4, help="Workers paralelos (default 4 para 12GB VRAM)")
    parser.add_argument("--batch-size", type=int, default=1, help="Chunks por llamada LLM (default 1=fiable con qwen3; 2-3 experimental)")
    parser.add_argument("--include-partial", action="store_true", default=True, help="Incluye filings con triplets parciales")
    parser.add_argument("--no-partial", dest="include_partial", action="store_false", help="Solo faltantes totales")
    args = parser.parse_args()

    load_env()
    missing = _find_missing(include_partial=args.include_partial)
    if not missing:
        print("No missing triplets found.")
        return

    print(f"Will reprocess graph for: {missing} (workers={args.workers}, batch={args.batch_size})")
    llm = create_llm()
    pipeline = FinancialGraphRAGPipeline(llm=llm, graph_max_workers=args.workers, graph_batch_size=args.batch_size)

    done = 0
    failed = []
    try:
        for ticker, year in missing:
            pair_dir = PROCESSED / f"{ticker}_{year}"
            # Si chunks.json está vacío o falta, fuerza re-ingesta (download+parse). Si no, reutiliza cache.
            chunks_f = pair_dir / "chunks.json"
            force_reingest = False
            if not chunks_f.exists():
                force_reingest = True
            else:
                try:
                    with open(chunks_f, "r", encoding="utf-8") as f:
                        recs = json.load(f)
                    if not recs or len(recs) == 0:
                        force_reingest = True
                except Exception:
                    force_reingest = True

            try:
                if force_reingest:
                    logging.info("Force re-ingest for %s/%s (chunks vacío/faltante) — llama a ingest_and_index(use_cache=False)", ticker, year)
                    chunks = pipeline.ingest_and_index(ticker, year, use_cache=False)
                    # ingest_and_index ya hizo graph+retrieval; evita duplicar
                    persist_ok = (pair_dir / "triplets.json").exists()
                    print("DONE", ticker, year, "chunks=", len(chunks),
                          "persisted=", persist_ok, "(via ingest_and_index)")
                else:
                    with open(pair_dir / "chunks.json", "r", encoding="utf-8") as f:
                        records = json.load(f)
                    chunks = [Chunk(**r) for r in records]
                    logging.info("Loaded %d cached chunks for %s/%s", len(chunks), ticker, year)

                    chunk_dicts = [c.to_dict() for c in chunks]

                    triplets = pipeline.graph.process_chunks(chunks)
                    pipeline.retrieval.index_chunks(chunk_dicts)

                    persist_ok = (pair_dir / "triplets.json").exists()
                    print("DONE", ticker, year, "chunks=", len(chunks),
                          "triplets=", len(triplets), "persisted=", persist_ok)
                done += 1
            except Exception as exc:
                logging.exception("Failed %s/%s: %s", ticker, year, exc)
                failed.append((ticker, year))
    finally:
        pipeline.close()
    print("Reprocessing complete. ok=%d failed=%s" % (done, failed))


if __name__ == "__main__":
    main()
