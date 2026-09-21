from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.env import load_env
from src.ingestion.chunker import Chunk
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)

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


def _has_usable_chunks(pair_dir: Path) -> bool:
    """True si chunks.json existe y no está vacío (retomar es posible)."""
    chunks_f = pair_dir / "chunks.json"
    if not chunks_f.exists():
        return False
    try:
        with open(chunks_f, "r", encoding="utf-8") as f:
            recs = json.load(f)
        return bool(recs)
    except Exception:
        return False


LOCK_NAME = ".ingest.lock"
LOCK_STALE_HOURS = 6


def _lock_path(ticker: str, year: int) -> Path:
    return PROCESSED / f"{ticker}_{year}" / LOCK_NAME


def _acquire_lock(ticker: str, year: int) -> tuple[bool, str]:
    """Single-flight entre procesos (API, CLI, reprocess): un solo escritor por par.

    Devuelve (True, "") si se adquiere; (False, motivo) si otro proceso vivo
    la tiene. Locks de PIDs muertos o de más de LOCK_STALE_HOURS se roban.
    """
    import os
    import time

    path = _lock_path(ticker, year)
    now = time.time()
    if path.exists():
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        pid = info.get("pid")
        started = info.get("started_at", 0)
        try:
            import psutil

            alive = bool(pid) and psutil.pid_exists(int(pid))
        except ImportError:
            alive = True  # sin psutil no se comprueba: solo decide la antigüedad
        if alive and (now - started) < LOCK_STALE_HOURS * 3600:
            return False, f"ingesta en curso por PID {pid} (lock de {path})"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": now}), encoding="utf-8"
    )
    return True, ""


def _release_lock(ticker: str, year: int) -> None:
    try:
        _lock_path(ticker, year).unlink(missing_ok=True)
    except Exception:
        pass


def _purge_pair(pipeline: FinancialGraphRAGPipeline, ticker: str, year: int) -> None:
    """Borra todo rastro de un par antes de una ingesta limpia: ficheros,
    nodos DocumentChunk + métricas year-scoped en Kuzu y filas en LanceDB.

    Evita linajes huérfanos cuando se re-parsea (los chunk_ids nuevos no
    coincidirían con nada anterior y duplicarían stores).
    """
    pair_dir = PROCESSED / f"{ticker}_{year}"
    for name in ("chunks.json", "triplets.json"):
        try:
            (pair_dir / name).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("No se pudo borrar %s: %s", pair_dir / name, exc)

    try:
        conn = pipeline.graph.schema.connection
        conn.execute(
            "MATCH (c:DocumentChunk) WHERE c.company_ticker = $t AND c.fiscal_year = $y "
            "DETACH DELETE c",
            {"t": ticker, "y": year},
        )
        conn.execute(
            "MATCH (m:FinancialMetric) WHERE m.id STARTS WITH $prefix DELETE m",
            {"prefix": f"{ticker}_{year}_"},
        )
        try:
            conn.execute("CHECKPOINT")
        except RuntimeError:
            pass
        logger.info("Purgados nodos Kuzu de %s/%s", ticker, year)
    except Exception as exc:
        logger.warning("Purge Kuzu %s/%s falló: %s", ticker, year, exc)

    try:
        dense = pipeline.retrieval.dense
        table = dense._db.open_table(dense.table_name)
        table.delete(f"company_ticker = '{ticker}' AND fiscal_year = {int(year)}")
        logger.info("Purgadas filas LanceDB de %s/%s", ticker, year)
    except Exception as exc:
        logger.debug("Purge LanceDB %s/%s omitido: %s", ticker, year, exc)


def reprocess_pair(
    pipeline: FinancialGraphRAGPipeline,
    ticker: str,
    year: int,
    force_clean: bool = False,
) -> dict:
    """Procesa un par ticker/año retomando parciales en disco (importable por el agente).

    - Si hay chunks.json usable y no se pide limpieza: reutiliza caché
      (conserva chunk_ids → triplets.json parcial se aprovecha).
    - Si no hay nada o force_clean=True: purga stores y re-parsea con
      ingest_and_index(use_cache=False).
    - Single-flight: si otro proceso lo está procesando, lanza RuntimeError.
    Devuelve dict con el resultado; lanza excepción si falla.
    """
    ok, reason = _acquire_lock(ticker, year)
    if not ok:
        raise RuntimeError(reason)
    try:
        pair_dir = PROCESSED / f"{ticker}_{year}"
        from src.agent.progress import report_current

        if _has_usable_chunks(pair_dir) and not force_clean:
            with open(pair_dir / "chunks.json", "r", encoding="utf-8") as f:
                records = json.load(f)
            chunks = [Chunk(**r) for r in records]
            logging.info("Loaded %d cached chunks for %s/%s (resume)", len(chunks), ticker, year)
            report_current(ticker=ticker, year=year, phase="chunks", done=len(chunks), total=len(chunks))

            chunk_dicts = [c.to_dict() for c in chunks]

            triplets = pipeline.graph.process_chunks(chunks)
            report_current(ticker=ticker, year=year, phase="index", done=0, total=len(chunk_dicts))
            pipeline.retrieval.index_chunks(chunk_dicts)
            report_current(ticker=ticker, year=year, phase="done", done=len(chunks), total=len(chunks))

            persist_ok = (pair_dir / "triplets.json").exists()
            logging.info("DONE %s %s chunks=%d triplets=%d persisted=%s (resume)",
                         ticker, year, len(chunks), len(triplets), persist_ok)
            return {
                "ticker": ticker, "year": year, "chunks": len(chunks),
                "triplets": len(triplets), "persisted": persist_ok, "resumed": True,
            }

        logging.info("Clean ingest for %s/%s (force_clean=%s) — purga stores y llama a ingest_and_index(use_cache=False)",
                     ticker, year, force_clean)
        _purge_pair(pipeline, ticker, year)
        chunks = pipeline.ingest_and_index(ticker, year, use_cache=False)
        report_current(ticker=ticker, year=year, phase="done", done=len(chunks), total=len(chunks))
        # ingest_and_index ya hizo graph+retrieval; evita duplicar
        persist_ok = (pair_dir / "triplets.json").exists()
        logging.info("DONE %s %s chunks=%d persisted=%s (clean)",
                     ticker, year, len(chunks), persist_ok)
        return {
            "ticker": ticker, "year": year, "chunks": len(chunks),
            "triplets": None, "persisted": persist_ok, "resumed": False,
        }
    finally:
        _release_lock(ticker, year)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Reprocesa pares sin triplets o con triplets parciales")
    parser.add_argument("--workers", type=int, default=4, help="Workers paralelos (default 4 para 12GB VRAM)")
    parser.add_argument("--batch-size", type=int, default=1, help="Chunks por llamada LLM en batch (default 1=fiable con qwen3; 2-3 experimental)")
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
            try:
                res = reprocess_pair(pipeline, ticker, year)
                print("DONE", ticker, year, "chunks=", res["chunks"],
                      "triplets=", res["triplets"], "persisted=", res["persisted"],
                      "(resume)" if res["resumed"] else "(clean)")
                done += 1
            except Exception as exc:
                logging.exception("Failed %s/%s: %s", ticker, year, exc)
                failed.append((ticker, year))
    finally:
        pipeline.close()
    print("Reprocessing complete. ok=%d failed=%s" % (done, failed))


if __name__ == "__main__":
    main()
