from __future__ import annotations

import json
import logging
import os
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

MISSING = []
for d in sorted(PROCESSED.iterdir()):
    if not d.is_dir():
        continue
    chunks_f = d / "chunks.json"
    triplets_f = d / "triplets.json"
    if chunks_f.exists() and not triplets_f.exists():
        name = d.name.rsplit("_", 1)
        MISSING.append((name[0], int(name[1])))


def main() -> None:
    load_env()
    if not MISSING:
        print("No missing triplets found.")
        return

    print("Will reprocess graph for:", MISSING)
    llm = create_llm()
    pipeline = FinancialGraphRAGPipeline(llm=llm)

    done = 0
    failed = []
    try:
        for ticker, year in MISSING:
            pair_dir = PROCESSED / f"{ticker}_{year}"
            try:
                with open(pair_dir / "chunks.json", "r", encoding="utf-8") as f:
                    records = json.load(f)
                chunks = [Chunk(**r) for r in records]
                logging.info("Loaded %d chunks for %s/%s", len(chunks), ticker, year)

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