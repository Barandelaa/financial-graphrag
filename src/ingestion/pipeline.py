from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from src.ingestion.chunker import Chunk, DocumentChunker
from src.ingestion.downloader import SEC10KDownloader
from src.ingestion.parser import SEC10KParser

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        raw_data_dir: str | Path = "data/raw_10k",
        processed_dir: str | Path = "data/processed_chunks",
        chunk_size: int = 600,
        chunk_overlap: int = 90,
        min_chunk_size: int = 50,
        company_name: str = "financial_graphrag",
        email: str = "user@example.com",
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.downloader = SEC10KDownloader(
            company_name=company_name,
            email=email,
            raw_data_dir=self.raw_data_dir,
        )
        self.parser = SEC10KParser()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    def run(
        self,
        ticker: str,
        year: int,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Chunk]:
        logger.info("Ingestion pipeline started for %s / %s", ticker, year)

        if use_cache:
            cached = self._load_cached_chunks(ticker, year)
            if cached:
                logger.info(
                    "Reusing %d cached chunks for %s / %s from %s",
                    len(cached),
                    ticker,
                    year,
                    self.processed_dir / f"{ticker}_{year}" / "chunks.json",
                )
                return cached

        files = self.downloader.download(ticker, year, after_date, before_date)
        if not files:
            logger.warning("No files downloaded for %s / %s", ticker, year)
            return []

        all_chunks: List[Chunk] = []
        for file_path in files:
            try:
                raw_text = self.parser.extract_text(file_path)
                md_text = self.parser.convert_to_markdown(raw_text)
                sections = self.parser.extract_sections(md_text, ticker, year)
                chunks = self.chunker.chunk_sections(sections)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(
                    "Failed to process %s: %s", file_path, exc
                )
                continue

        self._persist_chunks(all_chunks, ticker, year)
        logger.info(
            "Ingestion pipeline completed for %s / %s: %d chunks",
            ticker,
            year,
            len(all_chunks),
        )
        return all_chunks

    def _load_cached_chunks(
        self,
        ticker: str,
        year: int,
    ) -> List[Chunk]:
        output_path = self.processed_dir / f"{ticker}_{year}" / "chunks.json"
        if not output_path.exists():
            return []

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            return [Chunk(**r) for r in records]
        except Exception as exc:
            logger.warning(
                "Failed to load cached chunks from %s: %s", output_path, exc
            )
            return []

    def _persist_chunks(
        self,
        chunks: List[Chunk],
        ticker: str,
        year: int,
    ) -> None:
        output_dir = self.processed_dir / f"{ticker}_{year}"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = [c.to_dict() for c in chunks]
        output_path = output_dir / "chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        logger.info("Persisted %d chunks to %s", len(records), output_path)
