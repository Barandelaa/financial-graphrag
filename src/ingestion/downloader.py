from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


class SEC10KDownloader:
    def __init__(
        self,
        company_name: str = "financial_graphrag",
        email: str = "user@example.com",
        raw_data_dir: str | Path = "data/raw_10k",
    ) -> None:
        self.company_name = company_name
        self.email = email
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self._downloader = Downloader(
            self.company_name,
            self.email,
            self.raw_data_dir,
        )

    def download(
        self,
        ticker: str,
        year: int,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None,
    ) -> List[Path]:
        logger.info(
            "Downloading 10-K for %s | year=%s | after=%s | before=%s",
            ticker,
            year,
            after_date,
            before_date,
        )
        try:
            result = self._downloader.get(
                filing_type="10-K",
                ticker_or_cik=ticker,
                after_date=after_date or f"{year}-01-01",
                before_date=before_date or f"{year}-12-31",
            )
        except Exception as exc:
            logger.error("Failed to download 10-K for %s: %s", ticker, exc)
            raise

        downloaded: List[Path] = []
        ticker_dir = self.raw_data_dir / "sec-edgar-filings" / ticker / "10-K"
        if ticker_dir.exists():
            for filing_dir in sorted(ticker_dir.iterdir()):
                for f in filing_dir.rglob("*"):
                    if f.suffix.lower() in {".pdf", ".html", ".htm", ".txt"}:
                        downloaded.append(f)
        if not downloaded:
            logger.warning("No 10-K files found for %s/%s", ticker, year)

        return downloaded

    def clean_raw_data(self) -> None:
        shutil.rmtree(self.raw_data_dir / "sec-edgar-filings", ignore_errors=True)
        logger.info("Cleaned raw SEC EDGAR filings directory")
