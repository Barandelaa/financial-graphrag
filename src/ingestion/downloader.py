from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


class SEC10KDownloader:
    TICKER_CIK_ALIASES = {
        "BRK.B": "0001067983",
    }

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

        self.cache_manifest_path = self.raw_data_dir / ".download_cache.json"

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
        cache_key = self._cache_key(ticker, year, after_date, before_date)
        cached = self._load_cache()
        if cache_key in cached:
            cached_paths = self._valid_cached_paths(cached[cache_key])
            if cached_paths:
                logger.info(
                    "Reusing %d cached raw file(s) for %s / %s from disk cache",
                    len(cached_paths),
                    ticker,
                    year,
                )
                return cached_paths

        try:
            ticker_or_cik = self.TICKER_CIK_ALIASES.get(ticker, ticker)
            result = self._downloader.get(
                form="10-K",
                ticker_or_cik=ticker_or_cik,
                after=after_date or f"{year}-01-01",
                before=before_date or f"{year}-12-31",
            )
        except Exception as exc:
            logger.error("Failed to download 10-K for %s: %s", ticker, exc)
            raise

        downloaded: List[Path] = []
        ticker_dir = (
            self.raw_data_dir
            / "sec-edgar-filings"
            / self.TICKER_CIK_ALIASES.get(ticker, ticker)
            / "10-K"
        )
        if ticker_dir.exists():
            for filing_dir in sorted(ticker_dir.iterdir()):
                for f in filing_dir.rglob("*"):
                    if f.suffix.lower() in {".pdf", ".html", ".htm", ".txt"}:
                        downloaded.append(f)
        if not downloaded:
            logger.warning("No 10-K files found for %s/%s", ticker, year)
            return []

        cached[cache_key] = [str(p) for p in downloaded]
        self._save_cache(cached)
        logger.info("Cached %d raw file(s) for %s / %s", len(downloaded), ticker, year)
        return downloaded

    @staticmethod
    def _cache_key(ticker: str, year: int, after_date: Optional[str], before_date: Optional[str]) -> str:
        return f"{ticker}|{year}|{after_date or ''}|{before_date or ''}"

    def _load_cache(self) -> dict:
        if not self.cache_manifest_path.exists():
            return {}
        try:
            with open(self.cache_manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read download cache %s: %s", self.cache_manifest_path, exc)
            return {}

    def _save_cache(self, cache: dict) -> None:
        try:
            with open(self.cache_manifest_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write download cache %s: %s", self.cache_manifest_path, exc)

    def _valid_cached_paths(self, paths: List[str]) -> List[Path]:
        return [Path(p) for p in paths if p and Path(p).exists()]

    def clean_raw_data(self) -> None:
        shutil.rmtree(self.raw_data_dir / "sec-edgar-filings", ignore_errors=True)
        if self.cache_manifest_path.exists():
            self.cache_manifest_path.unlink()
        logger.info("Cleaned raw SEC EDGAR filings directory and download cache")
