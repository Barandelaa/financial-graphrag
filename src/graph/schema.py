from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import kuzu

logger = logging.getLogger(__name__)

NODE_TABLE_DDL: List[str] = [
    "CREATE NODE TABLE IF NOT EXISTS Company ("
    "  ticker STRING PRIMARY KEY,"
    "  name STRING,"
    "  sector STRING"
    ")",
    "CREATE NODE TABLE IF NOT EXISTS FinancialMetric ("
    "  id STRING PRIMARY KEY,"
    "  name STRING,"
    "  value STRING,"
    "  unit STRING,"
    "  fiscal_year INT64"
    ")",
    "CREATE NODE TABLE IF NOT EXISTS RiskFactor ("
    "  id STRING PRIMARY KEY,"
    "  name STRING,"
    "  category STRING,"
    "  description STRING"
    ")",
    "CREATE NODE TABLE IF NOT EXISTS BusinessSegment ("
    "  id STRING PRIMARY KEY,"
    "  name STRING,"
    "  description STRING"
    ")",
    "CREATE NODE TABLE IF NOT EXISTS MacroEvent ("
    "  id STRING PRIMARY KEY,"
    "  name STRING,"
    "  date STRING,"
    "  description STRING"
    ")",
    "CREATE NODE TABLE IF NOT EXISTS DocumentChunk ("
    "  chunk_id STRING PRIMARY KEY,"
    "  text STRING,"
    "  company_ticker STRING,"
    "  fiscal_year INT64,"
    "  section_id STRING,"
    "  page_number INT64"
    ")",
]

REL_TABLE_DDL: List[str] = [
    "CREATE REL TABLE IF NOT EXISTS OPERATES_IN ("
    "  FROM Company TO BusinessSegment"
    ")",
    "CREATE REL TABLE IF NOT EXISTS REPORTED_METRIC ("
    "  FROM Company TO FinancialMetric"
    ")",
    "CREATE REL TABLE IF NOT EXISTS IMPACTS_REVENUE ("
    "  FROM RiskFactor TO FinancialMetric"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MITIGATES_RISK ("
    "  FROM BusinessSegment TO RiskFactor"
    ")",
    "CREATE REL TABLE IF NOT EXISTS COMPETES_WITH ("
    "  FROM Company TO Company"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS_COMPANY ("
    "  FROM DocumentChunk TO Company"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS_METRIC ("
    "  FROM DocumentChunk TO FinancialMetric"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS_RISK ("
    "  FROM DocumentChunk TO RiskFactor"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS_SEGMENT ("
    "  FROM DocumentChunk TO BusinessSegment"
    ")",
    "CREATE REL TABLE IF NOT EXISTS MENTIONS_EVENT ("
    "  FROM DocumentChunk TO MacroEvent"
    ")",
]


@dataclass
class GraphConfig:
    db_path: str | Path = "data/graph/kuzu_db"
    buffer_pool_size: int = 1024**3
    max_threads: int = 4


class GraphSchema:
    def __init__(self, config: GraphConfig) -> None:
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: kuzu.Database = kuzu.Database(
            str(self.db_path),
            buffer_pool_size=config.buffer_pool_size,
            max_threads=config.max_threads,
        )
        self._connection: kuzu.Connection = kuzu.Connection(self._db)
        self._create_schema()

    @property
    def connection(self) -> kuzu.Connection:
        return self._connection

    @property
    def db(self) -> kuzu.Database:
        return self._db

    def _create_schema(self) -> None:
        for ddl in NODE_TABLE_DDL:
            try:
                self._connection.execute(ddl)
            except RuntimeError as exc:
                logger.warning("DDL node table error (may already exist): %s", exc)
        for ddl in REL_TABLE_DDL:
            try:
                self._connection.execute(ddl)
            except RuntimeError as exc:
                logger.warning("DDL rel table error (may already exist): %s", exc)
        logger.info("Kùzu schema initialised at %s", self.db_path)

    def close(self) -> None:
        self._connection.close()
        self._db.close()
