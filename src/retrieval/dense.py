from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _detect_device(preferred: str) -> str:
    if preferred and preferred != "cpu":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class DenseSearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class DenseRetriever:
    def __init__(
        self,
        db_uri: str | Path = "data/vector_store/lancedb",
        model_name: str = "BAAI/bge-m3",
        table_name: str = "chunks",
        device: str = "cpu",
    ) -> None:
        self.db_uri = str(db_uri)
        self.model_name = model_name
        self.table_name = table_name
        self._embedder: Optional[SentenceTransformer] = None
        self._device = _detect_device(device)
        self._scalar_indices_ready = False

        Path(db_uri).parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.db_uri)

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            from src.env import get_hf_token

            kwargs = {"device": self._device}
            token = get_hf_token()
            if token:
                kwargs["token"] = token
            self._embedder = SentenceTransformer(self.model_name, **kwargs)
        return self._embedder

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # batch_size explícito: evita picos de VRAM en GPU 12GB con chunks largos.
        return self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

    @staticmethod
    def build_ticker_filter(tickers: set[str] | List[str] | None) -> Optional[str]:
        """Construye un WHERE seguro para prefilter en LanceDB.

        Solo permite [A-Z0-9.-] para evitar inyección SQL.
        """
        if not tickers:
            return None
        clean = sorted({t.upper() for t in tickers if t and isinstance(t, str)})
        clean = [t for t in clean if t.replace(".", "").replace("-", "").isalnum()]
        if not clean:
            return None
        quoted = ", ".join(f"'{t}'" for t in clean)
        return f"company_ticker IN ({quoted})"

    def ensure_scalar_indices(self) -> None:
        """Crea índices BTREE para filtros (best-effort, una vez por proceso).

        Acelera los WHERE con prefilter=True sin cambiar el plan de búsqueda
        vectorial (seguimos en flat/exacto: N pequeño, recall 100%).
        """
        if self._scalar_indices_ready:
            return
        try:
            table = self._db.open_table(self.table_name)
        except Exception:
            return
        btree_cls = None
        try:
            from lancedb.index import BTree

            btree_cls = BTree
        except ImportError:
            pass

        success_count = 0
        for col in ("company_ticker", "fiscal_year", "section_id", "chunk_id"):
            try:
                if btree_cls is not None:
                    table.create_index(col, config=btree_cls(), replace=True)
                else:
                    table.create_scalar_index(col, index_type="BTREE", replace=True)
                success_count += 1
            except Exception as exc:
                logger.debug("Scalar index on %s skipped: %s", col, exc)
        if success_count > 0:
            self._scalar_indices_ready = True

    def index_chunks(
        self,
        chunks: List[dict],
        persist: bool = True,
    ) -> None:
        if not chunks:
            logger.info("No chunks to index")
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts)

        records = []
        for chunk, emb in zip(chunks, embeddings):
            records.append({
                "vector": emb.tolist(),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "company_ticker": chunk["company_ticker"],
                "fiscal_year": chunk["fiscal_year"],
                "section_id": chunk["section_id"],
                "page_number": chunk["page_number"],
                **chunk.get("metadata", {}),
            })

        if self.table_name in self._db.table_names():
            existing_ids = self._existing_chunk_ids()
            new_records = [
                r for r in records if r["chunk_id"] not in existing_ids
            ]
            if new_records:
                self._db.open_table(self.table_name).add(new_records)
                logger.info(
                    "Added %d new chunks to existing LanceDB table '%s'",
                    len(new_records),
                    self.table_name,
                )
            else:
                logger.info(
                    "All %d chunks already indexed in '%s'; skipping",
                    len(records),
                    self.table_name,
                )
            self.ensure_scalar_indices()
            return

        table = self._db.create_table(
            self.table_name, data=records, mode="overwrite"
        )
        logger.info(
            "Indexed %d chunks in LanceDB table '%s'", len(records), self.table_name
        )
        self.ensure_scalar_indices()

    def _existing_chunk_ids(self) -> set[str]:
        try:
            table = self._db.open_table(self.table_name)
        except Exception:
            return set()
        # Proyección solo chunk_id: evita cargar la columna vector (1024 floats/fila).
        try:
            rows = table.search().select(["chunk_id"]).limit(1_000_000).to_list()
            return {r["chunk_id"] for r in rows if r.get("chunk_id")}
        except Exception:
            pass
        try:
            df = table.to_pandas()
            return set(df["chunk_id"]) if len(df) else set()
        except Exception as exc:
            logger.warning("Could not read existing chunk ids: %s", exc)
            return set()

    def load_text_chunks(self) -> List[dict]:
        """Carga chunks sin la columna vector (para BM25 / sparse).

        Evita traer ~1024 floats por fila a memoria; en 32GB RAM es la
        diferencia entre un rebuild de ms y cientos de MB innecesarios.
        """
        try:
            table = self._db.open_table(self.table_name)
        except Exception as exc:
            logger.warning("Cannot open table '%s': %s", self.table_name, exc)
            return []
        columns = [
            "chunk_id", "text", "company_ticker",
            "fiscal_year", "section_id", "page_number",
        ]
        try:
            return table.search().select(columns).limit(1_000_000).to_list()
        except Exception as exc:
            logger.debug("Projected scan failed, fallback to pandas: %s", exc)
        try:
            df = table.to_pandas()
        except Exception as exc:
            logger.warning("Could not read LanceDB table: %s", exc)
            return []
        if not len(df):
            return []
        records = df.to_dict("records")
        for r in records:
            r.pop("vector", None)
        return records

    def load_all_chunks(self, include_vector: bool = False) -> List[dict]:
        """Carga chunks de LanceDB.

        Si include_vector=False (por defecto), delega a load_text_chunks() y omite
        los vectores (1024 floats) ahorrando cientos de MB de RAM (ej. para BM25).
        Si include_vector=True, carga todos los campos incluyendo vector y metadatos.
        """
        if not include_vector:
            return self.load_text_chunks()
        try:
            table = self._db.open_table(self.table_name)
            df = table.to_pandas()
            return df.to_dict("records") if len(df) else []
        except Exception as exc:
            logger.warning("Could not read full LanceDB table: %s", exc)
            return []

    def search(
        self,
        query: str,
        top_k: int = 20,
        tickers: set[str] | List[str] | None = None,
        filters: Optional[str] = None,
    ) -> List[DenseSearchResult]:
        """Búsqueda densa con prefilter opcional por ticker.

        Con N pequeño usamos flat/exacto (sin índice ANN): recall 100% y
        latencia despreciable. El filtro se aplica ANTES del scan vectorial
        (prefilter=True) para no desperdiciar top_k en otras empresas.
        No se usa HNSW/IVF: innecesario hasta ~50k+ vectores.
        """
        query_vector = self.embed_query(query)

        try:
            table = self._db.open_table(self.table_name)
        except Exception as exc:
            logger.error("Cannot open table '%s': %s", self.table_name, exc)
            return []

        where = filters or self.build_ticker_filter(tickers)
        try:
            builder = table.search(query_vector.tolist())
            if where:
                builder = builder.where(where, prefilter=True)
            results = builder.limit(top_k).to_list()
        except Exception as exc:
            logger.warning(
                "Prefiltered LanceDB search failed with filter %r (%s); retrying without filter",
                where,
                exc,
            )
            results = (
                table.search(query_vector.tolist())
                .limit(top_k)
                .to_list()
            )

        return [
            DenseSearchResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r.get("_distance", 0.0),
                metadata={
                    "company_ticker": r.get("company_ticker", ""),
                    "fiscal_year": r.get("fiscal_year", ""),
                    "section_id": r.get("section_id", ""),
                    "page_number": r.get("page_number", ""),
                },
            )
            for r in results
        ]

    def close(self) -> None:
        try:
            self._db.close()
        except AttributeError:
            logger.debug("LanceDB connection has no close() method; skipping")
