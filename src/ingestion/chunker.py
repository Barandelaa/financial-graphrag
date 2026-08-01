from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    company_ticker: str
    fiscal_year: int
    section_id: str
    page_number: int
    text: str
    token_count: int
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Section:
    company_ticker: str
    fiscal_year: int
    section_id: str
    page_number: int
    content: str


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 90,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_sections(self, sections: List[dict]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for sec in sections:
            section_obj = Section(**sec)
            section_chunks = self._chunk_single_section(section_obj)
            chunks.extend(section_chunks)
        logger.info("Produced %d chunks from %d sections", len(chunks), len(sections))
        return chunks

    def _chunk_single_section(self, section: Section) -> List[Chunk]:
        raw_text = section.content
        if not raw_text:
            return []

        tokens = self._tokenize(raw_text)
        total = len(tokens)

        if total <= self.chunk_size:
            single = self._build_chunk(section, tokens, 0)
            return [single] if single else []

        chunks: List[Chunk] = []
        start = 0
        seq = 0

        while start < total:
            end = start + self.chunk_size
            segment = tokens[start:end]
            chunk_obj = self._build_chunk(section, segment, seq)
            if chunk_obj:
                chunks.append(chunk_obj)
                seq += 1

            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    def _build_chunk(
        self,
        section: Section,
        tokens: List[str],
        seq: int,
    ) -> Optional[Chunk]:
        text = " ".join(tokens).strip()
        token_count = len(tokens)

        if not text or token_count < self.min_chunk_size:
            return None

        chunk_id = str(uuid.uuid4())

        return Chunk(
            chunk_id=chunk_id,
            company_ticker=section.company_ticker,
            fiscal_year=section.fiscal_year,
            section_id=section.section_id,
            page_number=section.page_number,
            text=text,
            token_count=token_count,
            metadata={
                "chunk_seq": str(seq),
                "source": f"{section.company_ticker}_{section.fiscal_year}",
            },
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.split()
