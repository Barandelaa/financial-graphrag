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

        # Table-aware: detect markdown table rows (contain " | ") y evita partirlos
        lines = raw_text.splitlines()
        # Si el texto contiene tablas, chunk por líneas preservando filas
        has_table = any(" | " in l and l.count("|") >= 2 for l in lines)
        if has_table:
            return self._chunk_table_aware(section, lines)

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

    def _chunk_table_aware(self, section: Section, lines: List[str]) -> List[Chunk]:
        """Mantiene bloques de tablas intactos; si la tabla no cabe, parte por filas."""
        chunks: List[Chunk] = []
        cur_lines: List[str] = []
        cur_tokens = 0
        seq = 0

        def flush():
            nonlocal cur_lines, cur_tokens, seq
            if not cur_lines:
                return
            text = "\n".join(cur_lines).strip()
            tokens = self._tokenize(text)
            if len(tokens) >= self.min_chunk_size:
                ck = self._build_chunk(section, tokens, seq)
                if ck:
                    chunks.append(ck)
                    seq += 1
            cur_lines = []
            cur_tokens = 0

        in_table = False
        table_buf: List[str] = []
        table_tokens = 0

        for line in lines:
            is_row = " | " in line and line.count("|") >= 2
            line_tokens = len(self._tokenize(line))

            if is_row:
                if not in_table:
                    # flush prose before table
                    if cur_lines:
                        flush()
                    in_table = True
                    table_buf = []
                    table_tokens = 0
                table_buf.append(line)
                table_tokens += line_tokens
                # si la tabla supera chunk_size, flush por filas
                if table_tokens >= self.chunk_size:
                    # flush tabla acumulada como chunk
                    text = "\n".join(table_buf).strip()
                    tokens = self._tokenize(text)
                    if len(tokens) >= self.min_chunk_size:
                        ck = self._build_chunk(section, tokens, seq)
                        if ck:
                            chunks.append(ck)
                            seq += 1
                    table_buf = []
                    table_tokens = 0
            else:
                if in_table:
                    # cierra tabla: decide si cabe con cur o flush separado
                    tbl_text = "\n".join(table_buf).strip()
                    tbl_tok = len(self._tokenize(tbl_text))
                    if cur_tokens + tbl_tok <= self.chunk_size and cur_tokens > 0:
                        cur_lines.extend(table_buf)
                        cur_tokens += tbl_tok
                    else:
                        if cur_lines:
                            flush()
                        # tabla sola como chunk si es sustancial
                        if tbl_tok >= self.min_chunk_size:
                            ck = self._build_chunk(section, self._tokenize(tbl_text), seq)
                            if ck:
                                chunks.append(ck)
                                seq += 1
                        else:
                            cur_lines.extend(table_buf)
                            cur_tokens += tbl_tok
                    table_buf = []
                    table_tokens = 0
                    in_table = False
                # prose line
                if cur_tokens + line_tokens > self.chunk_size and cur_lines:
                    flush()
                    # overlap: re-add last overlap tokens as lines si es posible
                cur_lines.append(line)
                cur_tokens += line_tokens
                if cur_tokens >= self.chunk_size:
                    flush()

        # trailing
        if in_table and table_buf:
            tbl_text = "\n".join(table_buf).strip()
            if cur_lines and cur_tokens + len(self._tokenize(tbl_text)) <= self.chunk_size:
                cur_lines.extend(table_buf)
            else:
                if cur_lines:
                    flush()
                ck = self._build_chunk(section, self._tokenize(tbl_text), seq)
                if ck:
                    chunks.append(ck)
                    seq += 1
        if cur_lines:
            flush()
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
