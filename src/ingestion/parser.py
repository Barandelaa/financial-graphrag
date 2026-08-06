from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pdfplumber
from markdownify import markdownify as md

logger = logging.getLogger(__name__)

SEC_SECTION_PATTERN = re.compile(
    r"(?i)(item\s*\d+[a-zA-Z]?(\.\d+)?(\s*(–|-|—|\.)\s*.*?)?)",
)

SEC_KNOWN_ITEMS = [
    "Item 1",
    "Item 1A",
    "Item 1B",
    "Item 2",
    "Item 3",
    "Item 4",
    "Item 5",
    "Item 6",
    "Item 7",
    "Item 7A",
    "Item 8",
    "Item 9",
    "Item 9A",
    "Item 9B",
    "Item 10",
    "Item 11",
    "Item 12",
    "Item 13",
    "Item 14",
    "Item 15",
    "Item 16",
]

TOC_KEYWORDS = [
    "table of contents",
    "index",
    "signatures",
    "exhibit index",
    "exhibits",
]


class SEC10KParser:
    def extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in {".html", ".htm"}:
            return self._extract_html(file_path)
        elif suffix == ".txt":
            return self._read_text(file_path)
        else:
            msg = f"Unsupported file format: {suffix}"
            raise ValueError(msg)

    def convert_to_markdown(self, text: str) -> str:
        cleaned = self._clean_html_tables(text)
        return md(cleaned, heading_style="ATX", strip=["script", "style"])

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return text.replace("\xa0", " ")

    @staticmethod
    def _clean_html_tables(text: str) -> str:
        if "<table" not in text.lower() and "<a " not in text.lower():
            return text
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            for tag in soup.find_all(["script", "style"]):
                tag.decompose()

            for a in soup.find_all("a"):
                for attr in list(a.attrs):
                    if attr in {"href", "name", "id"}:
                        del a[attr]

            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    cells = []
                    for cell in tr.find_all(["td", "th"]):
                        cell_text = cell.get_text(" ", strip=True)
                        cell_text = re.sub(r"\s+", " ", cell_text).strip()
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        rows.append(" | ".join(cells))
                table.replace_with("\n".join(rows) + "\n")

            return soup.get_text("\n")
        except ImportError:
            logger.warning("bs4 not available; skipping HTML table cleaning")
            return text

    def extract_sections(
        self,
        markdown_text: str,
        ticker: str,
        fiscal_year: int,
    ) -> List[dict]:
        markdown_text = self._normalize_whitespace(markdown_text)
        lines = markdown_text.splitlines()
        sections: List[dict] = []
        current_section: Optional[str] = None
        current_content: List[str] = []
        page_num = 1

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            if self._is_toc_row(stripped):
                continue

            matched_section = self._match_section_header(stripped)
            if matched_section:
                if current_section and current_content:
                    sections.append(
                        self._build_section(
                            ticker,
                            fiscal_year,
                            current_section,
                            current_content,
                            page_num,
                        )
                    )
                current_section = matched_section
                current_content = []
            else:
                current_content.append(line)

            if stripped.startswith("###") and stripped.lower() in TOC_KEYWORDS:
                pass

        if current_section and current_content:
            sections.append(
                self._build_section(
                    ticker,
                    fiscal_year,
                    current_section,
                    current_content,
                    page_num,
                )
            )

        return sections

    def _extract_pdf(self, file_path: Path) -> str:
        text_parts: List[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as exc:
            logger.error("Failed to extract PDF %s: %s", file_path, exc)
            raise
        return "\n".join(text_parts)

    def _extract_html(self, file_path: Path) -> str:
        try:
            from unstructured.partition.html import partition_html

            elements = partition_html(str(file_path))
            return "\n".join(str(el) for el in elements)
        except ImportError:
            logger.warning(
                "unstructured not available, falling back to raw text"
            )
            return self._read_text(file_path)
        except Exception as exc:
            logger.error("Failed to parse HTML %s: %s", file_path, exc)
            raise

    @staticmethod
    def _read_text(file_path: Path) -> str:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            raise
        return SEC10KParser._extract_main_document(text)

    @staticmethod
    def _extract_main_document(text: str) -> str:
        if "<TEXT>" not in text.upper():
            return text

        for block in re.split(r"<DOCUMENT>", text, flags=re.IGNORECASE):
            type_match = re.search(
                r"<TYPE>(.*?)(?:\n|</TYPE>)",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not type_match:
                continue
            doc_type = type_match.group(1).strip().upper()
            if doc_type != "10-K" and not doc_type.startswith("10-K/"):
                continue
            blocks = re.findall(
                r"<TEXT>(.*?)</TEXT>",
                block,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if blocks:
                main = max(blocks, key=len)
                logger.info(
                    "Extracted 10-K narrative from SEC full-submission "
                    "(%d of %d <TEXT> blocks)",
                    len(main),
                    len(blocks),
                )
                return main

        blocks = re.findall(
            r"<TEXT>(.*?)</TEXT>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not blocks:
            return text
        main = max(blocks, key=len)
        logger.info(
            "Extracted main document from SEC full-submission "
            "(%d of %d <TEXT> blocks)",
            len(main),
            len(blocks),
        )
        return main

    @staticmethod
    def _is_toc_row(line: str) -> bool:
        return bool(
            re.match(
                r"(?i)^\s*(#+\s*)?item\s+\d+[a-z]?(\.\d+)?\s*\.?\s*\|",
                line,
            )
        )

    @staticmethod
    def _match_section_header(line: str) -> Optional[str]:
        for item in SEC_KNOWN_ITEMS:
            pattern = re.compile(
                rf"^\s*#+\s*{re.escape(item)}\b", re.IGNORECASE
            )
            if pattern.match(line):
                return item
            alt = re.compile(rf"^\s*{re.escape(item)}\b", re.IGNORECASE)
            if alt.match(line):
                return item
        return None

    @staticmethod
    def _build_section(
        ticker: str,
        fiscal_year: int,
        section_id: str,
        content_lines: List[str],
        page_number: int,
    ) -> dict:
        return {
            "company_ticker": ticker.upper(),
            "fiscal_year": fiscal_year,
            "section_id": section_id,
            "page_number": page_number,
            "content": "\n".join(content_lines).strip(),
        }
