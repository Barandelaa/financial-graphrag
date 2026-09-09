from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.reranker import RerankedResult

logger = logging.getLogger(__name__)

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial analyst assistant. Answer the user's question "
            "using ONLY the provided CONTEXT (SEC 10-K report excerpts), "
            "GRAPH FACTS and METRICS TABLE (scoped by ticker/year).\n"
            "For each factual claim, cite the source using the format:\n"
            "[Source: TICKER | FY YEAR | SECTION | chunk: CHUNK_ID]\n\n"
            "STRICT RULES:\n"
            "- Use ONLY figures, dates and facts explicitly present in the "
            "CONTEXT, GRAPH FACTS or METRICS TABLE. Never invent, round, or recall numbers "
            "from memory.\n"
            "- METRICS TABLE rows are already scoped as | ticker | year | metric | value | - "
            "use ONLY the row whose ticker/year matches the question. Never mix AAPL 2024 revenue with MSFT 2024 revenue.\n"
            "- When a number appears in the context as a table row like "
            "'Total net sales | $ | 383,285', report it exactly as written and cite its chunk_id.\n"
            "- If the exact figure asked is NOT in the context/metrics, say so "
            "explicitly instead of guessing.\n"
            "- If the context does not contain enough information, say so.\n"
            "- Do NOT make up information.",
        ),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
            "GRAPH FACTS:\n{graph_facts}\n\n"
            "METRICS TABLE:\n{metrics_table}\n\n"
            "QUESTION: {question}\n\n"
            "Answer with precise citations:",
        ),
    ]
)


@dataclass
class GenerationResponse:
    answer: str
    citations: List[dict] = field(default_factory=list)


class ResponseGenerator:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        context: List[RerankedResult],
        graph_facts: Optional[List[str]] = None,
        metrics_table: Optional[str] = None,
    ) -> GenerationResponse:
        context_blocks = []
        citations: List[dict] = []

        for i, r in enumerate(context):
            meta = r.metadata
            ticker = meta.get("company_ticker", "N/A")
            year = meta.get("fiscal_year", "N/A")
            section = meta.get("section_id", "N/A")

            block = (
                f"[{i+1}] (Ticker: {ticker} | FY: {year} | "
                f"Section: {section} | Chunk: {r.chunk_id})\n"
                f"{r.text}\n"
            )
            context_blocks.append(block)

            citations.append({
                "chunk_id": r.chunk_id,
                "company_ticker": ticker,
                "fiscal_year": year,
                "section_id": section,
                "relevance_score": r.score,
            })

        full_context = "\n---\n".join(context_blocks)
        facts_block = "\n".join(
            f"- {f}" for f in (graph_facts or [])
        ) or "(no graph facts retrieved)"
        metrics_block = (metrics_table or "").strip() or "(no scoped metrics retrieved)"

        try:
            response = self.llm.invoke(
                GENERATION_PROMPT.format_messages(
                    context=full_context,
                    graph_facts=facts_block,
                    metrics_table=metrics_block,
                    question=question,
                )
            )
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            answer = "Error generating response."

        return GenerationResponse(answer=answer, citations=citations)
