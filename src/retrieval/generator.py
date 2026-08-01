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
            "using ONLY the provided context from SEC 10-K reports. "
            "For each factual claim, cite the source using the format:\n"
            "[Source: {ticker} | FY{year} | {section} | chunk: {chunk_id}]\n\n"
            "If the context does not contain enough information, say so.\n"
            "Do NOT make up information.",
        ),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
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

        try:
            response = self.llm.invoke(
                GENERATION_PROMPT.format_messages(
                    context=full_context,
                    question=question,
                )
            )
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            answer = "Error generating response."

        return GenerationResponse(answer=answer, citations=citations)
