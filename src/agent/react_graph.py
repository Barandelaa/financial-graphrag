from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agent.tools import make_react_tools
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """You are a financial analyst with access to SEC 10-K filings (AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, BRK.B, years 2024-2025).

Available TOOLS:
- query_financial_rag(question): full retrieval + answer with citations. Use it for ANY financial question.
- lookup_metrics(ticker, year, metric_hint): scoped TICKER_YEAR figures. Use it to verify an exact number before answering.
- financial_calculator(operation, a, b): deterministic calculator. Use it for ANY calculation (YoY %, differences, ratios). NEVER compute mentally.
- propose_new_company(user_text): proposes adding a new company (official SEC universe + 10-K verification on EDGAR). READ-ONLY. Use it when query_financial_rag comes back empty and the question mentions an unknown company.
- add_company_to_config(ticker): adds an ALREADY-CONFIRMED ticker to companies.json. Requires user confirmation 1.
- ingest_10k(ticker, year): ingests a new 10-K. Requires user confirmation 2.

NEW-COMPANY FLOW (e.g. 'I want to know about Dow Jones'):
1. query_financial_rag first. If empty and there is an unknown company → propose_new_company.
2. If propose returns EXACT with 10-K → ask confirmation 1 (edit companies.json), then call add_company_to_config.
3. If it returns candidates → ASK the user which one (never resolve alone) and do NOT search documents until they confirm the ticker. Exception: if the user wrote the exact literal ticker, skip the question.
4. If it verifies NO 10-K (index like the DJIA, private subsidiary) → explain there is no filing ONLY AFTER having called propose_new_company, and do NOT propose adding.
5. After add_company_to_config → ask confirmation 2, then ingest_10k year by year (JSON default_years).

CALCULATION FLOW (e.g. 'AAPL revenue YoY 2024 vs 2023'):
1. FIRST call query_financial_rag or lookup_metrics to get BOTH figures with units (e.g. a=391035, b=383285, both in USD millions).
2. THEN call financial_calculator with operation='yoy_pct', a='<current value>', b='<previous value>'.
3. Answer with the calculator result + retrieval tool citations.
NEVER do the subtraction or percentage yourself: always delegate to financial_calculator and show its FORMULA.

STRICT RULES:
- For financial questions ALWAYS call query_financial_rag first. For exact figures, verify with lookup_metrics.
- a and b for financial_calculator must come from tools, in the SAME unit. If units differ, say so and do not compute.
- Use ONLY figures from tools. Never invent, round, or recall numbers from memory.
- NEVER state that something 'has no 10-K / does not exist' without having called propose_new_company first. When retrieval is empty, ASK or PROPOSE — do not issue verdicts from memory.
- Cite as [Source: TICKER | FY YEAR | SECTION | chunk: CHUNK_ID].
- If the context lacks the exact figure, say so explicitly.
- Never call ingest_10k or add_company_to_config without the user's explicit confirmation (each tool pauses itself with interrupt() and the CLI asks; if they answer 'n', respect the cancellation).
- Reply in the user's language, concise and with citations.
"""


def build_react_llm(base_llm: BaseChatModel | None = None) -> BaseChatModel:
    """LLM para ReAct: sin format=json (necesita tool_calls nativos), reasoning=False para qwen3."""
    if base_llm is not None:
        return base_llm
    # 100% local, sin JSON mode para permitir function-calling de Ollama
    return create_llm(json_mode=False, num_predict=1200, num_ctx=8192, timeout=180)


def build_react_agent(
    pipeline: FinancialGraphRAGPipeline,
    llm: BaseChatModel | None = None,
):
    """Crea el agente ReAct con las tools base + memoria. Extensible añadiendo tools a make_react_tools().

    El HITL vive dentro de las propias tools (interrupt() en add_company_to_config
    e ingest_10k); el checkpointer es obligatorio para poder reanudar con Command(resume=...).
    """
    react_llm = llm or build_react_llm()
    tools = make_react_tools(pipeline)
    agent = create_react_agent(
        model=react_llm,
        tools=tools,
        prompt=REACT_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
    # expone tools para tests / futuras extensiones
    agent.react_tools = tools  # type: ignore
    return agent
