from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert financial analyst extracting structured knowledge "
            "from SEC 10-K reports. Extract all financial entities and their "
            "relationships from the text below. Use ONLY the ontology provided.\n\n"
            "ENTITY TYPES:\n"
            "- Company: a public corporation identified by ticker\n"
            "- FinancialMetric: a numeric metric (revenue, net income, EPS, etc.)\n"
            "- RiskFactor: a disclosed risk or uncertainty\n"
            "- BusinessSegment: an operating or reportable segment\n"
            "- MacroEvent: a macroeconomic or geopolitical event\n\n"
            "RELATIONSHIP TYPES:\n"
            "- operates_in: Company -> BusinessSegment\n"
            "- reported_metric: Company -> FinancialMetric\n"
            "- impacts_revenue: RiskFactor/MacroEvent -> FinancialMetric\n"
            "- mitigates_risk: BusinessSegment -> RiskFactor\n"
            "- competes_with: Company -> Company\n\n"
            "Respond ONLY with a JSON array of triplets matching the schema.",
        ),
        (
            "human",
            "TEXT: {chunk_text}\n\n"
            "COMPANY TICKER: {ticker}\n"
            "FISCAL YEAR: {year}\n"
            "SECTION: {section_id}",
        ),
    ]
)


class EntityType(str, Enum):
    company = "Company"
    financial_metric = "FinancialMetric"
    risk_factor = "RiskFactor"
    business_segment = "BusinessSegment"
    macro_event = "MacroEvent"


class RelationType(str, Enum):
    operates_in = "operates_in"
    reported_metric = "reported_metric"
    impacts_revenue = "impacts_revenue"
    mitigates_risk = "mitigates_risk"
    competes_with = "competes_with"


class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    entity_type: EntityType = Field(description="Type of the entity")
    properties: dict = Field(
        default_factory=dict,
        description="Additional properties (e.g. value, unit, sector)",
    )


class Triplet(BaseModel):
    source_entity: Entity = Field(description="Source entity")
    relation: RelationType = Field(description="Relationship type")
    target_entity: Entity = Field(description="Target entity")


class ExtractionResult(BaseModel):
    triplets: List[Triplet] = Field(
        description="List of extracted knowledge triplets"
    )


class FinancialTriplet:
    def __init__(
        self,
        triplet: Triplet,
        chunk_id: str,
        company_ticker: str,
        fiscal_year: int,
        section_id: str,
    ) -> None:
        self.triplet = triplet
        self.chunk_id = chunk_id
        self.company_ticker = company_ticker
        self.fiscal_year = fiscal_year
        self.section_id = section_id

    @property
    def source_name(self) -> str:
        return self.triplet.source_entity.name

    @property
    def target_name(self) -> str:
        return self.triplet.target_entity.name

    @property
    def relation(self) -> str:
        return self.triplet.relation.value


class TripletExtractor:
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 2,
    ) -> None:
        self.llm = llm.with_structured_output(ExtractionResult, method="json_mode")
        self.max_retries = max_retries

    def extract_from_chunk(
        self,
        chunk_text: str,
        ticker: str,
        year: int,
        section_id: str,
        chunk_id: str,
    ) -> List[FinancialTriplet]:
        payload = {
            "chunk_text": chunk_text[:4000],
            "ticker": ticker,
            "year": str(year),
            "section_id": section_id,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                response: ExtractionResult = self.llm.invoke(
                    EXTRACTION_PROMPT.format_messages(**payload)
                )
                return [
                    FinancialTriplet(
                        triplet=t,
                        chunk_id=chunk_id,
                        company_ticker=ticker,
                        fiscal_year=year,
                        section_id=section_id,
                    )
                    for t in response.triplets
                ]
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Extraction attempt %d failed for chunk %s: %s",
                    attempt + 1,
                    chunk_id,
                    exc,
                )

        logger.error(
            "All extraction attempts failed for chunk %s: %s",
            chunk_id,
            last_error,
        )
        return []
