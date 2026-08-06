from __future__ import annotations

import json
import logging
import re
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
            "Respond ONLY with a JSON object in the following shape, without extra text:\n"
            '{{"triplets": [{{"source": {{"name": "..."}}, "relation": "...", '
            '"target": {{"name": "..."}}}}]}}\n\n'
            "Rules:\n"
            "- Use the exact relation names above.\n"
            "- The source must be the company ticker where the relation is "
            "Company -> X (e.g. operates_in, reported_metric).\n"
            "- Only include entities explicitly mentioned in the text.\n"
            "- Omit relations when the target is unknown.",
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
        self.llm = llm
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
                raw = self._invoke_json(payload)
                parsed = self._parse_triplets(raw)
                return [
                    FinancialTriplet(
                        triplet=t,
                        chunk_id=chunk_id,
                        company_ticker=ticker,
                        fiscal_year=year,
                        section_id=section_id,
                    )
                    for t in parsed
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

    def _invoke_json(self, payload: dict) -> List[dict]:
        messages = EXTRACTION_PROMPT.format_messages(**payload)
        try:
            response = self.llm.invoke(messages)
        except Exception:
            structured = self.llm.with_structured_output(
                ExtractionResult, method="json_mode"
            )
            response = structured.invoke(messages)

        content = getattr(response, "content", None)
        if content is None:
            raise ValueError("LLM returned no content")

        json_text = self._strip_code_fences(str(content))
        data = self._loads_json(json_text)
        if isinstance(data, dict):
            triplets = data.get("triplets", [])
        elif isinstance(data, list):
            triplets = data
        else:
            raise ValueError(f"Unexpected JSON shape: {type(data)}")

        return [
            self._normalize_triplet(item)
            for item in triplets
            if isinstance(item, (dict, list))
        ]

    @staticmethod
    def _loads_json(json_text: str):
        try:
            return json.loads(json_text)
        except Exception as exc:
            repaired = TripletExtractor._repair_bare_value_objects(json_text)
            try:
                return json.loads(repaired)
            except Exception:
                repaired = TripletExtractor._repair_json(repaired)
                try:
                    return json.loads(repaired)
                except Exception:
                    raise exc

    _BARE_VALUE_OBJECT = re.compile(r'\{\s*("[^"]*")\s*\}')

    @classmethod
    def _repair_bare_value_objects(cls, text: str) -> str:
        return cls._BARE_VALUE_OBJECT.sub(
            r'{"name": \1}', text
        )

    @staticmethod
    def _repair_json(text: str) -> str:
        stack: List[str] = []
        in_string = False
        escaped = False
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        expected = "}" if stack[-1] == "{" else "]"
                        if ch == expected:
                            stack.pop()
                        else:
                            stack.pop()
            i += 1

        if in_string:
            text += '"'
        for opening in reversed(stack):
            text += "}" if opening == "{" else "]"
        return text

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _normalize_triplet(item) -> dict:
        if isinstance(item, list):
            if len(item) < 3:
                raise ValueError(f"Array triplet too short: {item}")
            src, rel, tgt = item[0], item[1], item[2]
            return {
                "source": {"name": src},
                "relation": rel,
                "target": {"name": tgt},
            }
        src = item.get("source_entity") or item.get("source") or {}
        tgt = item.get("target_entity") or item.get("target") or {}
        if isinstance(src, str):
            src = {"name": src}
        if isinstance(tgt, str):
            tgt = {"name": tgt}
        return {
            "source": src,
            "relation": item.get("relation"),
            "target": tgt,
        }

    def _parse_triplets(self, raw: List[dict]) -> List[Triplet]:
        triplets: List[Triplet] = []
        for item in raw:
            try:
                source = Entity(
                    name=item["source"]["name"],
                    entity_type=self._infer_entity_type(
                        item["relation"], side="source"
                    ),
                )
                target = Entity(
                    name=item["target"]["name"],
                    entity_type=self._infer_entity_type(
                        item["relation"], side="target"
                    ),
                )
                triplets.append(
                    Triplet(
                        source_entity=source,
                        relation=RelationType(item["relation"]),
                        target_entity=target,
                    )
                )
            except Exception as exc:
                logger.debug("Skipping invalid triplet %s: %s", item, exc)
        return triplets

    @staticmethod
    def _infer_entity_type(relation: str, side: str) -> EntityType:
        mapping = {
            ("operates_in", "source"): EntityType.company,
            ("operates_in", "target"): EntityType.business_segment,
            ("reported_metric", "source"): EntityType.company,
            ("reported_metric", "target"): EntityType.financial_metric,
            ("impacts_revenue", "source"): EntityType.macro_event,
            ("impacts_revenue", "target"): EntityType.financial_metric,
            ("mitigates_risk", "source"): EntityType.business_segment,
            ("mitigates_risk", "target"): EntityType.risk_factor,
            ("competes_with", "source"): EntityType.company,
            ("competes_with", "target"): EntityType.company,
        }
        try:
            return mapping[(relation, side)]
        except KeyError:
            return EntityType.company
