from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import kuzu

from src.graph.schema import GraphSchema
from src.retrieval.graph_traversal import GraphTraversalRetriever

logger = logging.getLogger(__name__)

_NOISE_NAMES: frozenset = frozenset(
    {
        "other companies",
        "others",
        "unspecified competitors",
        "unspecified companies",
        "unspecified",
        "our competitors",
        "our peers",
        "competitors",
        "not specified in the text",
        "none",
        "n/a",
        "1)",
        # reserved node-table labels leaked by extraction
        "businesssegment",
        "riskfactor",
        "financialmetric",
        "company",
        "macroevent",
    }
)


@dataclass
class GraphFact:
    subject: str
    relation: str
    object: str

    def to_text(self) -> str:
        return f"{self.subject} --{self.relation}--> {self.object}"


def _is_noise(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in _NOISE_NAMES:
        return True
    if lowered.startswith(
        (
            "companies that",
            "companies with",
            "other companies",
            "providers of",
            "including ",
            "such as ",
            "our other ",
            "other participants",
            "third-party ",
            "unspecified ",
            "unknown ",
            "across ",
        )
    ):
        return True
    if "various industries" in lowered:
        return True
    if len(name) > 60:
        return True
    return False


class GraphFactRetriever:
    """Surfaces structured *structural* knowledge from the knowledge graph.

    Only trustworthy relationship facts are emitted as extra context for the
    generator: the business segments a company operates in, and the companies
    it competes with. Numeric metric *values* are deliberately excluded because
    ``FinancialMetric`` nodes are keyed only by name and shared across
    companies, so a single value (e.g. "total assets = 352583") gets attributed
    to every company that reports that metric and is therefore unreliable.
    """

    def __init__(
        self,
        schema: GraphSchema,
        traversal: GraphTraversalRetriever,
        max_facts_per_entity: int = 10,
        max_total: int = 30,
    ) -> None:
        self.schema = schema
        self.traversal = traversal
        self.max_facts_per_entity = max_facts_per_entity
        self.max_total = max_total

    def search(self, query: str, top_k: int = 30) -> List[str]:
        matched = self.traversal.match_query_entities(query)
        if not matched:
            return []

        conn = self.schema.connection
        seen: set[tuple[str, str, str]] = set()
        facts: List[str] = []

        for name, table in matched:
            if table != "Company":
                continue
            entity_facts = (
                self._company_segment_facts(conn, name)
                + self._company_competitor_facts(conn, name)
            )
            for fact in entity_facts:
                key = (fact.subject, fact.relation, fact.object)
                if key in seen:
                    continue
                seen.add(key)
                object_normalized = fact.object.lower()
                if (
                    not fact.object
                    or _is_noise(fact.object)
                    or _is_noise(fact.subject)
                    or fact.object.lower() == fact.subject.lower()
                ):
                    continue
                facts.append(fact.to_text())
            if len(facts) >= self.max_total:
                break

        logger.debug(
            "Graph facts: %d facts for query %r (matched %d entities)",
            len(facts),
            query,
            len(matched),
        )
        return sorted(set(facts))[:top_k]

    def _company_segment_facts(
        self,
        conn: kuzu.Connection,
        ticker: str,
    ) -> List[GraphFact]:
        facts: List[GraphFact] = []
        result = conn.execute(
            "MATCH (a:Company)-[r:OPERATES_IN]->(b:BusinessSegment) "
            "WHERE a.ticker = $name "
            "RETURN DISTINCT b.id LIMIT $limit",
            {"name": ticker, "limit": self.max_facts_per_entity},
        )
        while result.has_next():
            row = result.get_next()
            facts.append(GraphFact(subject=ticker, relation="OPERATES_IN", object=str(row[0])))
        return facts

    def _company_competitor_facts(
        self,
        conn: kuzu.Connection,
        ticker: str,
    ) -> List[GraphFact]:
        facts: List[GraphFact] = []
        result = conn.execute(
            "MATCH (a:Company)-[r:COMPETES_WITH]->(b:Company) "
            "WHERE a.ticker = $name "
            "RETURN DISTINCT b.ticker LIMIT $limit",
            {"name": ticker, "limit": self.max_facts_per_entity},
        )
        while result.has_next():
            row = result.get_next()
            competitor = str(row[0])
            if _is_noise(competitor):
                continue
            facts.append(GraphFact(subject=ticker, relation="COMPETES_WITH", object=competitor))
        return facts