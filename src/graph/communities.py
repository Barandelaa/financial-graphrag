from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.graph.schema import GraphSchema

logger = logging.getLogger(__name__)

COMMUNITY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a financial analyst generating executive summaries of "
            "business communities extracted from SEC 10-K reports. Given a "
            "list of entities and their relationships within a community, "
            "provide a concise strategic summary covering: key entities, "
            "material risks, financial trends, and competitive dynamics.",
        ),
        (
            "human",
            "Community ID: {community_id}\n\n"
            "Entities:\n{entity_list}\n\n"
            "Relationships:\n{relationship_list}",
        ),
    ]
)


@dataclass
class CommunitySummary:
    community_id: int
    summary: str
    entity_count: int
    top_entities: List[str] = field(default_factory=list)


class CommunityDetector:
    def __init__(
        self,
        schema: GraphSchema,
        llm: Optional[BaseChatModel] = None,
        resolution: float = 1.0,
    ) -> None:
        self.schema = schema
        self.llm = llm
        self.resolution = resolution

    def detect_communities(self) -> Dict[int, List[str]]:
        logger.info("Running Leiden community detection via Kùzu export")

        node_labels = self._fetch_all_nodes()
        adjacency = self._fetch_adjacency()
        G = self._build_networkx(node_labels, adjacency)
        communities = self._run_leiden(G)

        logger.info(
            "Detected %d communities covering %d nodes",
            len(communities),
            sum(len(m) for m in communities.values()),
        )
        return communities

    def generate_summaries(
        self,
        communities: Dict[int, List[str]],
        max_communities: int = 20,
    ) -> List[CommunitySummary]:
        if self.llm is None:
            logger.warning("No LLM provided; skipping community summaries")
            return []

        all_summaries: List[CommunitySummary] = []
        sorted_cids = sorted(communities.keys(), key=lambda c: -len(communities[c]))

        for cid in sorted_cids[:max_communities]:
            members = communities[cid]
            entity_details = self._fetch_entity_details(members)
            relationships = self._fetch_relationships_for_entities(members)

            entity_lines = "\n".join(
                f"- {e['name']} ({e['type']})" for e in entity_details
            )
            rel_lines = "\n".join(
                f"- ({r['src']}) -[:{r['rel']}]-> ({r['dst']})"
                for r in relationships
            )

            try:
                response = self.llm.invoke(
                    COMMUNITY_SUMMARY_PROMPT.format_messages(
                        community_id=str(cid),
                        entity_list=entity_lines or "(none)",
                        relationship_list=rel_lines or "(none)",
                    )
                )
                summary_text = response.content if hasattr(response, "content") else str(response)
            except Exception as exc:
                logger.warning("Summary generation failed for community %d: %s", cid, exc)
                summary_text = f"Community {cid} — {len(members)} entities (summary unavailable)"

            all_summaries.append(
                CommunitySummary(
                    community_id=cid,
                    summary=summary_text,
                    entity_count=len(members),
                    top_entities=[e["name"] for e in entity_details[:5]],
                )
            )

        logger.info("Generated %d community summaries", len(all_summaries))
        return all_summaries

    def _fetch_all_nodes(self) -> List[dict]:
        conn = self.schema.connection
        nodes: List[dict] = []
        for table in ["Company", "FinancialMetric", "RiskFactor", "BusinessSegment", "MacroEvent"]:
            try:
                result = conn.execute(f"MATCH (n:{table}) RETURN n.*")
                while result.has_next():
                    row = result.get_next()
                    nodes.append({"name": str(row[0]), "type": table})
            except RuntimeError:
                continue
        return nodes

    def _fetch_adjacency(self) -> List[tuple[str, str]]:
        conn = self.schema.connection
        edges: List[tuple[str, str]] = []
        queries = {
            "OPERATES_IN": ("Company", "ticker", "BusinessSegment", "id"),
            "REPORTED_METRIC": ("Company", "ticker", "FinancialMetric", "id"),
            "IMPACTS_REVENUE": ("RiskFactor", "id", "FinancialMetric", "id"),
            "MITIGATES_RISK": ("BusinessSegment", "id", "RiskFactor", "id"),
            "COMPETES_WITH": ("Company", "ticker", "Company", "ticker"),
        }
        for rel, (src_t, src_pk, dst_t, dst_pk) in queries.items():
            try:
                q = (
                    f"MATCH (a:{src_t})-[r:{rel}]->(b:{dst_t}) "
                    f"RETURN a.{src_pk}, b.{dst_pk}"
                )
                result = conn.execute(q)
                while result.has_next():
                    row = result.get_next()
                    edges.append((str(row[0]), str(row[1])))
            except RuntimeError:
                continue
        return edges

    @staticmethod
    def _build_networkx(
        nodes: List[dict],
        edges: List[tuple[str, str]],
    ) -> "nx.Graph":
        import networkx as nx

        G = nx.Graph()
        for n in nodes:
            G.add_node(n["name"], node_type=n["type"])
        for src, dst in edges:
            G.add_edge(src, dst)
        return G

    @staticmethod
    def _run_leiden(G: "nx.Graph") -> Dict[int, List[str]]:
        try:
            import leidenalg as la
            import igraph as ig

            ig_graph = ig.Graph.from_networkx(G)
            partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
            communities: Dict[int, List[str]] = {}
            for idx, community in enumerate(partition):
                communities[idx] = [ig_graph.vs[v]["_nx_name"] for v in community]
            return communities
        except ImportError:
            logger.warning(
                "leidenalg or igraph not installed; falling back to connected components"
            )
            return _fallback_connected_components(G)

    def _fetch_entity_details(self, entity_names: List[str]) -> List[dict]:
        conn = self.schema.connection
        details: List[dict] = []
        for table in ["Company", "FinancialMetric", "RiskFactor", "BusinessSegment", "MacroEvent"]:
            try:
                pk = "ticker" if table == "Company" else "id"
                placeholders = ", ".join(f"'{n}'" for n in entity_names)
                result = conn.execute(
                    f"MATCH (n:{table}) WHERE n.{pk} IN [{placeholders}] RETURN n.{pk}, '{table}'"
                )
                while result.has_next():
                    row = result.get_next()
                    details.append({"name": str(row[0]), "type": table})
            except RuntimeError:
                continue
        return details

    def _fetch_relationships_for_entities(
        self,
        entity_names: List[str],
    ) -> List[dict]:
        conn = self.schema.connection
        rows: List[dict] = []
        rels = ["OPERATES_IN", "REPORTED_METRIC", "IMPACTS_REVENUE", "MITIGATES_RISK", "COMPETES_WITH"]
        for rel in rels:
            try:
                result = conn.execute(
                    f"""
                    MATCH (a)-[r:{rel}]->(b)
                    RETURN a, '{rel}', b
                    """
                )
                while result.has_next():
                    row = result.get_next()
                    src = str(row[0])
                    dst = str(row[2])
                    if src in entity_names or dst in entity_names:
                        rows.append({"src": src, "rel": rel, "dst": dst})
            except RuntimeError:
                continue
        return rows


def _fallback_connected_components(G: "nx.Graph") -> Dict[int, List[str]]:
    import networkx as nx

    components: Dict[int, List[str]] = {}
    for idx, comp in enumerate(nx.connected_components(G)):
        components[idx] = list(comp)
    return components
