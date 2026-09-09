from __future__ import annotations

import json
import logging
import random
import re
import time
from enum import Enum
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "/no_think You are an expert financial analyst extracting structured knowledge "
            "from SEC 10-K reports. Extract all financial entities and their "
            "relationships from the text below. Use ONLY the ontology provided.\n\n"
            "ENTITY TYPES:\n"
            "- Company: a public corporation identified by ticker\n"
            "- FinancialMetric: a numeric metric (revenue, net income, EPS, etc.) — "
            "ALWAYS include value/unit when a number is present in the text\n"
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
            '"target": {{"name": "...", "value": "...", "unit": "...", "year": "2024"}}}}]}}\n\n'
            "Examples:\n"
            "- Text: \"Total net sales | $ 391,035 | $ 383,285\" with header \"2024 | 2023\" → "
            "produce TWO triplets: {{\"name\":\"total net sales\",\"value\":\"391035\",\"unit\":\"USD millions\",\"year\":\"2024\"}} and "
            "{{\"name\":\"total net sales\",\"value\":\"383285\",\"unit\":\"USD millions\",\"year\":\"2023\"}}\n"
            "- Text: \"Net income was $99,800 for fiscal year 2024\" → {{\"name\":\"net income\",\"value\":\"99800\",\"unit\":\"USD millions\",\"year\":\"2024\"}}\n\n"
            "Rules:\n"
            "- Do NOT use <think> tags. Do NOT explain.\n"
            "- Use the exact relation names above.\n"
            "- The source must be the company ticker where the relation is "
            "Company -> X (e.g. operates_in, reported_metric).\n"
            "- For FinancialMetric targets, add \"value\" and \"unit\" when the text gives a number "
            "(e.g. {{\"name\":\"total net sales\",\"value\":\"383285\",\"unit\":\"USD millions\",\"year\":\"2023\"}}). "
            "CRITICAL for tables: if the chunk contains a table with multiple year columns (e.g., \"2024 | 2023\"), "
            "create ONE triplet per year column, mapping each value to its column year. Do NOT assign all values to filing year.\n"
            "- Only include entities explicitly mentioned in the text.\n"
            "- Omit relations when the target is unknown.",
        ),
        (
            "human",
            "/no_think TEXT: {chunk_text}\n\n"
            "COMPANY TICKER: {ticker}\n"
            "FISCAL YEAR: {year}\n"
            "SECTION: {section_id}",
        ),
    ]
)

# Batch prompt: N chunks in one LLM call to amortize latency (3-8x speedup)
BATCH_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "/no_think You are an expert financial analyst extracting structured knowledge "
            "from SEC 10-K reports. Use ONLY the ontology below.\n\n"
            "ENTITY TYPES: Company (ticker), FinancialMetric (with value/unit/year when numeric), RiskFactor, "
            "BusinessSegment, MacroEvent\n"
            "RELATIONSHIP TYPES: operates_in (Company->BusinessSegment), "
            "reported_metric (Company->FinancialMetric), "
            "impacts_revenue (RiskFactor/MacroEvent->FinancialMetric), "
            "mitigates_risk (BusinessSegment->RiskFactor), "
            "competes_with (Company->Company)\n\n"
            "You will receive N chunks, each prefixed with CHUNK_ID and SECTION.\n"
            "Do NOT use <think> tags. Do NOT add explanations. Return ONLY a JSON object with this exact shape, no extra text:\n"
            '{{\"results\": [{{\"chunk_id\": \"...\", \"triplets\": [{{\"source\": {{\"name\": \"...\"}}, \"relation\": \"...\", \"target\": {{\"name\": \"...\", \"value\": \"...\", \"unit\": \"...\", \"year\": \"2024\"}}}}]}}]}}\n\n'
            "Rules:\n"
            "- One entry per input chunk_id, even if no triplets (use empty list).\n"
            "- For Company->X relations, source must be the company ticker.\n"
            "- For FinancialMetric targets, add value/unit/year when the text gives a number. For tables with 2024|2023 columns, emit one triplet per year.\n"
            "- Only include entities explicitly mentioned in that chunk.\n"
            "- Use exact relation names listed above.",
        ),
        (
            "human",
            "/no_think COMPANY TICKER: {ticker}\n"
            "FISCAL YEAR: {year}\n\n"
            "CHUNKS:\n{chunks_block}",
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


# Heurística ligera para evitar llamar al LLM en chunks vacíos/boilerplate
_SKIPPABLE_SECTION_PATTERNS = re.compile(
    r"(table of contents|exhibit\s+\d+|power of attorney|signatures?)", re.IGNORECASE
)


def _is_skippable_chunk(text: str, section_id: str) -> bool:
    t = (text or "").strip()
    if not t or len(t.split()) < 20:
        return True
    # Secciones que casi nunca aportan tripletas con la ontología actual
    if _SKIPPABLE_SECTION_PATTERNS.search(section_id or ""):
        # pero no descartar si el texto menciona métricas/segmentos explícitos
        lower = t.lower()
        if not any(k in lower for k in ("revenue", "segment", "risk", "competitor", "subsidiary", "market")):
            return True
    return False


class TripletExtractor:
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 2,
        base_delay: float = 0.6,
        max_delay: float = 4.0,
    ) -> None:
        self.llm = llm
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    # ------------------------------------------------------------------ single
    def extract_from_chunk(
        self,
        chunk_text: str,
        ticker: str,
        year: int,
        section_id: str,
        chunk_id: str,
    ) -> List[FinancialTriplet]:
        if _is_skippable_chunk(chunk_text, section_id):
            logger.debug("Skipping boilerplate chunk %s (section=%s)", chunk_id, section_id)
            return []
        payload = {
            "chunk_text": chunk_text[:4000],
            "ticker": ticker,
            "year": str(year),
            "section_id": section_id,
        }
        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries + 1):
            try:
                raw = self._invoke_single_json(payload)
                raw = self._enrich_metric_year_fallback(raw, chunk_text, year)
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
                if attempt <= self.max_retries:
                    delay = min(self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5), self.max_delay)
                    logger.warning(
                        "Extraction attempt %d/%d failed for chunk %s: %s — retry in %.1fs",
                        attempt,
                        self.max_retries + 1,
                        chunk_id,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "Extraction attempt %d/%d failed for chunk %s: %s",
                        attempt,
                        self.max_retries + 1,
                        chunk_id,
                        exc,
                    )

        logger.error(
            "All extraction attempts failed for chunk %s: %s",
            chunk_id,
            last_error,
        )
        return []

    # ------------------------------------------------------------------ batch
    def extract_from_batch(
        self,
        batch: List[dict],
        ticker: str,
        year: int,
    ) -> Dict[str, List[FinancialTriplet]]:
        """
        Extrae tripletas para un lote de chunks en una sola llamada LLM.
        batch: list of {chunk_id, text, section_id}
        Retorna dict chunk_id -> List[FinancialTriplet]. En fallo total, cae
        a extracción single-chunk por elemento.
        """
        # Separa skippables sin LLM
        skippable_ids: set[str] = set()
        active: List[dict] = []
        for item in batch:
            if _is_skippable_chunk(item.get("text", ""), item.get("section_id", "")):
                skippable_ids.add(item["chunk_id"])
            else:
                active.append(item)

        result: Dict[str, List[FinancialTriplet]] = {cid: [] for cid in skippable_ids}

        if not active:
            return result

        # Fast-path: batch_size 1 → evita el prompt batch inestable de qwen3; usa single paralelo
        if len(active) == 1:
            item = active[0]
            cid = item["chunk_id"]
            triplets = self.extract_from_chunk(
                chunk_text=item.get("text", ""),
                ticker=ticker,
                year=year,
                section_id=item.get("section_id", ""),
                chunk_id=cid,
            )
            result[cid] = triplets
            return result

        # Formatea bloque para el prompt batch
        chunks_block_parts: List[str] = []
        id_to_meta: Dict[str, dict] = {}
        for item in active:
            cid = item["chunk_id"]
            sec = item.get("section_id", "")
            # 1800 chars ≈ 450 tokens; 4 chunks → ~1800 tokens + prompt < 8192 ctx
            txt = (item.get("text", "") or "")[:1900]
            chunks_block_parts.append(f"--- CHUNK_ID: {cid} | SECTION: {sec} ---\n{txt}")
            id_to_meta[cid] = item

        chunks_block = "\n\n".join(chunks_block_parts)
        payload = {"ticker": ticker, "year": str(year), "chunks_block": chunks_block}

        last_error: Optional[Exception] = None
        # Batch es inestable en qwen3; solo 1 reintento, luego fallback inmediato a per-chunk
        batch_retries = min(self.max_retries, 1)
        for attempt in range(1 + batch_retries + 1):
            try:
                raw_by_id = self._invoke_batch_json(payload)
                # Enrich year fallback por chunk antes de parsear
                for cid, lst in list(raw_by_id.items()):
                    meta = id_to_meta.get(cid)
                    if meta:
                        raw_by_id[cid] = self._enrich_metric_year_fallback(lst, meta.get("text",""), year)
                # raw_by_id: dict chunk_id -> List[dict triplets raw]
                for cid, raw_triplets in raw_by_id.items():
                    meta = id_to_meta.get(cid)
                    if meta is None:
                        logger.debug("Batch returned unknown chunk_id %s — ignoring", cid)
                        continue
                    parsed = self._parse_triplets(raw_triplets)
                    result[cid] = [
                        FinancialTriplet(
                            triplet=t,
                            chunk_id=cid,
                            company_ticker=ticker,
                            fiscal_year=year,
                            section_id=meta.get("section_id", ""),
                        )
                        for t in parsed
                    ]
                # Asegura que todo chunk activo tenga entrada
                for cid in id_to_meta:
                    result.setdefault(cid, [])
                return result
            except Exception as exc:
                last_error = exc
                if attempt <= batch_retries:
                    delay = min(self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3), self.max_delay)
                    logger.debug(
                        "Batch extraction attempt %d/%d failed (%d chunks): %s — retry in %.1fs",
                        attempt,
                        batch_retries + 1,
                        len(active),
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.debug(
                        "Batch extraction attempt %d/%d failed (%d chunks): %s",
                        attempt,
                        batch_retries + 1,
                        len(active),
                        exc,
                    )

        # Fallback: intenta uno a uno para no perder todo el lote
        logger.info("Batch fallback to per-chunk for %d chunks (%s)", len(active), last_error)
        for item in active:
            cid = item["chunk_id"]
            # evita sobrescribir si ya se obtuvo algo en intentos parciales
            if cid in result and result[cid]:
                continue
            triplets = self.extract_from_chunk(
                chunk_text=item.get("text", ""),
                ticker=ticker,
                year=year,
                section_id=item.get("section_id", ""),
                chunk_id=cid,
            )
            result[cid] = triplets
        return result

    # ---------------------------------------------------------------- internal
    def _invoke_single_json(self, payload: dict) -> List[dict]:
        messages = EXTRACTION_PROMPT.format_messages(**payload)
        # Solo invoke directo + parsing manual (structured falla con qwen3: espera source_entity vs source)
        response = self.llm.invoke(messages)
        content = getattr(response, "content", None)
        if content is None:
            if isinstance(response, dict):
                content = json.dumps(response)
            else:
                raise ValueError("LLM returned no content")
        raw = str(content)
        json_text = self._strip_code_fences(raw)
        if not json_text or json_text.strip() in ("", "null"):
            raise ValueError(f"LLM empty (single). Raw preview: {raw[:300]!r}")
        data = self._extract_json_object(json_text)
        if isinstance(data, dict):
            triplets = data.get("triplets", [])
        elif isinstance(data, list):
            triplets = data
        else:
            raise ValueError(f"Unexpected JSON shape: {type(data)}")
        return [self._normalize_triplet(item) for item in triplets if isinstance(item, (dict, list))]

    def _invoke_batch_json(self, payload: dict) -> Dict[str, List[dict]]:
        messages = BATCH_EXTRACTION_PROMPT.format_messages(**payload)
        response = self.llm.invoke(messages)
        content = getattr(response, "content", None)
        if content is None:
            if isinstance(response, dict):
                content = json.dumps(response)
            else:
                raise ValueError("LLM returned no content (batch)")
        raw_content = str(content)
        json_text = self._strip_code_fences(raw_content)
        if not json_text or json_text.strip() in ("", "null", "{}"):
            raise ValueError(f"LLM returned empty JSON (batch). Raw preview: {raw_content[:400]!r}")
        data = self._extract_json_object(json_text)
        return self._parse_batch_data(data)

    def _parse_batch_data(self, data) -> Dict[str, List[dict]]:
        """Normaliza respuesta batch a dict chunk_id -> raw triplets."""
        if isinstance(data, dict):
            # Formato esperado: {"results": [{"chunk_id": "...", "triplets": [...]}, ...]}
            if "results" in data and isinstance(data["results"], list):
                out: Dict[str, List[dict]] = {}
                for entry in data["results"]:
                    if not isinstance(entry, dict):
                        continue
                    cid = entry.get("chunk_id") or entry.get("id") or entry.get("chunkId")
                    if not cid:
                        continue
                    triplets = entry.get("triplets", entry.get("triples", []))
                    if not isinstance(triplets, list):
                        triplets = []
                    out[str(cid)] = [self._normalize_triplet(t) for t in triplets if isinstance(t, (dict, list))]
                return out
            # Fallback: diccionario chunk_id -> triplets
            # o {"triplets": [...]} sin chunk_id (caso degenerado)
            if "triplets" in data and isinstance(data["triplets"], list) and "results" not in data:
                # No hay forma de asignar sin chunk_id: retorna vacío y el caller hará fallback
                raise ValueError("Batch response missing chunk_id mapping (got single triplets list)")
            # Intenta tratar cada clave como chunk_id
            out2: Dict[str, List[dict]] = {}
            for k, v in data.items():
                if isinstance(v, list):
                    out2[str(k)] = [self._normalize_triplet(t) for t in v if isinstance(t, (dict, list))]
                elif isinstance(v, dict) and "triplets" in v:
                    out2[str(k)] = [self._normalize_triplet(t) for t in v["triplets"] if isinstance(t, (dict, list))]
            if out2:
                return out2
        elif isinstance(data, list):
            # Lista de {"chunk_id":..., "triplets":...}
            out3: Dict[str, List[dict]] = {}
            for entry in data:
                if isinstance(entry, dict) and "chunk_id" in entry:
                    cid = entry["chunk_id"]
                    triplets = entry.get("triplets", [])
                    out3[str(cid)] = [self._normalize_triplet(t) for t in triplets if isinstance(t, (dict, list))]
            if out3:
                return out3
        raise ValueError(f"Unrecognized batch JSON shape: {str(data)[:400]}")

    @staticmethod
    def _triplet_to_raw_dict(triplet: Triplet) -> dict:
        return {
            "source": {"name": triplet.source_entity.name, **triplet.source_entity.properties},
            "relation": triplet.relation.value,
            "target": {"name": triplet.target_entity.name, **triplet.target_entity.properties},
        }

    def _extract_json_object(self, text: str):
        """Extrae el primer objeto JSON válido del texto. Tolera prefijos/sufijos."""
        text = text.strip()
        # Intento directo
        try:
            return self._loads_json(text)
        except Exception:
            pass
        # Busca el JSON más externo entre primera { y última }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return self._loads_json(candidate)
            except Exception:
                pass
            # Intenta extraer array si el modelo devolvió lista
            start_a = text.find("[")
            end_a = text.rfind("]")
            if start_a != -1 and end_a != -1:
                candidate_a = text[start_a : end_a + 1]
                try:
                    return self._loads_json(candidate_a)
                except Exception:
                    pass
        # último intento: loads con reparaciones
        return self._loads_json(text)

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
        return cls._BARE_VALUE_OBJECT.sub(r'{"name": \1}', text)

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
    def _strip_thinking(text: str) -> str:
        # qwen3:8b en modo thinking emite <think>...</think> antes del JSON
        # format=json de Ollama no siempre lo elimina
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # por si queda etiqueta sin cerrar
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        # primero elimina thinking, luego code fences
        text = TripletExtractor._strip_thinking(text)
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
                "source": {"name": src} if not isinstance(src, dict) else src,
                "relation": rel,
                "target": {"name": tgt} if not isinstance(tgt, dict) else tgt,
            }
        src = item.get("source_entity") or item.get("source") or {}
        tgt = item.get("target_entity") or item.get("target") or {}
        if isinstance(src, str):
            src = {"name": src}
        if isinstance(tgt, str):
            tgt = {"name": tgt}
        # preserva value/unit/year si el LLM los devolvió para FinancialMetric
        for ent in (src, tgt):
            if isinstance(ent, dict):
                extra = {k: str(v).strip() for k, v in ent.items() if k in ("value", "unit", "fiscal_year", "year") and v not in (None, "")}
                ent.update(extra)
        return {
            "source": src,
            "relation": item.get("relation"),
            "target": tgt,
        }

    def _enrich_metric_year_fallback(self, raw: List[dict], chunk_text: str, filing_year: int) -> List[dict]:
        """Si el LLM no dio year para FinancialMetric, infiérelo por proximidad a año en el texto."""
        # Extrae años en el chunk (2022-2026 típico)
        year_positions = [(m.start(), int(m.group(1))) for m in re.finditer(r"\b(20(?:2[0-9]|19))\b", chunk_text)]
        if not year_positions:
            return raw
        # Para cada triplet métrica sin year, busca el año más cercano antes del valor
        for item in raw:
            try:
                rel = item.get("relation")
                tgt = item.get("target") or {}
                if rel not in ("reported_metric", "impacts_revenue"):
                    continue
                if tgt.get("year") or tgt.get("fiscal_year"):
                    continue
                val = str(tgt.get("value") or "")
                if not val:
                    continue
                # posición del valor en el texto (primer ocurrencia del número sin comas)
                val_digits = re.sub(r"[^\d]", "", val)[:6]
                pos = chunk_text.find(val_digits) if val_digits else -1
                if pos == -1:
                    # busca por nombre de métrica
                    pos = chunk_text.lower().find(str(tgt.get("name","")).lower())
                if pos == -1:
                    continue
                # año más cercano antes de pos (dentro de 400 chars)
                best_year = None
                best_dist = 10**9
                for y_pos, y_val in year_positions:
                    if y_pos < pos:
                        dist = pos - y_pos
                        if dist < 400 and dist < best_dist:
                            best_dist = dist
                            best_year = y_val
                if best_year:
                    tgt["year"] = str(best_year)
            except Exception:
                continue
        return raw

    def _parse_triplets(self, raw: List[dict]) -> List[Triplet]:
        triplets: List[Triplet] = []
        for item in raw:
            try:
                src_raw = item["source"] or {}
                tgt_raw = item["target"] or {}
                src_props = {k: v for k, v in src_raw.items() if k not in ("name",) and v not in (None, "")}
                tgt_props = {k: v for k, v in tgt_raw.items() if k not in ("name",) and v not in (None, "")}
                source = Entity(
                    name=src_raw["name"],
                    entity_type=self._infer_entity_type(item["relation"], side="source"),
                    properties=src_props,
                )
                target = Entity(
                    name=tgt_raw["name"],
                    entity_type=self._infer_entity_type(item["relation"], side="target"),
                    properties=tgt_props,
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
