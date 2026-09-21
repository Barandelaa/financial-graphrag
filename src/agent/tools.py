from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.tools import tool

from src.pipeline import FinancialGraphRAGPipeline

logger = logging.getLogger(__name__)

_VALID_TICKERS = frozenset({"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "BRK.B"})
_TICKER_RE = re.compile(r"\b(AAPL|MSFT|AMZN|GOOGL|NVDA|META|TSLA|BRK\.B)\b", re.IGNORECASE)


def _allowed_tickers() -> set[str]:
    """Tickers base + los dados de alta en companies.json (sin listas curadas nuevas)."""
    from src.agent.company_registry import get_config_tickers

    return set(_VALID_TICKERS) | {t.upper() for t in get_config_tickers()}


def _confirmed(value: object) -> bool:
    """El resume del interrupt() trae el 'y/n' del CLI (o True desde API)."""
    if value is True:
        return True
    return str(value or "").strip().lower() in ("y", "yes", "s", "si", "true", "ok")


def _make_ingest_tool(pipeline: FinancialGraphRAGPipeline, confirm_inside: bool = False):
    """Si confirm_inside=True, la confirmación HITL se pide dentro de la tool
    con interrupt() (patrón ReAct). Si False, la pausa la gestiona el llamador
    a nivel de nodo (patrón agente determinista)."""

    @tool
    def ingest_10k(ticker: str, year: int, force_clean: bool = False) -> str:
        """Ingiere el 10-K de la SEC para un ticker y año y lo indexa en grafo+vector. Usar para 'añade X' y también para 'continúa la ingesta de X': por defecto RETOMA donde se quedó (reutiliza chunks.json y aprovecha triplets.json). Solo re-parsea desde cero (borrando caché y stores del par) si force_clean=True explícito. La confirmación se pide automáticamente antes de ejecutar. ticker: cualquiera dado de alta en companies.json, year: 2020-2026."""
        from src.agent.company_registry import _ticker_key

        allowed = _allowed_tickers()
        by_key = {_ticker_key(t): t for t in allowed}
        key = _ticker_key(ticker)
        if key not in by_key:
            return f"Error: ticker '{ticker}' no está dado de alta. Usa propose_new_company primero."
        t = by_key[key]
        try:
            y = int(year)
        except Exception:
            return f"Error: year '{year}' invalido"
        if not (2020 <= y <= 2026):
            return f"Error: year {y} fuera de rango 2020-2026"
        if confirm_inside:
            from langgraph.types import interrupt

            ok = interrupt({"action": "confirm_ingest", "ticker": t, "year": y})
            if not _confirmed(ok):
                return f"Ingesta cancelada por el usuario. No se ha ingerido {t}/{y}."
        try:
            from reprocess_missing import reprocess_pair
            from src.agent.progress import IngestCancelled

            res = reprocess_pair(pipeline, t, y, force_clean=force_clean)
            mode = "retomada donde se quedó" if res["resumed"] else "limpia desde cero"
            return (
                f"OK: {t}/{y} ingesta {mode}: {res['chunks']} chunks, "
                f"triplets={res['triplets']}, persisted={res['persisted']} "
                f"-> data/processed_chunks/{t}_{y}/"
            )
        except IngestCancelled:
            logger.info("ingest_10k cancelada por el usuario para %s/%s", t, y)
            return f"Ingesta de {t}/{y} detenida por el usuario. Progreso guardado en disco: el usuario puede pedir 'continúa la ingesta' para reanudar donde se quedó."
        except Exception as exc:
            logger.exception("ingest_10k failed %s/%s: %s", t, y, exc)
            return f"Error ingesting {t}/{y}: {exc}"

    return ingest_10k


def make_react_tools(pipeline: FinancialGraphRAGPipeline):
    """Tools base para el agente ReAct. Lista extensible: añade aquí futuras tools."""
    from langchain_core.tools import tool as _tool

    ingest_tool = _make_ingest_tool(pipeline, confirm_inside=True)
    rp = pipeline.retrieval

    @_tool
    def query_financial_rag(question: str) -> str:
        """Busca en los 10-K indexados (dense BM25 grafo + rerank). Úsala para CUALQUIER pregunta sobre tickers/años: segmentos, métricas, riesgos Y COMPETIDORES (hechos COMPETES_WITH del grafo). Devuelve SOLO evidencia (hechos + tabla + citas): compón tu respuesta a partir de estos bloques."""
        try:
            result = pipeline.query(question)
            lines = ["EVIDENCE (compose your answer from these blocks; do not paste them verbatim):", ""]
            if result.graph_facts:
                lines.append("GRAPH FACTS (ideas de apoyo SIN chunk propio; citalas solo con un chunk_id real de CITATIONS):")
                lines.extend(f"- {f}" for f in result.graph_facts[:20])
                lines.append("")
            metrics = getattr(result, "metrics_rows", [])
            if metrics:
                lines.append("METRICS TABLE (| ticker | year | metric | value |):")
                for m in metrics[:20]:
                    lines.append(f"| {m.get('ticker')} | {m.get('year')} | {m.get('metric')} | {m.get('value')} {m.get('unit','') or ''} | {m.get('metric_id')} | {m.get('chunk_id','')} |")
                lines.append("")
            if result.citations:
                lines.append("CITATIONS (únicos chunk_id válidos para citar; NO inventes otros ni uses relaciones como chunk):")
                for c in result.citations[:5]:
                    lines.append(f"- {c.get('company_ticker')} | {c.get('fiscal_year')} | {c.get('section_id')} | chunk {c.get('chunk_id')}")
            # VRAM: cede a Ollama tras rerank/embeddings
            try:
                import gc as _gc
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                _gc.collect()
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("query_financial_rag failed: %s", exc)
            return f"Error en retrieval: {exc}"

    @_tool
    def lookup_metrics(ticker: str, year: int = 0, metric_hint: str = "") -> str:
        """Tabla de métricas scoping TICKER_YEAR (ej: AAPL_2024_total_net_sales). Úsala para cifras exactas revenue/net income/EPS. ticker requerido, year 0=todos, metric_hint filtra por nombre."""
        try:
            q = f"{ticker} {metric_hint or 'revenue net income EPS'} {year if year else ''}".strip()
            rows = rp.metrics_retriever.search(q)
            if year:
                rows = [r for r in rows if r.year == int(year)]
            if metric_hint:
                mh = metric_hint.lower()
                rows = [r for r in rows if mh in r.metric.lower() or any(w in r.metric.lower() for w in mh.split())]
            if not rows:
                return f"Sin métricas scoping para {ticker} {year or ''} '{metric_hint}'. Prueba query_financial_rag con la pregunta completa."
            out = [rp.metrics_retriever.format_as_block(rows[:20])]
            return "\n".join(out)
        except Exception as exc:
            return f"Error lookup_metrics: {exc}"

    @_tool
    def financial_calculator(operation: str, a: str, b: str = "") -> str:
        """Calculadora determinista para YoY y comparativas. FLUJO OBLIGATORIO: PRIMERO obtén 'a' y 'b' con query_financial_rag o lookup_metrics (misma unidad), LUEGO llama aquí. NUNCA calcules con cifras de memoria. operation: 'yoy_pct'|'pct_change' (a=current,b=previous), 'diff' (a-b), 'ratio' (a/b), 'sum', 'avg'. Acepta '$391,035', '383285', '12.5B'."""
        import re as _re

        def _parse(raw: str) -> tuple[float, str]:
            s = (raw or "").strip().replace(",", "").replace("$", "").replace("%", "").strip()
            m = _re.match(r"^([+-]?\d+(?:\.\d+)?)\s*([bmk]|billion|million|thousand)?s?$", s, _re.IGNORECASE)
            if not m:
                # último intento: primer número dentro del texto
                m2 = _re.search(r"[+-]?\d+(?:\.\d+)?", s)
                if not m2:
                    raise ValueError(f"no es un número: {raw!r}")
                return float(m2.group(0)), ""
            num = float(m.group(1))
            suf = (m.group(2) or "").lower()
            mult = {"b": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6, "k": 1e3, "thousand": 1e3}.get(suf, 1.0)
            return num * mult, suf

        try:
            op = (operation or "").strip().lower()
            if op not in ("yoy_pct", "pct_change", "diff", "ratio", "sum", "avg"):
                return f"Error: operation '{operation}' no válida. Usa: yoy_pct|pct_change|diff|ratio|sum|avg."
            av, _ = _parse(a)
            if op in ("yoy_pct", "pct_change", "diff", "ratio") and not (b or "").strip():
                return f"Error: operation '{op}' necesita 'b' (valor previo/base). Primero recupéralo con lookup_metrics."
            bv, _ = _parse(b) if (b or "").strip() else (0.0, "")
            if op in ("yoy_pct", "pct_change"):
                if bv == 0:
                    return "Error: división por cero (b=0)."
                pct = (av - bv) / abs(bv) * 100
                return f"FORMULA: ({av:g} - {bv:g}) / {bv:g} * 100 = {pct:.2f}%"
            if op == "diff":
                return f"FORMULA: {av:g} - {bv:g} = {av - bv:g}"
            if op == "ratio":
                if bv == 0:
                    return "Error: división por cero (b=0)."
                return f"FORMULA: {av:g} / {bv:g} = {av / bv:.4f}"
            if op == "sum":
                return f"FORMULA: {av:g} + {bv:g} = {av + bv:g}"
            # avg
            return f"FORMULA: ({av:g} + {bv:g}) / 2 = {(av + bv) / 2:g}"
        except Exception as exc:
            return f"Error financial_calculator: {exc}"

    @_tool
    def propose_new_company(user_text: str) -> str:
        """Propone el alta de una empresa nueva: resuelve texto libre contra el universo oficial SEC y verifica 10-K en EDGAR. SOLO LECTURA, no escribe nada. Si la coincidencia es exacta de ticker la propone directa; si no, devuelve candidatos para PREGUNTAR al usuario antes de buscar documentos."""
        try:
            from src.agent.company_registry import get_config_tickers, resolve_company, verify_10k

            res = resolve_company(user_text)
            if res.get("exact"):
                t = res["ticker"]
                if t in get_config_tickers():
                    return f"EXACTA: {t} ({res['name']}) ya está dada de alta en companies.json. Usa ingest_10k si falta algún año."
                v = verify_10k(res["cik"], t)
                if v.get("ok"):
                    return f"EXACTA: {t} ({res['name']}, CIK {res['cik']}) con 10-K recientes {v.get('recent_10k')}. Pide confirmación 1 para añadir a companies.json."
                return f"EXACTA pero SIN 10-K: {t} ({res['name']}): {v.get('reason')} No propongas alta."
            cands = res.get("candidates", [])
            if not cands:
                return f"Sin candidatos en el universo SEC para {user_text!r}. Pide al usuario el ticker exacto (ej: DOW) o aclara si es un índice (sin 10-K)."
            lines = [f"Candidatos SEC para {user_text!r} — PREGUNTA al usuario cuál es antes de buscar documentos:"]
            for c in cands:
                lines.append(f"- {c['ticker']} ({c['name']}) score={c['score']}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error propose_new_company: {exc}"

    @_tool
    def add_company_to_config(ticker: str) -> str:
        """Añade un ticker a data/companies.json (con backup .bak). La confirmación se pide automáticamente antes de escribir. No ingiere nada; la ingesta va después con ingest_10k."""
        from src.agent.company_registry import (
            _ticker_key,
            add_company,
            get_config_tickers,
            get_config_years,
            record_resolution,
        )
        from langgraph.types import interrupt

        t = (ticker or "").strip().upper()
        if _ticker_key(t) in {_ticker_key(c) for c in get_config_tickers()}:
            return f"{t} ya está dado de alta en companies.json. Pide confirmación 2 para ejecutar ingest_10k."
        # FUERA del try: interrupt() lanza GraphInterrupt y no debe ser tragado por el except.
        ok = interrupt({"action": "confirm_add_company", "ticker": t, "years": get_config_years()})
        if not _confirmed(ok):
            return "Alta cancelada por el usuario. No se ha modificado companies.json."
        try:
            out = add_company(ticker)
            record_resolution(ticker, out["ticker"])
            return f"OK: {out['ticker']} dado de alta. companies: {out['before']} -> {out['after']}. Años: {get_config_years()}. Ahora pide confirmación 2 para ejecutar ingest_10k año por año."
        except Exception as exc:
            return f"Error add_company_to_config: {exc}"

    @_tool
    def stock_price(ticker: str) -> str:
        """Precio actual de la acción vía Finnhub (precio, cambio, % cambio, máximo/mínimo del día, cierre previo). Úsala SIEMPRE que pregunten cuánto cotiza/valen las acciones; NUNCA des precios de memoria. Sin FINNHUB_API_KEY en .env devuelve cómo conseguirla."""
        try:
            from src.agent.market_data import MISSING_KEY_MSG, MissingFinnhubKeyError, get_quote

            try:
                q = get_quote(ticker)
            except MissingFinnhubKeyError:
                return MISSING_KEY_MSG
            except RuntimeError as exc:
                return f"Error stock_price: {exc}"
            if q.get("current") is None:
                return f"Finnhub no devolvió cotización para {q['ticker']}. Verifica el ticker."
            chg = q.get("change")
            pct = q.get("change_pct")
            chg_s = f"{chg:+g}" if isinstance(chg, (int, float)) else "n/d"
            pct_s = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else "n/d"
            return (
                f"{q['ticker']}: ${q['current']} ({chg_s} / {pct_s}). "
                f"Día: máx ${q['day_high']} / mín ${q['day_low']}. Cierre previo: ${q['prev_close']}."
            )
        except Exception as exc:
            return f"Error stock_price: {exc}"

    @_tool
    def company_news(company: str, days: int = 7) -> str:
        """Noticias recientes de una empresa vía Finnhub (titular, fecha, fuente, URL y resumen corto, ya filtradas por relevancia). Úsala SIEMPRE que pidan novedades/noticias; NUNCA inventes titulares. Lista como máximo 5-6 con resumen de UNA línea cada una. Sin FINNHUB_API_KEY en .env devuelve cómo conseguirla."""
        try:
            from src.agent.market_data import MISSING_KEY_MSG, MissingFinnhubKeyError, get_company_news

            ticker = (company or "").strip().upper()
            try:
                items = get_company_news(ticker, days=days)
            except MissingFinnhubKeyError:
                return MISSING_KEY_MSG
            except RuntimeError as exc:
                return f"Error company_news: {exc}"
            if not items:
                return f"Sin noticias recientes para {ticker} en los últimos {days} días según Finnhub."
            lines = [f"Noticias de {ticker} (Finnhub):"]
            for n in items:
                lines.append(f"- [{n['date']}] {n['headline']} ({n['source']}) — {n['url']}")
                if n["summary"]:
                    lines.append(f"  {n['summary']}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error company_news: {exc}"

    @_tool
    def suggest_companies(contexto: str = "") -> str:
        """Suggest up to 5 companies with VERIFIED 10-K not in the DB. Mixes Finnhub peers of your indexed tickers + COMPETES_WITH graph rivals (+ `contexto` candidates for a sector). Every candidate passes EDGAR 10-K verification. Present as a numbered list, one item per candidate WITH its details (never bare tickers); let the USER choose; adding follows the double HITL."""
        try:
            from src.agent.company_registry import (
                get_config_tickers,
                load_universe,
                resolve_company,
                verify_10k,
            )
            from src.agent.market_data import get_company_profile, get_peers

            indexed = {t.upper() for t in get_config_tickers()}
            universe = load_universe()
            ordered: list[tuple[str, str]] = []  # (ticker, evidencia)
            seen: set[str] = set()

            def _push(ticker: str, evidencia: str) -> None:
                tu = (ticker or "").strip().upper()
                if tu and tu not in indexed and tu not in seen:
                    seen.add(tu)
                    ordered.append((tu, evidencia))

            # (a) Peers Finnhub de las indexadas
            for t in sorted(indexed):
                try:
                    for p in get_peers(t):
                        _push(p, f"Finnhub peer of {t}")
                except Exception:
                    continue
            # (b) Competidores del grafo fuera de la BD
            try:
                for t in sorted(indexed):
                    for f in rp.graph_facts_retriever.search(f"{t} competitors", top_k=30):
                        if "COMPETES_WITH" in f:
                            _push(f.split("--COMPETES_WITH-->")[-1].strip(), f"graph rival of {t}")
            except Exception as exc:
                logger.debug("suggest_companies: grafo omitido: %s", exc)
            # (c) Contexto sectorial del usuario: intenta resolverlo en el universo SEC
            if (contexto or "").strip():
                try:
                    res = resolve_company(contexto)
                    for c in res.get("candidates", [])[:5]:
                        _push(c["ticker"], f"candidato para {contexto!r} (universo SEC)")
                except Exception:
                    pass

            # Verificación 10-K en EDGAR (filtro de verdad) hasta 5 OK.
            # Formato lista con ficha legible para el usuario.
            picked: list[str] = []
            for tu, evidencia in ordered:
                if len(picked) >= 5:
                    break
                info = universe.get(tu)
                if not info:
                    continue  # no es filer USA: no verificable, se descarta
                try:
                    v = verify_10k(info["cik"], tu)
                except Exception:
                    continue
                if not v.get("ok"):
                    continue
                dates = ", ".join(v.get("recent_10k", [])[:2])
                prof = get_company_profile(tu)
                name = prof.get("name") or info.get("name", "")
                # Acortar nombres de bolsa largos
                exchange = prof.get("exchange", "")
                _EXCHANGE_SHORT = {
                    "NEW YORK STOCK EXCHANGE, INC.": "NYSE",
                    "NEW YORK STOCK EXCHANGE": "NYSE",
                    "NASDAQ NMS - GLOBAL MARKET": "NASDAQ",
                    "NASDAQ CAPITAL MARKET": "NASDAQ",
                    "Nasdaq Global Select": "NASDAQ",
                }
                exchange = _EXCHANGE_SHORT.get(exchange, exchange)
                industry = prof.get("industry", "")
                cap = prof.get("cap", "")
                # Línea principal
                n = len(picked) + 1
                line1 = f"{n}. **{tu}** — {name}" if name else f"{n}. **{tu}**"
                details = " · ".join(p for p in [industry, f"Cap: {cap}" if cap else "", exchange] if p)
                line2 = f"   {details}" if details else ""
                line3 = f"   10-K: {dates}" if dates else ""
                entry = "\n".join(l for l in [line1, line2, line3] if l)
                picked.append(entry)
            if not picked:
                return "No verified candidates right now. Ask the user for a sector or a name to narrow down (propose_new_company)."
            lines = [
                "Verified companies with 10-K available:\n",
                *picked,
                "\n---",
                "INTERNAL — do NOT include this section in your reply. Present the candidates above with their details and ask the user which one(s) to add. After they choose: add_company_to_config, then ingest_10k.",
            ]
            return "\n".join(lines)
        except Exception as exc:
            return f"Error suggest_companies: {exc}"

    @_tool
    def lookup_company(ticker: str) -> str:
        """Ficha de una empresa: nombre, bolsa, industria, market cap, web y estado 10-K en EDGAR (+ si ya está en tu BD). Úsala cuando el usuario pida detalles de una candidata SUGERIDA o de cualquier ticker (NO uses company_news para esto: eso son titulares recientes, no la ficha). SOLO LECTURA."""
        try:
            from src.agent.company_registry import get_config_tickers, load_universe, verify_10k
            from src.agent.market_data import get_company_profile

            t = (ticker or "").strip().upper()
            universe = load_universe()
            info = universe.get(t, {})
            prof = get_company_profile(t)
            name = prof.get("name") or info.get("name", t)
            parts = [p for p in [
                prof.get("industry", ""),
                f"cap {prof['cap']}" if prof.get("cap") else "",
                prof.get("exchange", ""),
                prof.get("web", ""),
            ] if p]
            detail = f" ({' · '.join(parts)})" if parts else ""
            indexed = "SÍ, ya indexada" if t in {c.upper() for c in get_config_tickers()} else "no indexada"
            if info.get("cik"):
                try:
                    v = verify_10k(info["cik"], t)
                    filing = f"10-K recientes {', '.join(v.get('recent_10k', [])[:2])}" if v.get("ok") else f"sin 10-K: {v.get('reason', '')}"
                except Exception:
                    filing = "10-K no verificable ahora mismo"
            else:
                filing = "fuera del universo SEC (sin 10-K USA verificable)"
            return f"{t} ({name}){detail} — en tu BD: {indexed} · {filing}."
        except Exception as exc:
            return f"Error lookup_company: {exc}"

    # Registro extensible: futuras tools (web_search...) se añaden a esta lista
    return [query_financial_rag, lookup_metrics, financial_calculator, propose_new_company, add_company_to_config, ingest_tool, stock_price, company_news, suggest_companies, lookup_company]


def _rest_after_json_block(s: str) -> str | None:
    """Si `s` (ya sin espacios iniciales) abre con `{`, devuelve el texto tras
    el bloque JSON balanceado, o None si aún está incompleto (sin cerrar)."""
    if not s.startswith("{"):
        return s
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Solo lstrip: el espacio tras el bloque une con el siguiente token.
                return s[i + 1 :].lstrip()
    return None


def strip_leading_json_block(text: str) -> str:
    """Elimina un bloque JSON inicial (volcado crudo de tool) si después hay texto redactado.

    Red de seguridad para modelos pequeños que pegan el resultado de la tool
    antes de redactar. Si toda la respuesta es JSON, se deja intacta.
    """
    s = (text or "").lstrip()
    rest = _rest_after_json_block(s)
    if rest is None:
        return text
    return rest if len(rest) > 20 else text


class JsonPrefaceFilter:
    """Filtro con estado para streaming: retiene los tokens iniciales mientras
    parezcan JSON y solo libera prosa.

    En streaming no se puede "des-imprimir", así que ante un preámbulo
    `{"answer": ...}` se contiene la salida hasta cerrar el bloque y se
    muestra únicamente lo redactado. Uso: `feed(token)` por fragmento y
    `flush()` al terminar el turno.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._live = False

    def feed(self, token: str) -> str:
        if self._live:
            return token
        self._buf += token or ""
        if not self._buf.lstrip().startswith("{"):
            self._live = True
            out, self._buf = self._buf, ""
            return out
        rest = _rest_after_json_block(self._buf.lstrip())
        if not rest:
            # Bloque aún incompleto, o cerrado pero sin prosa detrás:
            # retener (puede llegar texto redactado en tokens posteriores).
            return ""
        self._live = True
        self._buf = ""
        return rest

    def flush(self) -> str:
        if self._live:
            out, self._buf = self._buf, ""
            return out
        out = strip_leading_json_block(self._buf)
        self._buf = ""
        self._live = True
        return out


def _dynamic_ticker_pattern() -> "re.Pattern":
    """Regex construida desde tickers base + companies.json (sin curado manual)."""
    tickers = sorted(_allowed_tickers(), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b", re.IGNORECASE)


def extract_ticker_year_fallback(question: str) -> tuple[Optional[str], Optional[int]]:
    """Fallback regex si la clasificación LLM no da ticker/year."""
    m = _dynamic_ticker_pattern().search(question or "") or _TICKER_RE.search(question or "")
    ticker = m.group(1).upper() if m else None
    from src.agent.company_registry import _ticker_key

    if ticker:
        by_key = {_ticker_key(t): t for t in _allowed_tickers()}
        ticker = by_key.get(_ticker_key(ticker), ticker)
    y = None
    ym = re.search(r"\b(20(?:2[0-9]|19))\b", question or "")
    if ym:
        try:
            y = int(ym.group(1))
        except Exception:
            pass
    return ticker, y
