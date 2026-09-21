# Financial GraphRAG Engine

Sistema de **preguntas y respuestas financieras** sobre informes anuales **10-K de la SEC** (las cuentas que las empresas cotizadas de EE. UU. presentan al regulador). Combina tres formas de recuperar información —**búsqueda vectorial densa, búsqueda léxica BM25 y grafo de conocimiento**— y genera respuestas con **citas a los fragmentos originales**. Con **dos agentes LangGraph** locales (`qwen3:8b`): determinista con **memoria Q/A** y **ReAct** con **10 tools** (retrieval RAG, métricas scoping, calculadora financiera, alta SEC, sugerencias de competidores con 10-K verificado, ficha corporativa y cotizaciones/noticias en tiempo real vía Finnhub API), **servidor web FastAPI con streaming SSE, barras de progreso de ingesta en vivo, cancelación cooperativa sin pérdida de datos e interfaz tipo ChatGPT** y tabla de métricas scoping `TICKER_YEAR`.

Ejemplo de lo que responde:

> **¿En qué segmentos opera MSFT?**
> *Productivity and Business Processes, Intelligent Cloud y More Personal Computing* — con citas a los chunks `Item 7` / `Reportable Segments` y hechos del grafo (`MSFT --OPERATES_IN--> Intelligent Cloud`).
>
> **What was AAPL revenue in 2024?**
> *$391,035 million* — fila `| AAPL | 2024 | total net sales | 391035 USD millions | Item 8 | chunk_id |` scoping `AAPL_2024_total_net_sales`.
>
> **¿A cuánto cotiza NVDA y qué noticias recientes hay?**
> *$119.10 (+2.15 / +1.84%)* con titulares recientes filtrados por relevancia y enlaces directos vía Finnhub.

## Cómo funciona

```
SEC EDGAR 10-K (PDF/HTML)             Finnhub REST API (en vivo)
        │  sec-edgar-downloader                 │ (cotizaciones + noticias)
        ▼                                       ▼
Ingesta: PDF/HTML → Markdown → secciones  ┌─────────────────────────┐
        │  chunker table-aware ~600 tok   │ stock_price, company_news│
        ▼                                 │ suggest, lookup_company │
┌──────────────┬──────────────┬───────────┴──────────────┐          │
│   LanceDB    │    BM25      │      Kùzu (grafo)        │          │
│  (bge-m3)    │ (sparse idx) │  tripletas LLM + caché   │          │
│  denso       │  léxico      │  en triplets.json        │          │
└──────┬───────┴──────┬───────┴──────────┬───────────────┘          │
       │              │                  │                          │
       └──────────────┼──────────────────┘                          │
                      ▼                                             │
        Dos modos de agente (100% local, memoria Q/A):              │
        A) Determinista: classify_intent → ingest_tool HITL | retrieve
        B) ReAct (--react y web API): el modelo decide tools
           tools: query_financial_rag | lookup_metrics | financial_calculator
                  propose_new_company | add_company_to_config (HITL 1) | ingest_10k (HITL 2)
                  stock_price | company_news | suggest_companies | lookup_company
           HITL moderno nativo con interrupt() dentro de las tools y reanudación con Command(resume=...)
           Filtro de preámbulo JSON en streaming (JsonPrefaceFilter)
           Canales: CLI interactivo y Servidor Web FastAPI (SSE + barras de progreso + cancelación + historial)
        retrieve: expanded query (revenue→net sales) → dense/sparse/graph+metrics_table scoping
        RRF (k=60) → grounding por ticker → dedup
                    → reranker cross-encoder (bge-reranker-v2-m3)
                    → VRAM empty_cache antes de generación
                      ▼
        LLM + generación con citas + graph facts + METRICS TABLE
        (OPERATES_IN/COMPETES_WITH como facts; métricas como tabla | ticker | year | metric | value |)
        Alta de empresas: universo oficial SEC (company_tickers.json) + verificación 10-K en EDGAR,
        sin mapas curados; ticker literal exacto → vía rápida, resto → pregunta al usuario
```

1. **Ingesta** (`src/ingestion/`): descarga el 10-K, lo pasa a Markdown (tablas HTML → `| col |`), lo trocea **table-aware** (no parte filas `| | |`) con metadatos (`ticker`, `año`, `sección`, `página`) y lo guarda en `data/processed_chunks/<TICKER>_<AÑO>/chunks.json` (table-aware desde el último rebuild).
2. **Grafo** (`src/graph/`): LLM (`qwen3:8b` `reasoning=False, format=json, num_ctx 8192`) extrae tripletas `(origen, relación, destino)` con `value/unit/year` para `FinancialMetric` según ontología. PK de métrica es **compuesta** `TICKER_YEAR_slug` (`AAPL_2024_total_net_sales`) con columna `year` inferida por prompt few-shot (`2024 | 2023` → 2 triplets) + fallback regex por proximidad. Se cachea en `triplets.json`, se normalizan tickers (`Apple→AAPL`, `AM,ZN→AMZN`), se filtran ruidos y se persiste en **Kùzu** con `MERGE + seen-set`.
3. **Recuperación** (`src/retrieval/`): cada pregunta pasa por `expand_query` sinónimos, consulta los tres índices en paralelo + `MetricsTable` scoping (`c.ticker IN $tickers AND m.id CONTAINS '_'`), fusiona con **RRF**, filtra por ticker, dedup, reordena con **cross-encoder** y genera con **citas** (`CHUNK_ID` + `METRICS TABLE` con `section`).
4. **Agentes** (`src/agent/`):
   - **Determinista**: `StateGraph` `classify_intent (with_structured_output IntentOutput ingest|retrieve) → parallel_retrieve → fuse_rerank (torch.cuda.empty_cache) → generate`. `ingest_10k(ticker,year)` con **HITL** `MemorySaver interrupt_before ingest_tool` y confirmación `y/n` en CLI. Memoria conversacional solo `Q/A` (no chunks, `4×500 chars`).
   - **ReAct** (`--react`, `react_graph.py`, system prompt en inglés para `qwen3:8b`): `create_react_agent` con **10 tools** — `query_financial_rag` (retrieval completo), `lookup_metrics` (cifras scoping), `financial_calculator` (`yoy_pct|pct_change|diff|ratio|sum|avg`), `propose_new_company` (universo SEC + 10-K EDGAR), `add_company_to_config` (**HITL 1**: editar `companies.json` con `.bak`), `ingest_10k` (**HITL 2**: ingesta con progreso y cancelación cooperativa), `stock_price` (cotización en tiempo real vía Finnhub), `company_news` (noticias financieras con enlaces directos), `suggest_companies` (sugiere hasta 5 candidatas competidoras con 10-K verificado en EDGAR) y `lookup_company` (ficha corporativa con bolsa, capitalización y estado del 10-K). HITL nativo vía `interrupt()` y `Command(resume=...)`. Saneamiento de historial ante cancelaciones con `_repair_dangling_tool_calls`.
   - **Alta de empresas** (`company_registry.py`, sin mapas curados): universo oficial SEC cacheado (`data/sec/company_tickers.json`, TTL 30 días) + difusa `difflib`; ticker literal exacto → vía rápida sin pregunta; resto → candidatos y pregunta obligatoria al usuario antes de buscar documentos; índices/filiales sin 10-K se explican y no se dan de alta. Las listas de tickers de ingesta/retrieval/grafo se construyen desde `companies.json` + universo SEC.
5. **Evaluación** (`evals/`): `51 Q/A` (15 viejas fuera de corpus `2023` + 36 nuevas `2024-2025` YoY `revenue/segments/risks`) y checks deterministas `answer, citas, ticker/año/sección, dense/sparse/graph, metrics_scoping` con gate `0.85` (sin LLM-juez local).

## Estado actual de los datos

Empresas `data/companies.json` `2024–2025`: `AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, BRK.B`.

* **Ingesta completa tabla-aware 2026-09-09:** `3067` chunks (`129-292` por filing, antes `1700`), `~19k` tripletas en cache.
* **Grafo (tras limpieza 5180 viejos):** `7907` `FinancialMetric` (`7837` con `value`, `id` scoping), `Company 216`, `BusinessSegment ~1273`, `216` `COMPETES_WITH`-like; `DocumentChunk 3067`; `REPORTED_METRIC 7923`, `OPERATES_IN ~1380`, `MENTIONS_EVENT ~2038`. `AMZN_2024` y `MSFT_2025` ya sin huecos.
* **Vector:** `LanceDB 1928→3067 rows`, `BM25` rebuild desde Lance, `bge-m3` + `bge-reranker-v2-m3`.

> **Aviso:** El contenido generado de `data/` (`10-K`, `chunks`, `lancedb`, `kuzu_db`) no está en el repo (`.gitignore`) y pesa `>500 MB` (solo se incluye la configuración inicial `data/companies.json`). Al clonar, `python cli.py --ingest --workers 4 --batch-size 1` tarda horas (extracción `workers 4 batch 1` + `reasoning=False` es el path estable para 12GB), los workers y batches dependerán del hardware. `docs/` (tutoriales y apuntes locales) también está ignorado (`.gitignore`).

## Instalación

```bash
git clone <repo-url>
cd financial-graphrag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt  # incluye langchain, langgraph
```

Configuración LLM (`.env`):

```bash
# Opción 1 (por defecto): Ollama local
ollama pull qwen3:8b        # DEFAULT_OLLAMA_MODEL = "qwen3:8b" (reasoning=False, format=json, num_ctx 8192)
# Para 12GB: OLLAMA_NUM_PARALLEL=2

# Opción 2 fallback si Ollama no responde: Groq
export GROQ_API_KEY="gsk_..."
# Opcional HF_TOKEN para bge-m3
export HF_TOKEN="hf_..."
# Datos de mercado y noticias (tools stock_price y company_news del agente ReAct):
# key gratuita en https://finnhub.io/register (60 llamadas/minuto)
export FINNHUB_API_KEY="c1...x9"
```

> `create_llm()` usa Ollama y cae a Groq. Generación/extracción requieren LLM; `dense/BM25/grafo` funcionan con índice ya construido.

## Uso

### Chat interactivo (CLI con agente)

```bash
python cli.py                          # agente LangGraph determinista + memoria Q/A
python cli.py --react                  # agente ReAct (el modelo decide tools) + doble HITL
python cli.py --no-agent               # pipeline directo sin agente
python cli.py --workers 2 --batch-size 1  # estable 12GB (default 4→2)
python cli.py --ticker AAPL --year 2024
python cli.py --ingest
```

Dentro del chat:

```
<pregunta>             -> retrieval RAG (ej: What was AAPL revenue in 2024?)
¿y en 2023?            -> follow-up usa memoria (history_ticker AAPL)
añade AAPL 2026        -> clasifica ingest → ¿Confirmas ingesta AAPL 2026? (y/n) → HITL Tool
I want to know about nasdaq / Fluence Energy  -> (modo --react) propone alta vía SEC → ¿Confirmas editar companies.json? (y/n) → ¿Confirmas ingesta? (y/n)
/ingest <ticker> <año> -> ingesta directa sin agente
/ingest-all            -> data/companies.json
/clear                 -> limpia memoria conversacional (nuevo thread_id)
/help, /exit
```

La ingesta del agente usa los mismos `workers/batch` del arranque (van con el pipeline, no son propios del agente). El alta escribe `companies.json` dejando backup `companies.json.bak`.

En modo `--react`, el CLI implementa **streaming en tiempo real** (`stream_mode="messages"`): anuncia dinámicamente cada herramienta al invocarse (`>> tool_name...`) y emite los tokens de respuesta en vivo, coordinándose con `Command(resume=...)` cuando salta un `interrupt()`. La arquitectura interna del CLI desacopla los comandos con una tabla de despacho (`_COMMANDS`) y un registro extensible de avisos (`INTERRUPT_HANDLERS`).

Cada respuesta muestra `[Facts: N | Metrics: N | Citations: N]` (truncado a 5), citas `chunk_id/ticker/año/sección/score` y `METRICS TABLE` scoping si aplica.

### API web local (FastAPI + página mínima, agente ReAct con streaming)

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt  # fastapi + uvicorn[standard]
.venv/Scripts/python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
# Abrir http://localhost:8000
```

Sin `--reload` a propósito: recargar duplicaría el modelo en VRAM. Un solo pipeline/agente compartidos (`GRAPH_WORKERS=2` opcional, por defecto 2); Kuzu es un solo escritor y los turnos se serializan con lock.

Endpoints (`api.py`):

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | `{status, model, tickers}` |
| `GET` | `/` | Chat SPA (`static/index.html`): sidebar de chats, tokens en vivo, barras de progreso por año, botón Detener, tarjetas HITL |
| `POST` | `/query {question, thread_id?}` | SSE (`start` / `token` / `tool` / `progress` / `interrupt` / `cancelled` / `done` / `error`) con el bucle ReAct del CLI |
| `POST` | `/confirm {thread_id, decision}` | Reanuda con `Command(resume=...)` tras un `interrupt` (`confirm_add_company`, `confirm_ingest`) |
| `POST` | `/cancel {thread_id}` | Detiene cooperativamente una ingesta en curso guardando el progreso en disco para retomarla después |
| `GET` | `/conversations` | Lista conversaciones guardadas en `data/conversations.json` ordenadas por fecha |
| `GET` | `/conversations/{thread_id}` | Obtiene el historial completo de mensajes y título de un thread |
| `DELETE` | `/conversations/{thread_id}` | Elimina una conversación del almacenamiento |

**Progreso y Cancelación Cooperativa**:
- Durante la ingesta de un 10-K, el pipeline emite eventos SSE `progress` (`download`, `parse`, `cache`, `extract`, `index`, `done`) que dibujan una barra de porcentaje en tiempo real por cada par `ticker/año`.
- Si el usuario pulsa **Detener** (`POST /cancel`), el backend eleva `IngestCancelled` (que hereda de `BaseException` para no ser capturada por reintentos de red), **cancela las tareas pendientes y sincroniza en disco las tripletas calculadas hasta ese segundo**.
- Al pedir *"continúa la ingesta"*, el sistema detecta los chunks ya persistidos en `triplets.json` y continúa sin repetir trabajo.
- Para evitar que cancelaciones abruptas dejen `tool_calls` sin resolver (lo que causaría un fallo `INVALID_CHAT_HISTORY` en LangGraph), `api.py` repara automáticamente el historial inyectando un `ToolMessage` sintético de cancelación antes del siguiente turno (`_repair_dangling_tool_calls`).
- Las conversaciones se persisten en disco de forma atómica y segura entre hilos (`data/conversations.json`). Al reiniciar el servidor API, el historial se reinyecta automáticamente en el `MemorySaver` de LangGraph.

```bash
curl -N -X POST localhost:8000/query -H "Content-Type: application/json" -d "{\"question\":\"¿a cuánto cotiza AAPL?\"}"
curl -X POST localhost:8000/confirm -H "Content-Type: application/json" -d "{\"thread_id\":\"...\",\"decision\":\"y\"}"
curl -X POST localhost:8000/cancel -H "Content-Type: application/json" -d "{\"thread_id\":\"...\"}"
```

### Desde Python

```python
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

llm = create_llm()  # reasoning=False
pipeline = FinancialGraphRAGPipeline(llm=llm, graph_max_workers=2, graph_batch_size=1)
pipeline.ingest_and_index(ticker="AAPL", year=2024)  # use_cache=False para re-extraer con year
result = pipeline.query("Which segments does MSFT operate in?")
print(result.answer)
print(result.citations)
print(result.metrics_rows)  # scoping
pipeline.close()

# Agente directo
from src.agent.graph import build_agent_graph
agent = build_agent_graph(pipeline)
agent.invoke({"question": "What was AAPL revenue in 2024?"}, config={"configurable":{"thread_id":"t1"}})

# Agente ReAct (10 tools + doble HITL)
from src.agent.react_graph import build_react_agent
from langchain_core.messages import HumanMessage
react = build_react_agent(pipeline)  # usa create_llm(json_mode=False) para tool_calls
react.invoke({"messages": [HumanMessage(content="YoY de AAPL revenue 2024 vs 2023")]},
             config={"configurable": {"thread_id": "t2"}})
```

### Reprocesar huecos

```bash
python reprocess_missing.py --workers 2 --batch-size 1
# detecta triplets parciales (cached_ids < len(chunks)) y fuerza re-ingesta si chunks vacío (AMZN_2024)
```

### Evaluación y tests

```bash
python evals/run_checks.py --samples 5            # gate 0.85, 9 checks (+metrics_scoping)
python evals/run_checks.py                         # 51 Q/A (~6 min con qwen3:8b)
& ".\.venv\Scripts\python.exe" evals/run_checks.py # en Windows con .venv
# pytest opcional (no en requirements por defecto)
pip install pytest && python -m pytest tests/ -v
```

## Estructura del repositorio

```
financial-graphrag/
├── api.py                      # Servidor FastAPI (SSE streaming, progreso, HITL wait/resume, cancelación, CRUD historial)
├── cli.py                      # REPL agente LangGraph (Streaming en vivo, Dispatch table, HITL Command(resume=...))
├── reprocess_missing.py        # Detecta faltantes/parciales y ingesta con workers/batch (soporta reanudar/limpio con locks)
├── static/
│   └── index.html              # Frontend Web SPA (ChatGPT-style, SSE, barras de progreso por año, botón Detener, tarjetas HITL)
├── data/                       # Datos y persistencia
│   ├── companies.json          # Registro de empresas seguidas (con .bak automático)
│   ├── conversations.json      # Historial persistente de chats (atómico / thread-safe)
│   ├── raw_10k/                # full-submission.txt + .download_cache.json (ignorado)
│   ├── processed_chunks/       # chunks.json + triplets.json (PK TICKER_YEAR_metric, ignorado)
│   ├── vector_store/lancedb/   # LanceDB bge-m3 (ignorado)
│   └── graph/kuzu_db/          # Kuzu DB (ignorado)
├── src/
│   ├── agent/                  # Determinista + ReAct (100% local)
│   │   ├── state.py            # AgentState (messages Q/A, no chunks)
│   │   ├── tools.py            # 10 tools ReAct + interrupt() nativo + calculator + SEC registry + Finnhub + suggestions
│   │   ├── progress.py         # Monitor de progreso por thread, cancelación cooperativa IngestCancelled y suscripción SSE
│   │   ├── market_data.py      # Cliente Finnhub (cotizaciones, noticias, peers, perfiles, resolución URLs, caché 60s)
│   │   ├── history_store.py    # Persistencia JSON atómica de conversaciones + reinyección en LangGraph
│   │   ├── company_registry.py # Universo oficial SEC + resolve difuso + verify_10k EDGAR + add_company (.bak)
│   │   ├── nodes.py            # classify_intent with_structured_output + parallel_retrieve + fuse_rerank + generate + history
│   │   ├── graph.py            # StateGraph determinista + MemorySaver interrupt_before ingest_tool
│   │   └── react_graph.py      # create_react_agent (prompt EN) + MemorySaver (HITL nativo vía interrupt() en tools)
│   ├── env.py                  # Carga de variables de entorno (.env)
│   ├── llm_factory.py          # ChatOllama qwen3:8b reasoning=False (json_mode=True → format=json; False → tool_calls nativos)
│   ├── pipeline.py             # FinancialGraphRAGPipeline
│   ├── ingestion/              # downloader, parser (HTML tables → | |), chunker table-aware, pipeline (reporta progreso)
│   ├── graph/                  # schema, extractor, graph_pipeline (PK compuesto, checkpoints incrementales y progreso)
│   └── retrieval/              # dense, sparse, graph_traversal, graph_facts, metrics_table scoping, rrf, reranker, generator
├── evals/
│   ├── test_dataset.json       # 51 Q/A (2024-2025 YoY)
│   └── run_checks.py           # 9 checks + gate 0.85 + metrics scoping
└── tests/
```

## Ontología del grafo

| Nodo | Clave | Descripción |
|---|---|---|
| `Company` | `ticker` | `AAPL, BRK.B` canónico |
| `FinancialMetric` | `id = TICKER_YEAR_slug` | Métrica scoping `AAPL_2024_total_net_sales` con `value/unit/fiscal_year` |
| `RiskFactor` | `id` | Riesgo Item 1A |
| `BusinessSegment` | `id` | Segmento reportable |
| `MacroEvent` | `id` | Evento macro |
| `DocumentChunk` | `chunk_id` | `ticker/año/sección/página` |

| Relación | Origen → Destino |
|---|---|
| `OPERATES_IN` | Company → BusinessSegment |
| `REPORTED_METRIC` | Company → FinancialMetric (scoping) |
| `IMPACTS_REVENUE` | MacroEvent → FinancialMetric |
| `MITIGATES_RISK` | BusinessSegment → RiskFactor |
| `COMPETES_WITH` | Company → Company |
| `MENTIONS_*` | DocumentChunk → entidad |

Notas:

* `FinancialMetric` ya no colisiona (`revenue` → `AAPL_2024_revenue` vs `MSFT_2024_revenue`) y tablas `2024|2023` generan 2 triplets con `year` por columna (prompt few-shot + fallback regex). `GENERATION` usa `METRICS TABLE scoping` no `graph_facts` para cifras.
* `chunker` respeta `| tablas |` y `parser` convierte `HTML <table>` a `| col |`.

## Stack

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Ingesta | sec-edgar-downloader, pdfplumber, markdownify, unstructured, bs4 |
| Embeddings | BAAI/bge-m3 (sentence-transformers) |
| Vector store | LanceDB |
| Léxico | BM25 (rank-bm25) |
| Grafo | Kùzu (MERGE scoping) |
| LLM | Ollama `qwen3:8b` `reasoning=False, num_ctx 8192` (`format=json` en determinista/extractor; `tool_calls` nativos en ReAct) → Groq fallback |
| Framework | LangChain / LangGraph (StateGraph determinista + ReAct `create_react_agent`, ToolNode HITL) |
| Web API & UI | FastAPI, Uvicorn, Server-Sent Events (SSE), HTML5/Tailwind/CSS |
| APIs Externas | Finnhub REST API (cotizaciones en tiempo real, noticias financieras) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Comunidades | Leiden + igraph + networkx |
| Evaluación | Checks deterministas gate 0.85 + metrics_scoping (`evals/run_checks.py`) |

## Licencia

MIT
