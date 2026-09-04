# Financial GraphRAG Engine

Sistema de **preguntas y respuestas financieras** sobre informes anuales **10-K de la SEC** (las cuentas que las empresas cotizadas de EE. UU. presentan al regulador). Combina tres formas de recuperar información —**búsqueda vectorial densa, búsqueda léxica BM25 y grafo de conocimiento**— y genera respuestas con **citas a los fragmentos originales**.

Ejemplo de lo que responde:

> **¿En qué segmentos opera MSFT?**
> *Productivity and Business Processes, Intelligent Cloud y More Personal Computing* — con citas a los chunks `Item 7` / `Reportable Segments` y hechos del grafo (`MSFT --OPERATES_IN--> Intelligent Cloud`).

## Cómo funciona

```
SEC EDGAR 10-K (PDF/HTML)
        │  sec-edgar-downloader
        ▼
Ingesta: PDF/HTML → Markdown → secciones (Item 1, 1A, 7, 8…)
        │  chunk ~600 tokens / overlap 90
        ▼
┌──────────────┬──────────────┬──────────────────────────┐
│   LanceDB    │    BM25      │      Kùzu (grafo)        │
│  (bge-m3)    │ (sparse idx) │  tripletas LLM + caché   │
│  denso       │  léxico      │  en triplets.json        │
└──────┬───────┴──────┬───────┴──────────┬───────────────┘
       │              │                  │
       └──────────────┼──────────────────┘
                      ▼
        RRF (k=60) → grounding por ticker → dedup
                      → reranker cross-encoder (bge-reranker-v2-m3)
                      ▼
        LLM + generación con citas + graph facts
        (segmentos y competidores del grafo)
```

1. **Ingesta** (`src/ingestion/`): descarga el 10-K, lo pasa a Markdown, lo trocea en chunks con metadatos (`ticker`, `año`, `sección`, `página`) y los guarda en `data/processed_chunks/<TICKER>_<AÑO>/chunks.json`.
2. **Grafo** (`src/graph/`): un LLM extrae tripletas `(origen, relación, destino)` según una ontología fija y se guardan en `triplets.json` como caché. Al persistir en Kùzu se **normalizan tickers** (`Apple→AAPL`, typos OCR `AM,ZN→AMZN`), se **filtran nodos ruidosos** y se indexan menciones `DocumentChunk → entidad` para **origen y destino**.
3. **Recuperación** (`src/retrieval/`): cada pregunta consulta los tres índices en paralelo, fusiona con RRF, filtra por ticker mencionado, reordena con cross-encoder y genera la respuesta con citas.
4. **Evaluación** (`evals/`): checks deterministas sobre un dataset de preguntas — respuesta no vacía, citas con ticker/año/sección correctos y cobertura de las tres vías de recuperación. Sin LLM-juez, para que sea estable con modelos locales pequeños.

## Estado actual de los datos

Empresas configuradas en `data/companies.json` (años 2024–2025): `AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, BRK.B`.

* ~1.700 chunks y ~11.700 tripletas en caché.
* Grafo (tras la última reconstrucción desde caché): ~180 compañías (incluye filiales como `BHE`, `Marmon`), ~4.700 métricas, ~1.700 macroeventos, ~1.200 segmentos; `IMPACTS_REVENUE ~2.300`, `REPORTED_METRIC ~4.700`, `MENTIONS_EVENT ~2.000`.
* Huecos conocidos: `AMZN_2024` sin chunks y `MSFT_2025` sin procesar (ver `reprocess_missing.py`).

> **Aviso:** la carpeta `data/` (10-K descargados, chunks, índices LanceDB y grafo Kùzu) **no está en el repositorio** — pesa cientos de MB y está ignorada por git. Al clonar empezarás sin datos y tendrás que ingerir desde cero (`python cli.py --ingest` o `/ingest-all` en el chat), lo que con LLM local puede tardar varias horas (la extracción de tripletas es lo más costoso).

## Instalación

```bash
git clone <repo-url>
cd financial-graphrag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Configuración del LLM (en `.env` o variables de entorno):

```bash
# Opción 1 (por defecto): Ollama local
ollama pull qwen3:8b        # DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# Opción 2 (fallback automático si Ollama no responde): Groq
export GROQ_API_KEY="gsk_..."   # DEFAULT_GROQ_MODEL = "mixtral-8x7b-32768"

# Opcional: token de HuggingFace para embeddings bge-m3
export HF_TOKEN="hf_..."
```

> `create_llm()` usa Ollama si está disponible y cae a Groq si no. Sin ninguno de los dos, la ingesta/extracción y la generación no funcionan (la búsqueda densa/BM25/grafo sí, una vez indexado).

## Uso

### Chat interactivo (CLI)

```bash
python cli.py
python cli.py --ticker AAPL --year 2024   # ingiere al arrancar
python cli.py --ingest                    # ingiere todo data/companies.json
```

Dentro del chat:

```
<pregunta>             -> consulta al pipeline RAG
/ingest <ticker> <año> -> ingiere e indexa un 10-K (ej: /ingest AAPL 2024)
/ingest-all            -> ingiere todas las empresas de data/companies.json
/clear, /help, /exit
```

Cada respuesta muestra `[Dense | Sparse | Graph | Facts]`, las citas (`chunk_id`, `ticker`, `año`, `sección`, `score`) y los hechos del grafo usados.

### Desde Python

```python
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

llm = create_llm()
pipeline = FinancialGraphRAGPipeline(llm=llm)

pipeline.ingest_and_index(ticker="AAPL", year=2024)  # usa caché si existe
result = pipeline.query("Which segments does MSFT operate in?")
print(result.answer)
print(result.citations)
print(result.graph_facts)
pipeline.close()
```

### Reprocesar huecos

```bash
python reprocess_missing.py   # detecta pares sin triplets.json y los procesa
```

### Evaluación y tests

```bash
python evals/run_checks.py             # checks: respuesta, citas, ticker/año/sección, dense/sparse/graph
python evals/run_checks.py --samples 3 # limita a las 3 primeras preguntas
python -m pytest tests/ -v             # requiere pytest (no incluido en requirements)
```

## Estructura del repositorio

```
financial-graphrag/
├── cli.py                      # Chat REPL (/ingest, /ingest-all, consultas)
├── reprocess_missing.py        # Reprocesa pares ticker/año sin triplets.json
├── data/                       # Generado, ignorado por git
│   ├── raw_10k/                # PDFs/HTML de SEC EDGAR
│   ├── processed_chunks/       # chunks.json + triplets.json por <TICKER>_<AÑO>
│   ├── vector_store/lancedb/   # Índice denso embebido
│   └── graph/kuzu_db/          # Grafo Kùzu embebido
├── src/
│   ├── env.py                  # Carga .env (HF_TOKEN, GROQ_API_KEY)
│   ├── llm_factory.py          # Ollama (qwen3:8b) con fallback a Groq
│   ├── pipeline.py             # FinancialGraphRAGPipeline end-to-end
│   ├── ingestion/              # downloader, parser, chunker, pipeline
│   ├── graph/                  # schema, extractor, graph_pipeline, communities
│   └── retrieval/              # dense, sparse, graph_traversal, graph_facts,
│                               # rrf, reranker, generator, pipeline
├── evals/                      # test_dataset.json + checks deterministas
└── tests/                      # unitarios (reranker) + integración (pipeline)
```

## Ontología del grafo

| Nodo | Clave | Descripción |
|---|---|---|
| `Company` | `ticker` | Empresa (ticker canónico: `AAPL`, `BRK.B`…) |
| `FinancialMetric` | `id` | Métrica numérica (revenue, EPS, total assets…) |
| `RiskFactor` | `id` | Riesgo divulgado en el 10-K |
| `BusinessSegment` | `id` | Segmento operativo/reportable |
| `MacroEvent` | `id` | Evento macroeconómico o geopolítico |
| `DocumentChunk` | `chunk_id` | Fragmento con `ticker`, `año`, `sección`, `página` |

| Relación | Origen → Destino |
|---|---|
| `OPERATES_IN` | Company → BusinessSegment |
| `REPORTED_METRIC` | Company → FinancialMetric |
| `IMPACTS_REVENUE` | MacroEvent → FinancialMetric |
| `MITIGATES_RISK` | BusinessSegment → RiskFactor |
| `COMPETES_WITH` | Company → Company |
| `MENTIONS_*` | DocumentChunk → cada tipo de entidad |

Notas de diseño:

* Los valores numéricos de `FinancialMetric` **no** se exponen como hechos al generador (una métrica comparte nombre entre empresas y el valor sería ambiguo); solo se exponen `OPERATES_IN` y `COMPETES_WITH` como `graph facts`. Los números llegan al LLM vía los chunks citados.
* La persistencia deduplica (MERGE + `seen-set` por chunk), normaliza compañías y descarta ruido (nombres vacíos, genéricos como `Services`/`other companies`, tickers colados en otras tablas).

## Stack

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Ingesta | sec-edgar-downloader, pdfplumber, markdownify, unstructured |
| Embeddings | BAAI/bge-m3 (sentence-transformers) |
| Vector store | LanceDB (embebido) |
| Léxico | BM25 (rank-bm25) |
| Grafo | Kùzu (embebido) |
| LLM | Ollama `qwen3:8b` por defecto · Groq `mixtral-8x7b-32768` como fallback |
| Framework | LangChain / langchain-core |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Comunidades | Leiden (leidenalg + igraph + networkx) |
| Evaluación | Checks deterministas (`evals/run_checks.py`) |

## Licencia

MIT
