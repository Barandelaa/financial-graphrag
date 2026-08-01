# Financial GraphRAG Engine

Plataforma de inteligencia financiera sobre informes **10-K de la SEC** que combina **Búsqueda Vectorial Densa**, **Búsqueda Léxica (BM25)**, **Grafos de Conocimiento (Kùzu)** y **evaluación continua con RAGAS**.

## Arquitectura

```
                    ┌─────────────────────────┐
                    │     SEC EDGAR 10-K       │
                    └──────────┬──────────────┘
                               │ sec-edgar-downloader
                               ▼
                    ┌─────────────────────────┐
                    │   Ingesta y Chunking     │
                    │  (pdfplumber → MD →      │
                    │   chunk 600tok / 90 overlap)│
                    └──────┬──────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  LanceDB   │ │   BM25     │ │   Kùzu DB  │
     │(bge-m3 emb)│ │(sparse idx)│ │ (graph KG) │
     └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    ┌─────────────────────────┐
                    │    RRF (k=60) +          │
                    │  Cross-Encoder Reranker  │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │   LLM + Generación       │
                    │   con Citas               │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RAGAS Evaluation    │
                    │  Faithfulness        │
                    │  Answer Relevancy    │
                    │  Context Precision   │
                    │  Context Recall      │
                    └─────────────────────┘
```

## Estructura del Repositorio

```
financial-graphrag/
├── data/
│   ├── raw_10k/               # Documentos PDF/HTML descargados de SEC EDGAR
│   └── processed_chunks/      # Chunks en JSON con metadatos estructurados
├── src/
│   ├── ingestion/             # Pipelines de descarga, OCR y parsing
│   │   ├── downloader.py      # SEC EDGAR downloader (sec-edgar-downloader)
│   │   ├── parser.py          # PDF/HTML → Markdown → secciones
│   │   ├── chunker.py         # Chunking consciente de estructura
│   │   └── pipeline.py        # Orquestador de ingesta
│   ├── graph/                 # Extracción de tripletes, schema Kùzu y Leiden
│   │   ├── schema.py          # Ontología financiera + DDL de Kùzu
│   │   ├── extractor.py       # Extracción LLM de tripletes con Pydantic
│   │   ├── graph_pipeline.py  # Persistencia de tripletes en Kùzu
│   │   └── communities.py     # Leiden + resúmenes ejecutivos por comunidad
│   ├── retrieval/             # Recuperación híbrida y generación
│   │   ├── dense.py           # Búsqueda vectorial (LanceDB + bge-m3)
│   │   ├── sparse.py          # Búsqueda léxica (BM25)
│   │   ├── graph_traversal.py # Recorrido multi-hop en Kùzu
│   │   ├── rrf.py             # Reciprocal Rank Fusion (k=60)
│   │   ├── reranker.py        # Cross-encoder reranking (BGE-Reranker)
│   │   ├── generator.py       # Generación con citas
│   │   └── pipeline.py        # Orquestador de recuperación
│   └── pipeline.py            # Orquestador end-to-end (FinancialGraphRAGPipeline)
├── evals/
│   ├── test_dataset.json      # 15 preguntas financieras con ground truth
│   └── run_ragas_eval.py      # Evaluación RAGAS (4 métricas)
├── notebooks/
│   └── 01_full_pipeline_demo.ipynb  # Demo paso a paso
├── docker-compose.yml         # Infraestructura local (opcional)
└── requirements.txt           # Dependencias
```

## Instalación

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd financial-graphrag

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
export GROQ_API_KEY="gsk_..."
```

## Uso Rápido

### Desde Python

```python
from langchain_groq import ChatGroq
from src.pipeline import FinancialGraphRAGPipeline

llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.0)
pipeline = FinancialGraphRAGPipeline(llm=llm)

# Ingestar un 10-K
pipeline.ingest_and_index(ticker="AAPL", year=2023)

# Preguntar
result = pipeline.query("What was Apple's revenue in 2023?")
print(result.answer)
print(result.citations)
```

### Desde Jupyter

```bash
jupyter notebook notebooks/01_full_pipeline_demo.ipynb
```

### Evaluación

```bash
python evals/run_ragas_eval.py
```

## Ontología del Grafo

| Nodo | Descripción |
|---|---|
| `Company` | Empresa identificada por ticker |
| `FinancialMetric` | Métrica financiera (ingresos, EPS, etc.) |
| `RiskFactor` | Factor de riesgo divulgado |
| `BusinessSegment` | Segmento operativo |
| `MacroEvent` | Evento macroeconómico |
| `DocumentChunk` | Fragmento de documento |

| Relación | Descripción |
|---|---|
| `OPERATES_IN` | Company → BusinessSegment |
| `REPORTED_METRIC` | Company → FinancialMetric |
| `IMPACTS_REVENUE` | RiskFactor/MacroEvent → FinancialMetric |
| `MITIGATES_RISK` | BusinessSegment → RiskFactor |
| `COMPETES_WITH` | Company → Company |
| `MENTIONS_*` | DocumentChunk → Entity |

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| Ingesta | sec-edgar-downloader, pdfplumber |
| Embeddings | bge-m3 (sentence-transformers) |
| Vector Store | LanceDB (embebido) |
| Búsqueda Léxica | BM25 (rank-bm25) |
| Graph DB | Kùzu (embebido) |
| LLM | Groq API / Ollama |
| Framework | LangChain / LangGraph |
| Reranker | BGE-Reranker-v2-m3 |
| Evaluación | RAGAS |

## Licencia

MIT
