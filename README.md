# Financial GraphRAG Engine

> Sistema local de **preguntas y respuestas financieras** sobre informes anuales **10-K de la SEC** (cuentas auditadas de empresas cotizadas de EE. UU.). 
> Combina **búsqueda vectorial densa**, **búsqueda léxica BM25** y un **Grafo de Conocimiento**, generando respuestas con citas exactas a los fragmentos originales.

100% local con `qwen3:8b` (optimizado para GPUs de 12GB) o con fallback a Groq. Incluye interfaz web tipo ChatGPT con streaming en tiempo real y empaquetado para Windows (.exe).

---

## Índice rápido

- [Ejemplos de lo que responde](#ejemplos-de-lo-que-responde)
- [Inicio Rápido (Quickstart)](#inicio-rápido-quickstart)
- [Formas de Uso](#formas-de-uso)
  - [1. En el Navegador Web (Recomendado)](#1-en-el-navegador-web-recomendado)
  - [2. App de Escritorio Windows (.exe)](#2-app-de-escritorio-windows-exe)
  - [3. Terminal interactiva (CLI)](#3-terminal-interactiva-cli)
  - [4. Como librería en Python](#4-como-librería-en-python)
- [Cómo Funciona la Arquitectura](#cómo-funciona-la-arquitectura)
- [Ingesta y Procesamiento (Chunking)](#ingesta-y-procesamiento-chunking)
- [Construcción y Consulta del Grafo](#construcción-y-consulta-del-grafo)
- [Estado de los Datos y Ontología](#estado-de-los-datos-y-ontología)
- [Estructura del Proyecto y Stack](#estructura-del-proyecto-y-stack)
- [Evaluación y Tests](#evaluación-y-tests)

---

## Ejemplos de lo que responde

> **¿En qué segmentos opera MSFT?**
> *Productivity and Business Processes, Intelligent Cloud y More Personal Computing* — con citas al fragmento `Item 7 / Reportable Segments` y hechos del grafo (`MSFT --OPERATES_IN--> Intelligent Cloud`).

> **What was AAPL revenue in 2024?**
> *$391,035 million* — con fila de tabla verificada: `| AAPL | 2024 | total net sales | 391035 USD millions | Item 8 | chunk_id |`.

> **¿A cuánto cotiza NVDA y qué noticias recientes hay?**
> *$119.10 (+1.84%)* con titulares financieros recientes y enlaces directos vía Finnhub API.

---

## Inicio Rápido (Quickstart)

### 1. Clonar e instalar dependencias

```bash
git clone <repo-url>
cd financial-graphrag

# Crear y activar entorno virtual
python -m venv .venv

# En Windows:
.\.venv\Scripts\activate
# En Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configurar el LLM (`.env`)

Crea un archivo `.env` en la raíz (puedes basarte en la plantilla):

```bash
# Opción A (Por defecto): Ollama local (100% privado y gratuito)
ollama pull qwen3:8b

# Opción B (Fallback opcional): Groq en la nube
GROQ_API_KEY="gsk_..."

# Datos de mercado y noticias en tiempo real (gratuito en https://finnhub.io):
FINNHUB_API_KEY="c1...x9"

# Opcional (acelera descarga de embeddings bge-m3):
HF_TOKEN="hf_..."
```

---

## Formas de Uso

Elige la forma de interactuar que más te convenga:

### 1. En el Navegador Web (Recomendado)

Una aplicación web completa estilo ChatGPT: historial de conversaciones en disco, tokens en tiempo real (streaming SSE), barras de progreso e interacción con confirmaciones humanas (HITL).

**Paso 1:** Arranca el servidor local:
```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```
*(Al arrancar tarda algo menos de 1 minuto en calentar los modelos y mostrará el aviso `API lista`)*.

**Paso 2:** Abre tu navegador favorito y entra en:
**`http://localhost:8000`**

---

### 2. App de Escritorio Windows (.exe)

Puedes usar la aplicación en una ventana nativa de Windows sin abrir el navegador:

* **Modo desarrollo:**
  ```powershell
  .\.venv\Scripts\python.exe desktop.py
  ```
* **Compilar un único archivo ejecutable (`.exe` independiente):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File build_desktop.ps1
  # Genera: dist\FinancialGraphRAG.exe (~1.5 GB empaquetado)
  ```
  **Al hacer doble clic en el `.exe`**:
  - Crea solo sus carpetas (`data/`, `logs/`) y restaura los datos base.
  - Si falta Ollama o el modelo, abre una ventana de bienvenida y descarga `qwen3:8b` automáticamente.
  - Incluye bloqueo de instancia única para evitar corromper la base de datos de grafos.

---

### 3. Terminal interactiva (CLI)

Ideal para desarrolladores que prefieren la consola:

```bash
# Agente ReAct (el modelo razona y elige entre 10 herramientas + streaming en vivo)
python cli.py --react

# Agente determinista estructurado
python cli.py

# Ingesta manual directa sin agente
python cli.py --ingest --ticker AAPL --year 2024
```

**Comandos útiles dentro del chat:**
- `<pregunta>`: Consulta RAG directa (ej: *¿Cuáles fueron los ingresos de Amazon en 2024?*).
- `¿y en 2023?`: Pregunta de seguimiento que reutiliza el contexto del ticker anterior.
- `/ingest <ticker> <año>`: Descarga e indexa un informe 10-K en caliente.
- `/clear`: Limpia la memoria de la conversación actual.
- `/exit`: Salir.

---

### 4. Como librería en Python

Puedes integrar el pipeline o los agentes directamente en tus propios scripts:

```python
from src.llm_factory import create_llm
from src.pipeline import FinancialGraphRAGPipeline

# 1. Pipeline directo
llm = create_llm()
pipeline = FinancialGraphRAGPipeline(llm=llm, graph_max_workers=2, graph_batch_size=1)

# Ingesta y consulta
pipeline.ingest_and_index(ticker="AAPL", year=2024)
res = pipeline.query("What was AAPL revenue in 2024?")

print("Respuesta:", res.answer)
print("Citas:", res.citations)
print("Tabla de métricas:", res.metrics_rows)
pipeline.close()

# 2. Agente ReAct con LangGraph
from src.agent.react_graph import build_react_agent
from langchain_core.messages import HumanMessage

agent = build_react_agent(pipeline)
respuesta = agent.invoke(
    {"messages": [HumanMessage(content="Compara el margen operativo de MSFT y AAPL")]},
    config={"configurable": {"thread_id": "mi_chat_1"}}
)
```

---

## Cómo Funciona la Arquitectura

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
        B) ReAct (--react y web API): el modelo decide herramientas
           10 tools: query_rag | lookup_metrics | calculator | propose_company
                     add_company (HITL 1) | ingest_10k (HITL 2) | stock_price
                     company_news | suggest_companies | lookup_company
                      ▼
        Recuperación Híbrida:
        Query Expansion → Prefilter por Ticker → Búsqueda paralela (Densa + BM25 + Grafo)
        → Fusión RRF (k=60) → Re-ranking con Cross-Encoder (bge-reranker-v2-m3)
                      ▼
        Generación final con LLM + Citas a fragmentos + TABLA DE MÉTRICAS
```

### Componentes principales:

1. **Recuperación Híbrida en 3 Vías**:
   - **Densa:** Vectores de 1024 dimensiones con `bge-m3` en **LanceDB** (almacenamiento en disco NVMe).
   - **Léxica:** Índice invertido **BM25** para palabras clave exactas.
   - **Grafo:** Recorridos por saltos en **Kùzu Graph DB** para hechos estructurales.
2. **Fusión y Re-ranking**:
   - Los resultados se combinan mediante **Reciprocal Rank Fusion (RRF)**.
   - Se reordenan los mejores candidatos con un **Cross-Encoder** (`bge-reranker-v2-m3`) para máxima precisión semántica.
3. **Agentes Inteligentes (LangGraph)**:
   - **10 Herramientas ReAct**: Consulta RAG, calculadora matemática financiera, registro oficial SEC, cotizaciones en vivo y noticias.
   - **Human-in-the-Loop (HITL) nativo**: Acciones sensibles (editar empresas seguidas o descargar 10-Ks) se pausan con `interrupt()` y esperan la confirmación del usuario con botones en la web o `y/n` en la consola.
   - **Auto-reparación de historial**: Si el usuario cancela un turno a mitad de proceso, el sistema inyecta mensajes sintéticos para evitar que LangGraph falle por llamadas huérfanas.

---

## Ingesta y Procesamiento (Chunking)

Los informes anuales 10-K contienen tablas financieras críticas. Un troceado ciego rompería las filas y mezclaría números con conceptos erróneos.

1. **Descarga oficial:** `sec-edgar-downloader` obtiene el documento oficial de la SEC (`full-submission.txt`).
2. **Conversión a Markdown:** Detecta tablas HTML con `beautifulsoup4` y las convierte a formato Markdown (`| col1 | col2 |`).
3. **División por secciones:** Detecta encabezados reglamentarios (`Item 1`, `Item 1A Riesgos`, `Item 7 MD&A`, `Item 8 Estados Financieros`).
4. **Chunking Table-Aware:**
   - Trocea el texto en bloques de unas ~600 palabras con solape de 90 palabras.
   - **Las tablas nunca se cortan por la mitad**: se procesan como bloques íntegros o se dividen fila por fila conservando las cabeceras.
5. **IDs deterministas:** Cada fragmento recibe un ID único por hash SHA-1 (`ticker/año/sección/página/seq`), evitando duplicados al reprocesar.

---

## Construcción y Consulta del Grafo

### Extracción de Tripletas
Cada fragmento pasa por `qwen3:8b` para extraer conocimiento estructurado según una ontología cerrada:
- **5 Entidades:** `Company`, `FinancialMetric`, `RiskFactor`, `BusinessSegment`, `MacroEvent`.
- **5 Relaciones:** `OPERATES_IN`, `REPORTED_METRIC`, `IMPACTS_REVENUE`, `MITIGATES_RISK`, `COMPETES_WITH`.

### Scoping de Métricas (Claves compuestas)
Para evitar que una cifra de ingresos de Apple colisione con una de Microsoft, cada métrica genera un ID único con ámbito:
$$\texttt{AAPL\_2024\_total\_net\_sales} \quad \text{vs} \quad \texttt{MSFT\_2024\_total\_net\_sales}$$
Las cifras fiables para la generación se consultan directamente mediante **tablas Markdown estructuradas**, garantizando que el LLM nunca confunda años o empresas.

---

## Estado de los Datos y Ontología

### Resumen del Corpus Indexado:
Empresas seguidas en `data/companies.json` (ejercicios 2024–2025):  
`AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, BRK.B, DELL, FLNC`.

| Componente | Volumen Actual | Detalle técnico |
|---|---|---|
| **Document Chunks** | **4.165 fragmentos** | 20 informes 10-K completos (10 empresas × 2 años) |
| **Tripletas extraídas** | **~33.689 tripletas** | Persistidas en caché incremental `triplets.json` |
| **Nodos en Grafo Kùzu** | **~17.200 nodos** | 15.491 métricas, 1.251 segmentos, 422 empresas |
| **Aristas en Grafo** | **~17.600 relaciones** | 15.586 métricas reportadas, 1.399 segmentos |
| **Vectores en LanceDB** | **4.165 vectores** | Embeddings `bge-m3` con índices escalares BTREE |

> **Nota:** Las carpetas con datos generados (`data/raw_10k/`, `data/processed_chunks/`, `data/graph/`, `data/vector_store/`) están excluidas del repositorio (`.gitignore`) para evitar subir cientos de megabytes. Al clonar el repositorio, se incluye la configuración inicial limpia en `data/companies.json`.

---

## Estructura del Proyecto y Stack

```
financial-graphrag/
├── api.py                      # Servidor FastAPI (SSE streaming, progreso, HITL, cancelación, chats)
├── desktop.py                  # Lanzador escritorio Windows (ventana nativa pywebview/WebView2)
├── build_desktop.ps1           # Script de empaquetado para generar FinancialGraphRAG.exe
├── cli.py                      # Interfaz interactiva de consola (CLI con streaming)
├── reprocess_missing.py        # Detección y reanudación de huecos de ingesta
├── static/
│   └── index.html              # Frontend web SPA (estilo ChatGPT, SSE, barras de progreso)
├── data/
│   └── companies.json          # Lista de empresas seguidas (AAPL, MSFT, NVDA...)
├── src/
│   ├── bootstrap.py            # Inicialización autónoma para el .exe (crea carpetas y descarga modelo)
│   ├── env.py                  # Gestión de variables de entorno (.env)
│   ├── llm_factory.py          # Factoría de modelos (Ollama local / Groq fallback)
│   ├── pipeline.py             # Pipeline unificado de FinancialGraphRAG
│   ├── agent/                  # Agentes LangGraph (ReAct 10 tools, determinista, progreso, historial)
│   ├── ingestion/              # Descarga de la SEC, parser HTML/PDF y chunker table-aware
│   ├── graph/                  # Extracción de tripletas, ontología y persistencia en Kùzu DB
│   └── retrieval/              # Búsqueda híbrida (densa, dispersa, grafo), RRF y re-ranking
└── evals/                      # Suite de 51 pruebas deterministas con gate de calidad 0.85
```

### Tecnologías Principales (Stack):

| Capa | Herramienta |
|---|---|
| **Lenguaje** | Python 3.10+ |
| **LLM Local** | Ollama (`qwen3:8b`, reasoning=False, context=8192) con fallback a Groq |
| **Orquestación de Agentes** | LangChain & LangGraph (ReAct con ToolNode y `interrupt()`) |
| **Embeddings & Reranker** | BAAI/bge-m3 & BAAI/bge-reranker-v2-m3 |
| **Base de Datos Vectorial** | LanceDB (formato columnar Apache Arrow sobre disco) |
| **Base de Datos de Grafos** | Kùzu DB (grafo de propiedades embebido ultrarrápido) |
| **Búsqueda Léxica** | BM25 (`rank-bm25`) |
| **Web & API** | FastAPI, Uvicorn, Server-Sent Events (SSE) |
| **Escritorio Windows** | `pywebview` (Microsoft Edge WebView2) y PyInstaller |
| **Datos Financieros en Vivo** | Finnhub REST API y SEC EDGAR |

---

## Evaluación y Tests

El proyecto incluye una suite de evaluación determinista de **51 preguntas reales** de balances (márgenes, ventas YoY, riesgos y segmentos):

```bash
# Ejecutar verificación rápida (5 muestras):
python evals/run_checks.py --samples 5

# Ejecutar la suite completa de 51 pruebas (gate de calidad 0.85):
python evals/run_checks.py
```

---

## Licencia

MIT License. Creado con fines de investigación, análisis financiero y desarrollo de agentes avanzados de IA.
