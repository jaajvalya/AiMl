# 🗑️ Garbage In → Hallucination Out

> **"AI accuracy is a data engineering problem, not a model problem."**

A controlled RAG experiment that demonstrates, side-by-side, how data quality directly determines AI reliability. Feed an LLM noisy, conflicting data with no retrieval grounding → hallucinations. Apply data engineering discipline and RAG → trustworthy answers.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1 (Bad State)                      │
│                                                                 │
│  [Raw Bad Data]  ──────────────►  [LLM directly]               │
│  • Duplicates                                                   │
│  • Conflicting facts               ▼                            │
│  • Missing fields           Hallucination ⚠️                    │
│  • No governance                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 2 (Good State)                     │
│                                                                 │
│  [Raw Data]  ──► [Data Cleaner]  ──► [Clean / Golden Records]  │
│                                              │                  │
│                                        [Embeddings]             │
│                                              │                  │
│                                        [Vector DB (FAISS)]      │
│                                              │                  │
│  [User Query]  ──────────────────►  [Retriever]                │
│                                              │                  │
│                                         [LLM + Context]         │
│                                              │                  │
│                                    Grounded Answer ✅           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
garbage_in_hallucination_out/
│
├── config/
│   └── config.yaml               # All configuration (LLM, embeddings, paths)
│
├── data/
│   ├── raw/
│   │   ├── products_bad.csv      # Synthetic: conflicting, duplicate, null-heavy data
│   │   └── knowledge_bad.txt     # Synthetic: vague, inconsistent knowledge docs
│   └── clean/
│       ├── products_clean.csv    # Synthetic: golden records, resolved data
│       └── knowledge_clean.txt   # Synthetic: precise, verified knowledge docs
│
├── src/
│   ├── config_loader.py          # YAML config singleton
│   ├── llm_factory.py            # LLM + Embeddings factory (OpenAI / Ollama)
│   │
│   ├── data_processing/
│   │   ├── data_cleaner.py       # Dedup, standardize, golden-record resolution
│   │   └── document_loader.py    # Loads text + CSV → LangChain Documents
│   │
│   ├── phase1_hallucination/
│   │   └── hallucination_bot.py  # No-retrieval bot → hallucination demo
│   │
│   ├── phase2_grounded/
│   │   ├── vector_store.py       # FAISS build / load / get-or-build
│   │   └── grounded_bot.py       # RAG bot with strict context-only prompting
│   │
│   ├── evaluation/
│   │   └── evaluator.py          # Side-by-side comparison + accuracy report
│   │
│   └── ui/
│       └── app.py                # Streamlit demo UI
│
├── scripts/
│   ├── clean_data.py             # Run cleaning pipeline + print quality report
│   ├── build_index.py            # Build FAISS vector store from clean data
│   └── run_evaluation.py         # Full side-by-side evaluation run
│
├── tests/
│   ├── conftest.py
│   ├── test_data_cleaner.py      # Unit tests for cleaning pipeline
│   └── test_document_loader.py   # Unit tests for document loading
│
├── logs/                         # Auto-created; holds eval_report.json
├── docs/
│   ├── README.md                 # This file
│   └── ARCHITECTURE.md           # Detailed architecture decisions
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone <repo>
cd garbage_in_hallucination_out
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...
```

Or switch to Ollama in `config/config.yaml`:
```yaml
llm:
  provider: "ollama"
  model: "llama3"
```

### 3. Clean Data

```bash
python scripts/clean_data.py
```

### 4. Build Vector Index

```bash
python scripts/build_index.py
```

### 5. Run Evaluation

```bash
python scripts/run_evaluation.py
```

### 6. Launch UI

```bash
streamlit run src/ui/app.py
```

---

## 🔬 What the Experiment Demonstrates

| Aspect | Phase 1 (Bad) | Phase 2 (Good) |
|---|---|---|
| Data | Duplicates, conflicts, nulls | Golden records, standardized |
| Retrieval | None | FAISS vector search |
| Prompt | "Answer from your knowledge" | "Answer ONLY from context" |
| Outcome | Hallucination | Grounded, verifiable |
| Governance | None | Source + verified metadata |

### Data Transformations Applied

1. **Schema Standardization** — column names, product name casing, category normalization
2. **Null Handling** — empty strings, `"null"` text, `NaN` unified
3. **Deduplication** — rows grouped by `product_id`
4. **Conflict Resolution (Golden Record Logic)**
   - `price` → median (robust to outliers)
   - `description` → longest non-null string
   - `stock` → max (conservative)
   - `supplier`, `category` → mode (majority vote)
   - `last_updated` → most recent date
5. **Metadata Tagging** — `source="golden_record"`, `verified=True`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Configuration Reference

See `config/config.yaml`:

| Key | Description |
|---|---|
| `llm.provider` | `"openai"` or `"ollama"` |
| `llm.model` | Model name (e.g. `"gpt-3.5-turbo"` or `"llama3"`) |
| `embeddings.provider` | `"openai"` or `"ollama"` |
| `vector_store.index_path` | Where to save/load the FAISS index |
| `evaluation.test_queries` | List of questions used in evaluation |

---

## 🏢 Enterprise Angle

This project maps directly to enterprise data & AI governance patterns:

- **Data Contracts** → Schema enforcement before AI ingestion
- **Medallion Architecture** → Bronze (raw) → Silver (cleaned) → Gold (indexed/served)
- **Data Quality Frameworks** → Great Expectations / Deequ patterns (implemented in `data_cleaner.py`)
- **AI Governance** → Source tracking, verified flags, grounding guardrails
- **Semantic Layer** → The vector store acts as a governed semantic retrieval layer

---

## 📄 License

MIT
