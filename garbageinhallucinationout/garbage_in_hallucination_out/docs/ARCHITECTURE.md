# Architecture Deep Dive

## Core Thesis

This project demonstrates one central idea:

> **AI hallucination is primarily caused by data quality failure, not model failure.**

When you give a language model noisy, conflicting, or absent context, it fills the gaps with plausible-sounding but fabricated information. Fix the data, add retrieval grounding, and the same model becomes reliable.

---

## Phase 1: Engineering Hallucination

### What happens

```
User Query
    │
    ▼
HALLUCINATION_PROMPT_TEMPLATE
"Answer based on your general knowledge. Be confident..."
    │
    ▼
LLM (parametric memory only)
    │
    ▼
Confident, wrong answer ⚠️
```

### Why it fails

- **No context provided** → model relies on training weights
- **Permissive prompt** → model is encouraged to fill gaps confidently
- **Bad data not even used** → the raw CSV is so noisy that even naive retrieval would introduce conflicts

### Key code: `src/phase1_hallucination/hallucination_bot.py`

The prompt is intentionally loose:
```
Answer the question based on your general knowledge.
Be confident and provide as much detail as you can.
```
This is the worst-case production pattern seen in rushed AI deployments.

---

## Phase 2: Data Engineering + RAG

### What happens

```
Raw Bad Data
    │
    ▼
DataCleaner (data_processing/data_cleaner.py)
  • Schema standardization
  • Null handling
  • Deduplication
  • Golden Record resolution
    │
    ▼
Clean Documents
    │
    ▼
Embeddings (OpenAI / Ollama)
    │
    ▼
FAISS Vector Store (persisted to disk)

─────────────── At query time ───────────────

User Query
    │
    ▼
FAISS Retriever (top-k=3 most similar docs)
    │
    ▼
GROUNDED_PROMPT_TEMPLATE
"Answer ONLY from context. Say 'I don't know' if not found."
    │
    ▼
LLM + Retrieved Context
    │
    ▼
Grounded, verifiable answer ✅
```

### Why it succeeds

- **Clean data** → no conflicting facts enter the vector store
- **Retrieval grounding** → LLM only sees relevant, verified facts
- **Strict prompt guardrail** → explicitly prohibits invention
- **"I don't know" escape hatch** → honest uncertainty > confident hallucination

---

## Module Responsibilities

### `src/config_loader.py`
Singleton config from `config/config.yaml`. All paths, model names, and parameters live here. Nothing is hardcoded.

### `src/llm_factory.py`
Factory pattern for LLM and embeddings creation. Decouples the rest of the codebase from provider-specific imports. Switching from OpenAI to Ollama requires only a config change.

### `src/data_processing/data_cleaner.py`
The core data engineering module. Implements golden record logic using pandas groupby + custom aggregation. Each transformation is a separate private function (`_standardize_columns`, `_handle_nulls`, `_resolve_conflicts`) to make the pipeline testable and composable.

### `src/data_processing/document_loader.py`
Converts both text and CSV data sources into LangChain `Document` objects. Separation of concerns: loading is distinct from cleaning.

### `src/phase2_grounded/vector_store.py`
FAISS lifecycle management: build, save, load, or get-or-build. Handles the `allow_dangerous_deserialization=True` flag required by newer LangChain versions.

### `src/phase2_grounded/grounded_bot.py`
The production-pattern RAG bot. Uses the retriever to fetch context, then injects it into a strict prompt. Returns structured dicts including sources and retrieved doc count for auditability.

### `src/evaluation/evaluator.py`
Side-by-side comparison engine. Uses a lightweight ground truth dictionary and heuristic substring matching. For production, replace with LLM-as-judge or human evaluation.

### `src/ui/app.py`
Streamlit app with three tabs:
1. **Ask Questions** — live comparison in chosen mode
2. **Data Quality** — before/after metrics and transformation table
3. **Raw vs Clean Data** — direct DataFrame inspection

---

## Data Flow Diagram

```
data/raw/products_bad.csv          data/raw/knowledge_bad.txt
         │                                    │
         └──────────────┬───────────────────-─┘
                        │
                        ▼ (Phase 2 only)
              DataCleaner.clean_dataframe()
                        │
                        ▼
         data/clean/products_clean.csv
         data/clean/knowledge_clean.txt
                        │
                        ▼
              DocumentLoader.load_all_clean_documents()
                        │
                        ▼
              FAISS.from_documents(embeddings)
                        │
                        ▼
              data/faiss_index/   (persisted)
                        │
                        ▼ (at query time)
              vector_store.as_retriever(k=3)
                        │
                        ▼
              GroundedBot.ask(query)  →  Grounded Answer
```

---

## Synthetic Data Design

The synthetic data was designed to exercise every cleaning transformation:

| Issue in Raw Data | Cleaning Transformation |
|---|---|
| `P001` appears 3× with prices $100, $120, $115 | Median price → $110 |
| Product names: `Product A`, `product-a`, `Product A` | Title-case + deduplicate |
| `stock: "null"` (string null) | `_handle_nulls()` → NaN |
| `P007` row — all nulls | Dropped (no product_id) |
| `category: "accessories"` vs `"Accessories"` | Title-case standardization |
| Missing `description` on some rows | Longest non-null wins |
| Conflicting suppliers | Mode (majority) wins |
| Multiple `last_updated` dates | Most recent wins |
| `P008` appears twice with identical data | Deduplicated cleanly |

---

## Extension Points

- **Replace FAISS** → Pinecone, Weaviate, or pgvector for production scale
- **Replace heuristic eval** → LLM-as-judge, RAGAS, or human eval pipeline
- **Add Great Expectations** → `data_cleaner.py` outputs can be validated with GE suites
- **Add Medallion layers** → Bronze = `data/raw/`, Silver = `data/clean/`, Gold = vector store
- **Add data lineage** → extend metadata dict in Document objects with transformation history
