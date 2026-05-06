# Garbage In → Hallucination Out
## Complete Project Documentation

> **Core thesis:** AI hallucination is a data engineering problem, not a model problem.  
> This project proves it — with code, data, and a live side-by-side demo.

---

## Table of Contents

1. [Project Plan](#1-project-plan)
2. [Tech Stack](#2-tech-stack)
3. [Execution Plan — Step-by-Step](#3-execution-plan--step-by-step)
4. [Required Tools and Licenses](#4-required-tools-and-licenses)
5. [Required Software and LLMs](#5-required-software-and-llms)
6. [Installation and Setup Guide](#6-installation-and-setup-guide)

---

## 1. Project Plan

### 1.1 Objective

Build a controlled RAG (Retrieval-Augmented Generation) experiment that demonstrates — through a live, interactive demo — how data quality directly determines AI output quality.

The experiment runs in two clearly separated phases:

| Phase | Data State | AI Behaviour | Expected Output |
|---|---|---|---|
| Phase 1 | Raw, noisy, conflicting | No retrieval grounding | Confident hallucinations |
| Phase 2 | Cleaned, governed, standardized | RAG + strict prompt guardrails | Accurate, verifiable answers |

### 1.2 Problem Statement

Enterprise AI deployments routinely fail not because the model is bad, but because the data fed into it is bad. When an LLM has no reliable context to retrieve from, it fills gaps from its parametric memory — producing plausible-sounding but factually wrong answers. This project makes that failure mode visible and measurable, then shows the exact data engineering steps that fix it.

### 1.3 Project Scope

**In scope:**
- Synthetic dataset generation (raw bad data + clean golden records)
- Data cleaning pipeline (deduplication, conflict resolution, standardization)
- Phase 1: Hallucination bot (no retrieval)
- Phase 2: Grounded bot (FAISS vector store + RAG)
- Side-by-side evaluation engine with accuracy scoring
- Streamlit interactive demo UI
- Full unit test coverage for the data pipeline
- Documentation (this file)

**Out of scope:**
- Production deployment / cloud hosting
- Real-world proprietary datasets
- Fine-tuning or RLHF
- Multi-modal data (images, audio)

### 1.4 Deliverables

```
garbage_in_hallucination_out/
├── src/                    ← Modularized Python source code
├── data/raw/               ← Synthetic bad data (CSV + text)
├── data/clean/             ← Synthetic clean data (CSV + text)
├── scripts/                ← Runnable CLI scripts
├── tests/                  ← Pytest unit tests (17 tests, all passing)
├── docs/                   ← Documentation
└── config/config.yaml      ← All configuration in one place
```

### 1.5 Success Criteria

- Phase 1 bot produces visibly wrong or fabricated answers on known test queries
- Phase 2 bot produces correct, source-cited answers on the same queries
- Data quality report shows measurable improvement: null reduction %, duplicate removal, conflict resolution
- All unit tests pass
- Demo can be launched with a single `streamlit run` command

### 1.6 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — Bad State                                        │
│                                                             │
│  [Noisy Raw Data]  ──►  [LLM (no context)]  ──►  ⚠️ Hallucination  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — Good State                                       │
│                                                             │
│  [Raw Data]                                                 │
│      │                                                      │
│      ▼                                                      │
│  [DataCleaner]  ──►  [Golden Records]                       │
│                            │                                │
│                       [Embeddings]                          │
│                            │                                │
│                      [FAISS Vector DB]                      │
│                            │                                │
│  [User Query]  ──►  [Retriever (top-k)]                    │
│                            │                                │
│                   [LLM + Context + Guardrail]               │
│                            │                                │
│                      ✅ Grounded Answer                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

### 2.1 Summary Table

| Layer | Component | Technology | Purpose |
|---|---|---|---|
| Language | Python | Python 3.10+ | Core language |
| LLM (hosted) | OpenAI API | GPT-3.5-turbo / GPT-4 | Language model |
| LLM (local) | Ollama | llama3 / mistral | Local alternative to OpenAI |
| Embeddings (hosted) | OpenAI API | text-embedding-3-small | Vector embeddings |
| Embeddings (local) | Ollama | nomic-embed-text | Local embeddings |
| RAG Framework | LangChain | langchain, langchain-community | Retrieval chains, document handling |
| Vector Store | FAISS | faiss-cpu | Local vector similarity search |
| Data Processing | pandas + numpy | pandas 2.x | Cleaning, deduplication, golden records |
| Configuration | PyYAML | pyyaml | Centralized YAML config |
| UI / Demo | Streamlit | streamlit 1.35+ | Interactive web demo |
| Testing | pytest | pytest 9.x | Unit test framework |
| Environment | python-dotenv | python-dotenv | .env API key management |

### 2.2 LangChain Modules Used

```
langchain_core.documents       ← Document dataclass
langchain_community.vectorstores.FAISS  ← Vector store
langchain_community.llms.Ollama        ← Ollama LLM wrapper
langchain_openai.ChatOpenAI            ← OpenAI LLM wrapper
langchain_openai.OpenAIEmbeddings      ← OpenAI embeddings
langchain_community.embeddings.OllamaEmbeddings  ← Ollama embeddings
```

### 2.3 Provider Options (Configurable)

The entire project supports two LLM/Embedding providers, switchable in `config/config.yaml`:

**Option A — OpenAI (recommended for demos, requires API key + billing)**

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
```

**Option B — Ollama (fully local, free, no API key)**

```yaml
llm:
  provider: "ollama"
  model: "llama3"
embeddings:
  provider: "ollama"
  ollama_model: "nomic-embed-text"
```

### 2.4 Data Flow by Technology

```
CSV (pandas)  ──►  DataCleaner (pandas)  ──►  Clean CSV
Text file (pathlib)  ──────────────────►  Clean Text

Clean CSV + Clean Text
      │
      ▼
DocumentLoader (LangChain Document objects)
      │
      ▼
Embeddings (OpenAI / Ollama)
      │
      ▼
FAISS.from_documents()  ──►  Persisted Index (disk)
      │
      ▼ (at query time)
FAISS.as_retriever(k=3)
      │
      ▼
LLM.invoke(prompt_with_context)
      │
      ▼
Structured response dict
```

---

## 3. Execution Plan — Step-by-Step

### 3.1 High-Level Phases

```
Phase 0: Setup & Installation         (15–30 min)
Phase 1: Data Preparation             (5 min)
Phase 2: Vector Index Build           (2–5 min)
Phase 3: Run Evaluation               (2–5 min)
Phase 4: Launch Demo UI               (< 1 min)
```

### 3.2 Detailed Execution Steps

#### Step 0 — Environment Setup

```bash
# 0.1 Clone the repository
git clone <repo-url>
cd garbage_in_hallucination_out

# 0.2 Create and activate Python virtual environment
python -m venv .venv

# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# 0.3 Install all dependencies
pip install -r requirements.txt

# 0.4 Configure API keys
cp .env.example .env
# Edit .env → add OPENAI_API_KEY=sk-...
```

#### Step 1 — Inspect the Raw (Bad) Data

```bash
python scripts/clean_data.py
```

**What this does:**
- Loads `data/raw/products_bad.csv` (15 rows with duplicates, conflicts, nulls)
- Runs the full cleaning pipeline
- Prints a before/after quality report
- Saves clean golden records to `data/clean/products_clean.csv`

**Expected output:**
```
raw_rows                            15
clean_rows                          7
duplicates_removed                  3
raw_null_count                      22
clean_null_count                    0
null_reduction_pct                  100%
unique_products_before              8
unique_products_after               7
```

#### Step 2 — Build the Vector Index

```bash
python scripts/build_index.py
```

**What this does:**
- Loads `data/clean/products_clean.csv` and `data/clean/knowledge_clean.txt`
- Converts them into LangChain Documents
- Generates embeddings (via OpenAI or Ollama)
- Builds a FAISS index and saves it to `data/faiss_index/`

**Expected output:**
```
[1/2] Loading clean documents...
      Loaded 14 documents.
[2/2] Embedding and building FAISS index...
      Index saved successfully.
✅ Vector store build complete.
```

#### Step 3 — Run Side-by-Side Evaluation

```bash
python scripts/run_evaluation.py
```

**What this does:**
- Initializes both bots (Phase 1 and Phase 2)
- Runs 7 test queries against both bots
- Prints a comparison report showing Phase 1 (wrong) vs Phase 2 (correct) answers
- Saves `logs/eval_report.json`

**Sample output:**
```
[Query 1] What is the price of Product A?
  Ground Truth : $110
  Phase 1 (No Grounding) : Product A typically costs around $99.99 or $129.99...
    → Correct? ❌
  Phase 2 (RAG Grounded) : Product A is priced at $110.
    → Correct? ✅

SUMMARY
  Phase 1 (Hallucination) Accuracy : 1/7
  Phase 2 (Grounded RAG)  Accuracy : 7/7
```

#### Step 4 — Launch the Interactive UI

```bash
streamlit run src/ui/app.py
```

**What this does:**
- Opens browser at `http://localhost:8501`
- Tab 1 "Ask Questions": type any product question, see both bots answer
- Tab 2 "Data Quality": view before/after quality metrics and transformation table
- Tab 3 "Raw vs Clean Data": browse both DataFrames side by side

#### Step 5 — Run Unit Tests

```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_data_cleaner.py::TestHandleNulls::test_null_strings_replaced PASSED
tests/test_data_cleaner.py::TestStandardizeText::test_product_name_title_case PASSED
...
17 passed in 0.47s
```

### 3.3 Execution Dependency Graph

```
install dependencies
       │
       ▼
copy .env + add API key
       │
       ▼
scripts/clean_data.py       ← generates data/clean/
       │
       ▼
scripts/build_index.py      ← generates data/faiss_index/
       │
       ├──► scripts/run_evaluation.py   (CLI report)
       │
       └──► streamlit run src/ui/app.py (interactive demo)
```

### 3.4 Module Execution Map

| Script / Entry Point | Modules Called | Output |
|---|---|---|
| `scripts/clean_data.py` | `data_cleaner.py` | `data/clean/products_clean.csv` |
| `scripts/build_index.py` | `document_loader.py`, `vector_store.py`, `llm_factory.py` | `data/faiss_index/` |
| `scripts/run_evaluation.py` | `hallucination_bot.py`, `grounded_bot.py`, `evaluator.py` | `logs/eval_report.json` |
| `streamlit run src/ui/app.py` | All modules | Browser UI |
| `pytest tests/` | `data_cleaner.py`, `document_loader.py` | Test report |

---

## 4. Required Tools and Licenses

### 4.1 Core Python Libraries

| Library | Version | License | Purpose |
|---|---|---|---|
| `langchain` | ≥ 0.2.0 | MIT | RAG chains, document handling |
| `langchain-community` | ≥ 0.2.0 | MIT | FAISS, Ollama wrappers |
| `langchain-openai` | ≥ 0.1.0 | MIT | OpenAI LLM + Embeddings |
| `faiss-cpu` | ≥ 1.7.4 | MIT | Local vector similarity search |
| `openai` | ≥ 1.0.0 | MIT | OpenAI Python SDK |
| `pandas` | ≥ 2.0.0 | BSD-3-Clause | Data processing |
| `numpy` | ≥ 1.24.0 | BSD-3-Clause | Numeric operations |
| `pyyaml` | ≥ 6.0 | MIT | YAML config parsing |
| `streamlit` | ≥ 1.35.0 | Apache 2.0 | Demo web UI |
| `python-dotenv` | ≥ 1.0.0 | BSD-3-Clause | .env file loading |
| `colorama` | ≥ 0.4.6 | BSD-3-Clause | Colored terminal output |
| `tabulate` | ≥ 0.9.0 | MIT | Table formatting in CLI |
| `pytest` | ≥ 9.0.0 | MIT | Unit testing |

**All libraries are open-source with permissive licenses (MIT / BSD / Apache 2.0). No commercial licenses required for the software itself.**

### 4.2 External Service Licenses

| Service | License / Plan | Required For | Cost |
|---|---|---|---|
| OpenAI API | Pay-per-use (commercial) | LLM inference + embeddings (Option A) | ~$0.01–0.10 per demo run |
| Ollama | MIT (free, self-hosted) | Local LLM inference (Option B) | Free |

> **Note:** If using OpenAI, you must agree to [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use). API usage is billed per token. For a full demo run (7 queries + embedding build), expect under $0.05 with GPT-3.5-turbo.

### 4.3 Development Tools (Optional)

| Tool | License | Purpose |
|---|---|---|
| VS Code | MIT | IDE |
| Git | GPL-2.0 | Version control |
| GitHub / GitLab | Proprietary (free tier available) | Repository hosting |
| Docker | Apache 2.0 | Containerization (optional) |

---

## 5. Required Software and LLMs

### 5.1 Runtime Requirements

| Software | Minimum Version | Recommended | Notes |
|---|---|---|---|
| Python | 3.10 | 3.11 or 3.12 | Type hint syntax requires 3.10+ |
| pip | 23.0+ | latest | Use `pip install --upgrade pip` |
| Git | Any | latest | For cloning the repo |
| macOS / Linux / Windows | Any modern version | Ubuntu 22.04 / macOS 13+ | FAISS has native support on all three |

### 5.2 LLM Options

#### Option A — OpenAI (Hosted)

| Model | Use | Cost | Notes |
|---|---|---|---|
| `gpt-3.5-turbo` | LLM inference | ~$0.0015 / 1K tokens | Fastest, cheapest — recommended for demos |
| `gpt-4o` | LLM inference | ~$0.005 / 1K tokens | Stronger reasoning, better for showcase |
| `gpt-4o-mini` | LLM inference | ~$0.00015 / 1K tokens | Budget option |
| `text-embedding-3-small` | Embeddings | ~$0.00002 / 1K tokens | Default embedding model |
| `text-embedding-3-large` | Embeddings | ~$0.00013 / 1K tokens | Higher quality vectors |

**Setup:** Requires an OpenAI account, API key, and payment method. Create at [platform.openai.com](https://platform.openai.com).

#### Option B — Ollama (Fully Local, Free)

| Model | RAM Required | Quality | Notes |
|---|---|---|---|
| `llama3` (8B) | 8 GB | Good | Best balance for demos |
| `mistral` (7B) | 8 GB | Good | Fast inference |
| `llama3.1` (8B) | 8 GB | Better | Improved instruction following |
| `phi3` (3.8B) | 4 GB | Fair | Good for low-RAM machines |
| `nomic-embed-text` | 500 MB | Good | Embedding model (required for Ollama path) |

**Setup:** Download Ollama from [ollama.com](https://ollama.com), then:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 5.3 Hardware Requirements

| Scenario | CPU | RAM | Storage | GPU |
|---|---|---|---|---|
| OpenAI (API calls only) | Any modern CPU | 4 GB | 1 GB | Not required |
| Ollama — 7B/8B model | 4-core+ CPU | 16 GB recommended | 8–10 GB | Optional (speeds up inference) |
| Ollama — with GPU | Any | 8 GB system + VRAM | 8–10 GB | 6+ GB VRAM |

> **Tip:** For presentation purposes, OpenAI is strongly recommended — it is faster, requires no local GPU, and the cost for a full demo is under $0.10.

### 5.4 Vector Store — FAISS

FAISS (`faiss-cpu`) runs entirely locally with no external service. It persists the index to `data/faiss_index/` on disk. It requires no GPU for the scale of this project (< 100 documents).

---

## 6. Installation and Setup Guide

### 6.1 Prerequisites Checklist

Before starting, verify these are installed:

```bash
python --version          # Must be 3.10 or higher
pip --version             # Should be 23+
git --version             # Any version
```

If Python is not installed: download from [python.org](https://python.org/downloads) or use `pyenv`.

### 6.2 Project Setup

#### Step 1 — Get the Code

```bash
git clone <repository-url>
cd garbage_in_hallucination_out
```

Or if you received a `.zip` file:
```bash
unzip garbage_in_hallucination_out.zip
cd garbage_in_hallucination_out
```

#### Step 2 — Create Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal prompt after activation.

#### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages. Expect 2–5 minutes depending on your connection.

**Verify the install:**
```bash
python -c "import langchain, faiss, pandas, streamlit; print('All dependencies OK')"
```

#### Step 4 — Configure API Keys (OpenAI Path)

```bash
cp .env.example .env
```

Open `.env` in any text editor and add your key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

**Where to get an OpenAI API key:**
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys → Create new secret key
4. Copy the key (you will only see it once)
5. Paste into `.env`

**Verify:**
```bash
python -c "from openai import OpenAI; c = OpenAI(); print('OpenAI connection OK')"
```

#### Step 4 (Alternative) — Configure Ollama (Local Path)

**Install Ollama:**

```bash
# macOS:
brew install ollama

# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Windows:
# Download installer from https://ollama.com/download
```

**Pull required models:**
```bash
ollama pull llama3            # ~4.7 GB download
ollama pull nomic-embed-text  # ~274 MB download
```

**Start the Ollama server:**
```bash
ollama serve
# Leave this running in a background terminal
```

**Update `config/config.yaml`:**
```yaml
llm:
  provider: "ollama"
  model: "llama3"
  ollama_base_url: "http://localhost:11434"

embeddings:
  provider: "ollama"
  ollama_model: "nomic-embed-text"
```

**Verify:**
```bash
curl http://localhost:11434/api/generate -d '{"model":"llama3","prompt":"hello","stream":false}'
```

#### Step 5 — Run Data Cleaning

```bash
python scripts/clean_data.py
```

Expected: quality report printed, `data/clean/products_clean.csv` updated.

#### Step 6 — Build Vector Index

```bash
python scripts/build_index.py
```

Expected: `data/faiss_index/` directory created with `index.faiss` and `index.pkl`.

#### Step 7 — Run Tests

```bash
pytest tests/ -v
```

Expected: `17 passed`.

#### Step 8 — Launch the Demo

```bash
streamlit run src/ui/app.py
```

Browser opens at `http://localhost:8501`.

---

### 6.3 Configuration Reference (`config/config.yaml`)

```yaml
llm:
  provider: "openai"           # "openai" | "ollama"
  model: "gpt-3.5-turbo"       # OpenAI model name, or Ollama model name
  ollama_base_url: "http://localhost:11434"
  temperature: 0.0             # 0.0 = deterministic (recommended for demos)
  max_tokens: 512

embeddings:
  provider: "openai"           # "openai" | "ollama"
  model: "text-embedding-3-small"
  ollama_model: "nomic-embed-text"

vector_store:
  type: "faiss"
  index_path: "data/faiss_index"

data:
  raw_csv: "data/raw/products_bad.csv"
  clean_csv: "data/clean/products_clean.csv"
  raw_knowledge: "data/raw/knowledge_bad.txt"
  clean_knowledge: "data/clean/knowledge_clean.txt"

evaluation:
  test_queries:
    - "What is the price of Product A?"
    - "Tell me about Product B."
    - "Is Product G available?"
    - "What category is Product C in?"
    - "Which products are in the Budget category?"
    - "Who supplies Product D?"
    - "What is the stock level of Product E?"
```

---

### 6.4 Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain` | Dependencies not installed | Run `pip install -r requirements.txt` inside the virtualenv |
| `AuthenticationError: OpenAI` | Missing or wrong API key | Check `.env` file has correct `OPENAI_API_KEY` |
| `FileNotFoundError: data/faiss_index` | Index not built yet | Run `python scripts/build_index.py` |
| `Connection refused: localhost:11434` | Ollama server not running | Run `ollama serve` in a separate terminal |
| `KeyError: product_id` | Running old Python / pandas | Upgrade: `pip install --upgrade pandas` |
| `streamlit: command not found` | Not in virtualenv | Run `source .venv/bin/activate` first |
| Slow response on Ollama | Large model / no GPU | Switch to `phi3` (smaller model) in config, or use OpenAI |
| FAISS import error on Windows | Binary compatibility | Use `pip install faiss-cpu --force-reinstall` |

---

### 6.5 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (OpenAI path) | Your OpenAI secret API key |
| `LLM_PROVIDER` | No | Override config: `"openai"` or `"ollama"` |
| `LLM_MODEL` | No | Override config: e.g. `"gpt-4o"` |

---

### 6.6 Quick Reference Card

```
# Full setup from scratch (OpenAI path)
git clone <repo> && cd garbage_in_hallucination_out
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then add your OPENAI_API_KEY
python scripts/clean_data.py
python scripts/build_index.py
pytest tests/ -v                # should show 17 passed
streamlit run src/ui/app.py     # opens demo in browser

# Full setup from scratch (Ollama / local path)
ollama pull llama3 && ollama pull nomic-embed-text
# update config/config.yaml: provider = "ollama"
python scripts/clean_data.py
python scripts/build_index.py
streamlit run src/ui/app.py
```

---

*Last updated: May 2026*  
*Project: Garbage In → Hallucination Out | Version 1.0*
