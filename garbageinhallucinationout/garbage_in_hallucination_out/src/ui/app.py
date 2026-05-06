"""
ui/app.py
---------
Streamlit demo UI for the Garbage In → Hallucination Out experiment.

Run with:
    streamlit run src/ui/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from src.phase1_hallucination.hallucination_bot import HallucinationBot
from src.phase2_grounded.grounded_bot import GroundedBot
from src.phase2_grounded.vector_store import get_or_build_vector_store
from src.data_processing.document_loader import load_all_clean_documents
from src.data_processing.data_cleaner import load_raw_data, clean_dataframe, get_quality_report


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Garbage In → Hallucination Out",
    page_icon="🤖",
    layout="wide",
)

# ── Session state init ──────────────────────────────────────────────────────
@st.cache_resource
def get_hallucination_bot():
    return HallucinationBot()


@st.cache_resource
def get_grounded_bot():
    docs = load_all_clean_documents()
    vs = get_or_build_vector_store(docs)
    return GroundedBot(vs)


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "Demo Mode",
    ["Side-by-Side Comparison", "Phase 1 Only (Bad Data)", "Phase 2 Only (Clean Data)"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This demo shows how **data quality** directly impacts **AI accuracy**.\n\n"
    "- **Phase 1**: No grounding → hallucinations\n"
    "- **Phase 2**: RAG + clean data → reliable answers\n\n"
    "_AI accuracy is a data engineering problem, not a model problem._"
)

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🗑️ Garbage In → Hallucination Out")
st.caption("A controlled RAG experiment demonstrating data quality's impact on AI reliability.")

tabs = st.tabs(["💬 Ask Questions", "📊 Data Quality", "📁 Raw vs Clean Data"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tabs[0]:
    query = st.text_input(
        "Ask a question about products:",
        placeholder="e.g. What is the price of Product A?",
    )

    example_queries = [
        "What is the price of Product A?",
        "Tell me about Product B.",
        "Is Product G available?",
        "Which products are in the Budget category?",
        "Who supplies Product D?",
    ]
    st.caption("Try: " + " | ".join(f"`{q}`" for q in example_queries))

    if query:
        if mode == "Side-by-Side Comparison":
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("❌ Phase 1 — No Grounding")
                with st.spinner("Asking ungrounded bot..."):
                    bot1 = get_hallucination_bot()
                    r1 = bot1.ask(query)
                st.error("⚠️ Hallucination Risk — No context provided")
                st.write(r1["answer"])

            with col2:
                st.subheader("✅ Phase 2 — RAG Grounded")
                with st.spinner("Retrieving and answering..."):
                    bot2 = get_grounded_bot()
                    r2 = bot2.ask(query)
                st.success(f"Grounded on {r2['retrieved_docs']} source documents")
                st.write(r2["answer"])
                if r2.get("sources"):
                    with st.expander("📎 Sources used"):
                        for s in r2["sources"]:
                            st.code(s)

        elif mode == "Phase 1 Only (Bad Data)":
            with st.spinner("Asking ungrounded bot..."):
                bot1 = get_hallucination_bot()
                r1 = bot1.ask(query)
            st.error("⚠️ Hallucination Risk")
            st.write(r1["answer"])

        else:  # Phase 2 only
            with st.spinner("Retrieving and answering..."):
                bot2 = get_grounded_bot()
                r2 = bot2.ask(query)
            st.success(f"Grounded on {r2['retrieved_docs']} documents")
            st.write(r2["answer"])
            with st.expander("🔍 Retrieved Context"):
                st.text(r2["context"])

# ── Tab 2: Data Quality Report ────────────────────────────────────────────────
with tabs[1]:
    st.subheader("📊 Data Quality Before vs After Cleaning")
    raw_df = load_raw_data()
    clean_df = clean_dataframe(raw_df.copy())
    report = get_quality_report(raw_df, clean_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Rows", report["raw_rows"])
    col2.metric("Clean Rows (Golden Records)", report["clean_rows"])
    col3.metric("Null Values Removed", report["raw_null_count"] - report["clean_null_count"])
    col4.metric("Null Reduction", f"{report['null_reduction_pct']}%")

    st.markdown("---")
    st.markdown("#### Data Transformations Applied")
    transformations = {
        "Step": [
            "Schema Standardization",
            "Null Handling",
            "Deduplication",
            "Conflict Resolution (Golden Record)",
            "Metadata Tagging",
        ],
        "Description": [
            "Column names lowercased; product names title-cased; categories standardized",
            "Empty strings, 'null' text, and NaN replaced consistently",
            "Duplicate rows for same product_id merged",
            "Price: median | Description: longest | Stock: max | Supplier: mode",
            "source='golden_record', verified=True added to all records",
        ],
    }
    st.table(pd.DataFrame(transformations))

# ── Tab 3: Raw vs Clean Data ───────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📁 Raw Data (Bad)")
    st.warning("Contains duplicates, conflicts, nulls, and inconsistent formatting.")
    st.dataframe(raw_df, use_container_width=True)

    st.subheader("✅ Clean Data (Golden Records)")
    st.success("Deduplicated, conflict-resolved, standardized, and governance-tagged.")
    st.dataframe(clean_df, use_container_width=True)
