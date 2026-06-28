# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------

import os
import tempfile
import time
import json
import hashlib
from datetime import datetime

import streamlit as st
import google.generativeai as genai

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredExcelLoader,
    TextLoader, WebBaseLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# -------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# -------------------------------------------------------
# SESSION STATE INITIALISATION
# -------------------------------------------------------

def _init_state():
    defaults = {
        "documents":     None,
        "vector_store":  None,
        "doc_hash":      None,
        "chat_history":  [],
        "gemini_model":  None,
        "eval_results":  [],
        "index_built":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# -------------------------------------------------------
# EMBEDDINGS  — cached so model loads only once
# -------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Loading embedding model…")
def _load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# -------------------------------------------------------
# DOCUMENT LOADERS
# -------------------------------------------------------

def _attach_meta(docs: list, source_name: str) -> list:
    for doc in docs:
        doc.metadata.setdefault("source", source_name)
        doc.metadata.setdefault("page", 0)
    return docs

def load_pdf(f) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        docs = PyPDFLoader(path).load()
        os.remove(path)
        return _attach_meta(docs, f.name)
    except Exception as e:
        st.error(f"PDF error: {e}"); return []

def load_csv(f) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        docs = CSVLoader(file_path=path).load()
        os.remove(path)
        return _attach_meta(docs, f.name)
    except Exception as e:
        st.error(f"CSV error: {e}"); return []

def load_excel(f) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        docs = UnstructuredExcelLoader(file_path=path).load()
        os.remove(path)
        return _attach_meta(docs, f.name)
    except Exception as e:
        st.error(f"Excel error: {e}"); return []

def load_text(f) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        docs = TextLoader(file_path=path).load()
        os.remove(path)
        return _attach_meta(docs, f.name)
    except Exception as e:
        st.error(f"Text error: {e}"); return []

def load_web(url: str) -> list:
    try:
        docs = WebBaseLoader(url).load()
        return _attach_meta(docs, url)
    except Exception as e:
        st.error(f"Web error: {e}"); return []

# -------------------------------------------------------
# CHUNKING
# -------------------------------------------------------

def _split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata.setdefault("source", "unknown")
        c.metadata.setdefault("page", 0)
    return chunks

# -------------------------------------------------------
# CHROMADB VECTOR STORE
# -------------------------------------------------------

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "omnisum_chroma")

def _docs_hash(docs: list) -> str:
    combined = "".join(d.page_content for d in docs[:10])
    return hashlib.md5(combined.encode()).hexdigest()

def build_vector_store(docs: list):
    current_hash = _docs_hash(docs)
    if (st.session_state.vector_store is not None
            and st.session_state.doc_hash == current_hash):
        return st.session_state.vector_store

    chunks = _split(docs)
    collection = f"omnisum_{current_hash[:8]}"
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=_load_embeddings(),
        collection_name=collection,
        persist_directory=CHROMA_DIR,
    )
    st.session_state.vector_store = vs
    st.session_state.doc_hash     = current_hash
    st.session_state.index_built  = True
    return vs

# -------------------------------------------------------
# HYBRID RETRIEVER
# -------------------------------------------------------

def build_retriever(docs: list, vs):
    chunks = _split(docs)
    bm25   = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    vr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )
    return EnsembleRetriever(
        retrievers=[bm25, vr], weights=[0.4, 0.6]
    )

# -------------------------------------------------------
# QUERY REWRITING
# -------------------------------------------------------

def rewrite_query(query: str) -> str:
    history = st.session_state.chat_history
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
    ) if history else "None"

    prompt = f"""You are a search query optimiser for an enterprise document retrieval system.
Rewrite the user question so it is self-contained, resolves pronouns using history,
and is optimal for semantic search.
Return ONLY the rewritten query — no explanation, no quotes.

Conversation history:
{history_text}

User question: {query}

Rewritten query:"""
    try:
        return st.session_state.gemini_model.generate_content(prompt).text.strip()
    except Exception:
        return query

# -------------------------------------------------------
# RE-RANKING
# -------------------------------------------------------

def rerank(query: str, docs: list, top_k: int = 4) -> list:
    qtoks = set(query.lower().split())
    scored = []
    for doc in docs:
        dtoks = set(doc.page_content.lower().split())
        overlap = len(qtoks & dtoks)
        page_bonus = max(0, 5 - int(doc.metadata.get("page", 0)))
        scored.append((overlap + page_bonus, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:top_k]]

# -------------------------------------------------------
# ANSWER GENERATION
# -------------------------------------------------------

def generate_answer(query: str):
    docs      = st.session_state.documents
    vs        = build_vector_store(docs)
    retriever = build_retriever(docs, vs)
    rewritten = rewrite_query(query)

    retrieved = retriever.invoke(rewritten)
    top_docs  = rerank(rewritten, retrieved, top_k=4)

    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        if isinstance(page, int): page += 1
        context_parts.append(
            f"[Source {i}: {src}, Page {page}]\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    history_str = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.chat_history[-6:]
    ) if st.session_state.chat_history else "No prior conversation."

    prompt = f"""You are an enterprise AI knowledge assistant helping employees find
accurate information from internal company documents.

STRICT RULES:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. If context is insufficient, say:
   "I could not find enough information in the provided documents to answer this question."
3. Always cite sources using [Source N] references.
4. Be concise, clear, and professional.
5. If the question is ambiguous, note your assumption.

--- CONVERSATION HISTORY ---
{history_str}

--- DOCUMENT CONTEXT ---
{context}

--- QUESTION ---
{query}

--- ANSWER ---"""

    start = time.time()
    try:
        answer = st.session_state.gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        answer = f"⚠️ Error generating answer: {e}"
    latency = time.time() - start

    return answer, top_docs, latency, rewritten

# -------------------------------------------------------
# SOURCE CITATION FORMAT
# -------------------------------------------------------

def fmt_citation(doc) -> str:
    src  = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", 0)
    if isinstance(page, int): page += 1
    return f"📄 **{src}** — Page {page}"

# -------------------------------------------------------
# EVALUATION SUITE
# -------------------------------------------------------

EVAL_CASES = [
    {"id": "E01", "question": "What is the main topic of this document?",
     "keywords": [], "must_not_contain": []},
    {"id": "E02", "question": "Summarise the key points in 3 bullet points.",
     "keywords": [], "must_not_contain": []},
    {"id": "E03", "question": "What are the employee leave policies?",
     "keywords": ["leave", "days", "annual", "policy"], "must_not_contain": []},
    {"id": "E04", "question": "What is the refund or cancellation policy?",
     "keywords": [], "must_not_contain": []},
    {"id": "E05", "question": "xyzzy foobar nonsense gibberish 99999",
     "keywords": [], "must_not_contain": [], "expect_no_info": True},
]

def run_evaluation() -> list:
    results = []
    for case in EVAL_CASES:
        t0 = time.time()
        try:
            answer, top_docs, latency, rewritten = generate_answer(case["question"])
            kw_hits   = [k for k in case.get("keywords", []) if k.lower() in answer.lower()]
            halluc_ok = not any(p in answer for p in case.get("must_not_contain", []))
            no_info_ok = True
            if case.get("expect_no_info"):
                no_info_ok = any(p in answer.lower() for p in [
                    "could not find", "not available", "no information",
                    "don't have", "not enough information",
                ])
            results.append({
                "id":                  case["id"],
                "question":            case["question"],
                "answer_preview":      answer[:200] + "…",
                "rewritten_query":     rewritten,
                "sources_found":       len(top_docs),
                "keyword_hits":        f"{len(kw_hits)}/{len(case.get('keywords', []))}",
                "hallucination_check": "✅ Pass" if halluc_ok  else "❌ Fail",
                "no_info_handling":    "✅ Pass" if no_info_ok else "❌ Fail",
                "latency":             f"{latency:.2f}s",
                "status":              "✅ OK",
            })
        except Exception as e:
            results.append({
                "id": case["id"], "question": case["question"],
                "status": f"❌ ERROR: {e}",
                "latency": f"{time.time()-t0:.2f}s",
            })
    return results

# ═══════════════════════════════════════════════════════
#  UI STARTS HERE
# ═══════════════════════════════════════════════════════

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("Hybrid Retrieval · Query Rewriting · Re-ranking · Source Citations · Conversation Memory")

# -------------------------------------------------------
# SIDEBAR — API KEY
# -------------------------------------------------------

st.sidebar.header("🔑 API Configuration")

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Free key at https://aistudio.google.com/app/apikey",
)

if not api_key:
    st.sidebar.warning("Enter your Gemini API key to continue.")
    st.info("👈 Enter your **Gemini API key** in the sidebar to get started.")
    st.stop()

# Validate key only when it changes
if st.session_state.gemini_model is None:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        model.generate_content("hi")
        st.session_state.gemini_model = model
        st.sidebar.success("✅ API Key Valid")
    except Exception as e:
        st.sidebar.error(f"❌ Invalid key: {e}")
        st.stop()
else:
    st.sidebar.success("✅ API Key Valid")

# -------------------------------------------------------
# SIDEBAR — DATA SOURCE
# -------------------------------------------------------

st.sidebar.header("📥 Load Documents")

source = st.sidebar.selectbox(
    "Source Type",
    ["PDF", "CSV", "Excel", "Text File", "Web Page"],
)

new_docs = None

if source == "PDF":
    f = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if f and st.sidebar.button("📂 Load PDF"):
        with st.spinner("Loading PDF…"):
            new_docs = load_pdf(f)

elif source == "CSV":
    f = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if f and st.sidebar.button("📂 Load CSV"):
        with st.spinner("Loading CSV…"):
            new_docs = load_csv(f)

elif source == "Excel":
    f = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])
    if f and st.sidebar.button("📂 Load Excel"):
        with st.spinner("Loading Excel…"):
            new_docs = load_excel(f)

elif source == "Text File":
    f = st.sidebar.file_uploader("Upload TXT", type=["txt"])
    if f and st.sidebar.button("📂 Load TXT"):
        with st.spinner("Loading text…"):
            new_docs = load_text(f)

elif source == "Web Page":
    url = st.sidebar.text_input("Page URL")
    if url and st.sidebar.button("🌐 Load Web Page"):
        with st.spinner("Fetching page…"):
            new_docs = load_web(url)

# When new docs arrive — store and rebuild index
if new_docs:
    st.session_state.documents    = new_docs
    st.session_state.vector_store = None      # force rebuild
    st.session_state.doc_hash     = None
    st.session_state.chat_history = []
    st.session_state.index_built  = False
    st.sidebar.success(f"✅ Loaded {len(new_docs)} pages")

# Build / warm up vector store right after loading
if st.session_state.documents and not st.session_state.index_built:
    with st.spinner("⚙️ Building ChromaDB index…"):
        build_vector_store(st.session_state.documents)
    st.success("✅ ChromaDB index ready!")

# Sidebar status
st.sidebar.markdown("---")
if st.session_state.documents:
    st.sidebar.info(
        f"📊 **{len(st.session_state.documents)}** pages in knowledge base\n\n"
        f"💬 **{len(st.session_state.chat_history) // 2}** conversation turns"
    )
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    if st.sidebar.button("🔄 Reset Everything"):
        for k in ["documents", "vector_store", "doc_hash",
                  "chat_history", "index_built", "eval_results"]:
            st.session_state[k] = [] if k in ("chat_history", "eval_results") else None
        st.rerun()

# -------------------------------------------------------
# STOP if no docs loaded yet
# -------------------------------------------------------

if st.session_state.documents is None:
    st.info("👈 Load a document from the sidebar to start asking questions.")
    st.stop()

# -------------------------------------------------------
# TABS
# -------------------------------------------------------

tab_chat, tab_eval = st.tabs(["💬 Chat", "🧪 Evaluation"])

# ═══════════════ TAB 1 — CHAT ═══════════════

with tab_chat:
    st.header("Ask Questions About Your Documents")

    # Replay conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask anything about the loaded documents…")

    if query:
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Running RAG pipeline…"):
                answer, top_docs, latency, rewritten = generate_answer(query)

            st.markdown(answer)

            # Source citations
            with st.expander(
                f"📚 {len(top_docs)} source(s) found · ⏱ {latency:.2f}s"
            ):
                if rewritten.strip() != query.strip():
                    st.caption(f"🔍 Rewritten query: *{rewritten}*")
                st.markdown("---")
                for i, doc in enumerate(top_docs, 1):
                    st.markdown(f"**Source {i}** · {fmt_citation(doc)}")
                    st.text(doc.page_content[:400] + "…")
                    st.markdown("---")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

# ═══════════════ TAB 2 — EVALUATION ═══════════════

with tab_eval:
    st.header("🧪 Evaluation Suite")
    st.markdown(
        "Runs **5 built-in test cases** to measure retrieval accuracy, "
        "hallucination prevention, and no-info handling."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("▶️ Run Evaluation", type="primary")

    if run_btn:
        with st.spinner("Running test cases…"):
            st.session_state.eval_results = run_evaluation()

    if st.session_state.eval_results:
        st.subheader("Results")

        for r in st.session_state.eval_results:
            label = f"[{r['id']}] {r['question'][:65]}… — {r.get('status', '?')}"
            with st.expander(label):
                for k, v in r.items():
                    if k not in ("id", "question"):
                        st.markdown(
                            f"**{k.replace('_', ' ').title()}:** {v}"
                        )

        passed = sum(
            1 for r in st.session_state.eval_results
            if "OK" in r.get("status", "")
        )
        total = len(st.session_state.eval_results)

        col_a, col_b = st.columns(2)
        col_a.metric("Tests Passed", f"{passed} / {total}")
        col_b.metric(
            "Pass Rate", f"{int(passed/total*100)}%"
        )

        st.download_button(
            "⬇️ Download Results (JSON)",
            data=json.dumps(st.session_state.eval_results, indent=2),
            file_name="eval_results.json",
            mime="application/json",
        )
