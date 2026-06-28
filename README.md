# 🧠 OmniSum — Enterprise Knowledge Copilot

An AI-powered Retrieval Augmented Generation (RAG) system that lets employees ask questions in plain language and get accurate, cited answers from internal company documents — no manual searching required.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 Hybrid Retrieval | BM25 keyword search (40%) + ChromaDB vector search with MMR (60%) |
| 🔄 Query Rewriting | LLM rewrites the user's question for better retrieval, resolving pronouns from chat history |
| 📊 Re-ranking | Retrieved chunks are re-scored by keyword overlap + page-position bonus |
| 💬 Conversation Memory | Last 6 chat turns kept in context for follow-up questions |
| 📌 Source Citations | Every answer shows document name, page number, and the exact passage used |
| 🧪 Evaluation Suite | Built-in test runner with 5 test cases; results exportable as JSON |
| 🛡️ Hallucination Guard | Prompt enforces answers strictly from context; explicit fallback when info is unavailable |

---

## 🗂️ Supported Document Types

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| Plain Text | `.txt` |
| Web Page | Any public URL |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 Query Rewriting  ──── Gemini 1.5 Flash  (resolves pronouns, improves search terms)
      │
      ▼
 Hybrid Retrieval
  ├── BM25 keyword search        (weight: 40%)
  └── ChromaDB MMR vector search (weight: 60%)
      │
      ▼
 Re-ranking  (keyword overlap + early-page bonus)
      │
      ▼
 Top-4 Chunks + Metadata  (source, page number)
      │
      ▼
 Answer Generation  ──── Gemini 1.5 Flash
  └── Strict grounding prompt + conversation history
      │
      ▼
 Streamlit UI
  ├── 💬 Chat Tab    — conversational Q&A with source expanders
  └── 🧪 Eval Tab   — automated test suite with JSON export
```

### Data flow for document ingestion

```
File Upload / URL
      │
      ▼
 Document Loader  (PyPDF / CSV / Excel / Text / Web)
      │
      ▼
 Metadata Enrichment  (source name, page number)
      │
      ▼
 Recursive Text Splitter  (1000 tokens, 200 overlap)
      │
      ▼
 HuggingFace Embeddings  (all-MiniLM-L6-v2, CPU)
      │
      ▼
 ChromaDB  (persistent, disk-backed, cosine similarity)
```

---

## 🔧 Technology Choices

| Component | Choice | Reason |
|---|---|---|
| LLM | Gemini 2.5 Flash | Free tier, 1M token context, fast |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, CPU-friendly, strong semantic quality |
| Vector DB | ChromaDB (persistent) | Open-source, disk-backed (survives restarts), cosine similarity built-in |
| Keyword search | BM25 via LangChain | Catches exact terms, codes, and policy names that semantic search misses |
| Vector search mode | MMR | Maximal Marginal Relevance reduces redundant chunks in results |
| Framework | Streamlit | Rapid UI, native chat interface, built-in session state |

### Chunking strategy

`RecursiveCharacterTextSplitter` with **1000-token chunks and 200-token overlap**, splitting on paragraph → sentence → word boundaries. This preserves semantic coherence — a policy rule is unlikely to be cut mid-sentence.

### Hybrid retrieval weights

- **Vector (60%)** handles paraphrasing, synonyms, and semantic intent
- **BM25 (40%)** handles exact term matching — essential for policy codes, names, and specific numbers

### Re-ranking heuristic

Each retrieved chunk is re-scored:  
`score = keyword_overlap_count + max(0, 5 − page_number)`  
Earlier pages in enterprise documents (like HR policies) tend to contain key definitions and summaries, so they receive a small bonus.

---

## 🚀 Quick Start

### 1. Clone / download

```
your-project/
├── assistant.py
├── requirements.txt
└── README.md
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Gemini API key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key → Create API Key**
3. Copy the key (starts with `AIza...`)

### 4. Run the app

```bash
streamlit run assistant.py
```

Then paste your API key into the sidebar when the app opens.

---

## 📦 Requirements

```
streamlit>=1.35.0
google-generativeai>=0.7.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.3
langchain-text-splitters>=0.2.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pypdf>=4.0.0
unstructured[xlsx]>=0.14.0
openpyxl>=3.1.0
rank-bm25>=0.2.2
requests>=2.31.0
beautifulsoup4>=4.12.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🧪 Evaluation Suite

Click the **🧪 Evaluation** tab in the app to run the built-in test suite. It runs 5 test cases automatically against your loaded document:

| ID | Question | What it tests |
|---|---|---|
| E01 | What is the main topic of this document? | General comprehension |
| E02 | Summarise the key points in 3 bullet points | Summarisation |
| E03 | What are the employee leave policies? | Keyword retrieval accuracy |
| E04 | What is the refund or cancellation policy? | Cross-domain retrieval |
| E05 | `xyzzy foobar nonsense gibberish` | Hallucination prevention |

Results show:
- **Rewritten query** used for retrieval
- **Sources found** (chunk count)
- **Keyword hits** (for E03/E04)
- **Hallucination check** (must-not-contain phrases)
- **No-info handling** (E05 must refuse to answer)
- **Latency** per query

Results can be downloaded as **JSON** for submission.

---

## 💬 Usage Examples

**HR policy question:**
> "What is the employee leave policy?"
> → "Employees are eligible for 30 paid leaves annually. [Source 1: HR_Policy.pdf, Page 12]"

**Follow-up (conversation memory):**
> "What about sick leave?"
> → Query is rewritten to "What is the sick leave policy for employees?" before retrieval

**No information available:**
> "What is the refund policy?" *(on an HR document)*
> → "I could not find enough information in the provided documents to answer this question."

---

## ⚠️ Known Limitations

- **ChromaDB is temp-directory scoped** — the index may be cleared on machine restart. For production use, set a fixed `persist_directory`.
- **Re-ranking is heuristic** — keyword overlap is a proxy for relevance. A neural cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would be more accurate.
- **BM25 is rebuilt per query** — on very large corpora (>50k chunks) this adds latency. Caching the BM25 index in session state would fix this.
- **No authentication** — anyone with the URL can use the app. Add Streamlit's built-in auth or a reverse proxy for production.
- **Single document at a time** — loading a new document replaces the previous index.

---

## 🔮 Future Improvements

- **Neural cross-encoder reranking** for higher answer precision
- **Multi-document indexing** — query across several files simultaneously
- **Persistent user sessions** with login and per-user document namespacing
- **RAGAS evaluation metrics** — faithfulness, answer relevancy, context precision
- **FastAPI backend** to expose a `POST /ask` REST endpoint
- **Streaming responses** via Gemini's streaming API for faster perceived latency
- **Confidence scores** derived from retrieval similarity distances

---

## 📁 Project Structure

```
assistant.py          # Main application (all-in-one Streamlit app)
requirements.txt      # Python dependencies
README.md             # This file
```

---

## 🔑 Environment

No `.env` file needed — the Gemini API key is entered securely through the Streamlit sidebar UI at runtime and is never stored to disk.
