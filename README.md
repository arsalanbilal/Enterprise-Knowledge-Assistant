# 🧠 Enterprise Knowledge Assistant

> An AI-powered document Q&A platform built with Streamlit, Google Gemini, and a hybrid RAG pipeline — enabling employees to instantly find accurate answers from internal company documents.

---

## 📌 Overview

Enterprise Knowledge Assistant is a production-ready, conversational AI application that lets teams query their own documents using natural language. It combines semantic vector search with keyword-based retrieval, intelligent query rewriting, and source citations — all wrapped in a clean, browser-based interface.

Whether the knowledge lives in PDFs, spreadsheets, CSVs, plain text files, or web pages, this tool indexes it and makes it instantly searchable through a chat interface powered by Google Gemini 2.5 Flash.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 Multi-format ingestion | Upload PDF, CSV, Excel, TXT, or load any web page URL |
| 🔍 Hybrid retrieval | Combines BM25 keyword search + ChromaDB vector search via an Ensemble Retriever |
| ✍️ Query rewriting | Gemini rewrites ambiguous or pronoun-heavy questions before retrieval |
| 📊 Re-ranking | Token-overlap + page-position scoring surfaces the most relevant chunks |
| 💬 Conversation memory | Multi-turn chat with rolling history context |
| 📚 Source citations | Every answer links back to the exact document page it came from |
| 🧪 Built-in evaluation suite | 5 automated test cases measure retrieval accuracy, hallucination prevention, and graceful no-info handling |
| ⬇️ Export results | Download evaluation results as JSON in one click |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Query Rewriter (Gemini)
    │
    ▼
Hybrid Retriever
 ├── BM25 Retriever        (keyword match, weight 0.4)
 └── ChromaDB MMR Retriever (semantic search, weight 0.6)
    │
    ▼
Re-Ranker (token overlap + page bonus)
    │
    ▼
Answer Generator (Gemini 2.5 Flash)
    │
    ▼
Response + Source Citations
```

Documents are chunked with `RecursiveCharacterTextSplitter` (1 000-token chunks, 200-token overlap) and embedded using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.

---

## 🛠️ Tech Stack

- **Frontend / UI** — [Streamlit](https://streamlit.io/)
- **LLM** — Google Gemini 2.5 Flash (`google-generativeai`)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs on CPU)
- **Vector Store** — [ChromaDB](https://www.trychroma.com/) (persisted locally)
- **Retrieval Framework** — [LangChain](https://www.langchain.com/) (loaders, splitters, retrievers)
- **Keyword Search** — BM25 (`langchain-community`)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A free [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run assistant.py
```

### Dependencies (`requirements.txt`)

```
streamlit
google-generativeai
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-classic
langchain-huggingface
chromadb
sentence-transformers
pypdf
unstructured[xlsx]
rank-bm25
```

---

## 📖 How to Use

1. **Enter your Gemini API key** in the sidebar (stored only for the session, never saved).
2. **Select a source type** — PDF, CSV, Excel, Text File, or Web Page — and upload or paste a URL.
3. **Click Load** — the app chunks the document, builds embeddings, and indexes them in ChromaDB.
4. **Ask questions** in the chat tab. Each answer includes cited sources and a rewritten query preview.
5. **Run the Evaluation tab** to benchmark retrieval accuracy on 5 built-in test cases.

---

## 🧪 Evaluation Suite

The built-in evaluation runs 5 test cases automatically against the loaded document:

| ID | Question | What it tests |
|---|---|---|
| E01 | What is the main topic? | Basic comprehension |
| E02 | Summarise in 3 bullet points | Summarisation |
| E03 | What are the leave policies? | Keyword retrieval accuracy |
| E04 | What is the refund/cancellation policy? | Domain-specific retrieval |
| E05 | `xyzzy foobar nonsense gibberish` | Hallucination / no-info handling |

Each result reports: answer preview, rewritten query, sources found, keyword hit rate, hallucination check, no-info handling, and latency. Results are downloadable as JSON.

---
##📁 Project Structure
enterprise-knowledge-assistant/
│
├── assistant.py              
├── requirements.txt          
├── Dockerfile                
├── docker-compose.yaml       
├── README.md
```

---

## ⚙️ Configuration

All configuration is handled at runtime via the Streamlit UI. Key parameters are set as constants in `assistant.py`:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1 000 | Characters per document chunk |
| `chunk_overlap` | 200 | Overlap between consecutive chunks |
| `BM25 k` | 5 | Number of BM25 results to retrieve |
| `MMR k` | 5 | Number of vector results to retrieve |
| `MMR fetch_k` | 20 | Candidate pool for MMR diversity |
| `Ensemble weights` | 0.4 / 0.6 | BM25 vs. vector search balance |
| `Re-rank top_k` | 4 | Final chunks passed to the LLM |

---

## 🔒 Privacy & Security

- The Gemini API key is entered at runtime and held only in Streamlit session state — it is never written to disk.
- Uploaded documents are written to a system temp directory, used for indexing, then deleted immediately.
- ChromaDB data is stored in the system temp directory and is wiped on OS restart.
- No document content is sent to any third-party service other than the Google Gemini API for answer generation.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss your idea before submitting a pull request. Make sure to follow existing code style and include a brief description of what your change does.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

Built with [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Streamlit](https://streamlit.io/), and [Google Gemini](https://deepmind.google/technologies/gemini/).
