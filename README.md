# PDF Q&A RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask natural language questions about them. Built with **LangChain**, **FAISS**, **FastAPI**, and **Groq LLM (LLaMA 3 70B)**.

---

## Architecture

```
User uploads PDF
      │
      ▼
Text Extraction (pypdf)
      │
      ▼
Text Chunking (RecursiveCharacterTextSplitter)
      │
      ▼
Embedding (HuggingFace: all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Index
      │
   Question
      │
      ▼
Semantic Retrieval (Top-4 chunks)
      │
      ▼
LLM Generation (Groq: LLaMA 3 70B)
      │
      ▼
Answer + Source Chunks
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| API Framework | FastAPI |
| RAG Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA 3 70B) |
| PDF Parsing | pypdf |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at: https://console.groq.com

### 5. Run the server
```bash
uvicorn main:app --reload
```
Server runs at: `http://localhost:8000`

---

## API Endpoints

### `GET /`
Health check — confirms the server is running.

### `POST /upload`
Upload a PDF file to be indexed.

**Request:** `multipart/form-data` with a `file` field (PDF only)

**Response:**
```json
{
  "message": "'document.pdf' uploaded and indexed successfully.",
  "filename": "document.pdf"
}
```

### `POST /ask`
Ask a question about an uploaded PDF.

**Request:**
```json
{
  "filename": "document.pdf",
  "question": "What are the main findings of this report?"
}
```

**Response:**
```json
{
  "filename": "document.pdf",
  "question": "What are the main findings of this report?",
  "answer": "The main findings are...",
  "source_chunks": ["chunk1 text...", "chunk2 text..."]
}
```

### `GET /files`
List all currently indexed PDF files.

---

## Example Usage (curl)

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"filename": "your_document.pdf", "question": "Summarize the key points."}'
```

---

## Interactive API Docs

FastAPI provides auto-generated docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Project Structure

```
pdf-rag-chatbot/
├── main.py           # FastAPI app and route definitions
├── rag_pipeline.py   # RAG logic: extraction, chunking, embedding, retrieval, generation
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore
└── README.md
```

---

## Key Design Decisions

- **RecursiveCharacterTextSplitter** with 500-token chunks and 50-token overlap ensures context is preserved across chunk boundaries.
- **all-MiniLM-L6-v2** is a lightweight but highly effective embedding model — no API key required, runs locally.
- **Top-4 chunk retrieval** balances context richness with LLM token limits.
- **Groq API** provides ultra-fast LLM inference, making responses near real-time.
- **In-memory vectorstore cache** allows multiple PDFs to be indexed in the same session without re-processing.

---
## Production Deployment

Deployed on AWS EC2 (Ubuntu, free-tier t3.small) with systemd for process management and GitHub Actions for CI/CD — every push to `main` triggers automatic redeployment.

**The hard part wasn't the RAG pipeline — it was making it run on 1GB of RAM.**
Loading the HuggingFace embedding model crashed the instance with OOM errors on first deploy. Fixed by:
- Adding swap space to cover the memory gap during model load
- Switching to a CPU-only PyTorch build (no CUDA dependencies needed, and a much smaller install)
- Running the app under systemd so it auto-restarts on failure instead of silently dying

This taught me that "production AI" is mostly disciplined systems engineering — memory budgeting, process supervision, deploy automation — not the model itself.

## Author

**Kethana Harshitha**  
B.E. in Artificial Intelligence and Data Science  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
