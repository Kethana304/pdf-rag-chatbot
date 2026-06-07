from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

from rag_pipeline import build_vectorstore, answer_question

app = FastAPI(
    title="PDF Q&A RAG Chatbot",
    description="Upload a PDF and ask questions about it using RAG + Groq LLM.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory store: one vectorstore per session (keyed by filename)
vectorstore_cache = {}


class QuestionRequest(BaseModel):
    filename: str
    question: str


@app.get("/")
def root():
    return {"message": "PDF RAG Chatbot is running. Use /upload to upload a PDF and /ask to query it."}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. It will be chunked and indexed into a FAISS vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        vectorstore = build_vectorstore(tmp_path)
        vectorstore_cache[file.filename] = vectorstore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully.",
        "filename": file.filename
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Ask a question about a previously uploaded PDF.
    """
    if request.filename not in vectorstore_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No vectorstore found for '{request.filename}'. Please upload the PDF first."
        )

    vectorstore = vectorstore_cache[request.filename]

    try:
        result = answer_question(vectorstore, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    return {
        "filename": request.filename,
        "question": request.question,
        "answer": result["answer"],
        "source_chunks": result["source_chunks"]
    }


@app.get("/files")
def list_uploaded_files():
    """
    List all currently indexed PDF files.
    """
    return {"indexed_files": list(vectorstore_cache.keys())}
