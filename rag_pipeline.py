import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    if not text.strip():
        raise ValueError("No extractable text found in the PDF. It may be a scanned document.")
    return text


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def build_vectorstore(pdf_path: str):
    """
    Full pipeline: extract text → chunk → embed → store in FAISS.
    Returns a FAISS vectorstore ready for retrieval.
    """
    print(f"[1/3] Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print(f"[2/3] Chunking text into segments...")
    chunks = chunk_text(text)
    print(f"      → {len(chunks)} chunks created.")

    print(f"[3/3] Embedding chunks and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"      → FAISS index built successfully.")

    return vectorstore

def build_qa_chain(vectorstore):
    """Build a RetrievalQA chain using Groq LLM and FAISS retriever."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2
    )

    prompt_template = """You are a helpful assistant that answers questions based strictly on the provided document context.
If the answer is not found in the context, say "I could not find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain


def answer_question(vectorstore, question: str) -> dict:
    """
    Run the RAG pipeline for a given question.
    Returns the answer and the retrieved source chunks.
    """
    chain = build_qa_chain(vectorstore)
    result = chain.invoke({"query": question})

    source_chunks = [doc.page_content[:300] for doc in result.get("source_documents", [])]

    return {
        "answer": result["result"],
        "source_chunks": source_chunks
    }
