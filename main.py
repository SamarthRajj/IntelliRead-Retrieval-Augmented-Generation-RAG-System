import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

app = FastAPI(title="MultiPDF Chat API")


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Missing OPENAI_API_KEY environment variable.",
        )


def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF: {pdf.filename}") from e
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted
    return text


def get_text_chunks(text: str) -> List[str]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def _embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents(texts)
    return np.array(vectors, dtype=np.float32)


def _embed_query(query: str) -> np.ndarray:
    embeddings = OpenAIEmbeddings()
    v = embeddings.embed_query(query)
    return np.array(v, dtype=np.float32)


def _top_k_chunks(question: str, chunks: List[str], k: int = 4) -> List[str]:
    if not chunks:
        return []

    doc_vecs = _embed_texts(chunks)  # (n, d)
    q = _embed_query(question)  # (d,)

    doc_norms = np.linalg.norm(doc_vecs, axis=1) + 1e-10
    q_norm = float(np.linalg.norm(q) + 1e-10)
    sims = (doc_vecs @ q) / (doc_norms * q_norm)

    k = min(k, len(chunks))
    idx = np.argsort(-sims)[:k]
    return [chunks[int(i)] for i in idx]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
async def root():
    return {"message": "IntelliRead API is running!"}


@app.post("/chat")
async def chat(
    question: str = Form(...),
    pdfs: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    # Note: Vercel serverless is stateless; `session_id` is accepted for clients,
    # but conversation memory is per-request unless you add external storage.
    _require_openai_key()

    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDFs uploaded.")

    raw_text = get_pdf_text(pdfs)
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in PDFs.")

    chunks = get_text_chunks(raw_text)
    context_chunks = _top_k_chunks(question, chunks, k=4)
    context = "\n\n---\n\n".join(context_chunks)

    llm = ChatOpenAI()
    messages = [
        SystemMessage(
            content=(
                "You answer questions using ONLY the provided PDF context. "
                "If the answer isn't in the context, say you don't know."
            )
        ),
        HumanMessage(
            content=f"PDF context:\n{context}\n\nQuestion: {question}",
        ),
    ]

    try:
        answer = llm(messages).content
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate response.") from e

    return JSONResponse(
        {
            "question": question,
            "answer": answer,
            "session_id": session_id,
        }
    )

