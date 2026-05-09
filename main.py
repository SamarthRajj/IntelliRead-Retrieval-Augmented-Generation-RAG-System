import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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


def get_vectorstore(text_chunks: List[str]) -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )


@app.get("/health")
def health():
    return {"ok": True}


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
    vectorstore = get_vectorstore(chunks)
    chain = get_conversation_chain(vectorstore)

    try:
        result = chain({"question": question})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate response.") from e

    chat_history = result.get("chat_history", [])
    answer = ""
    if chat_history:
        answer = getattr(chat_history[-1], "content", "") or ""

    return JSONResponse(
        {
            "question": question,
            "answer": answer,
            "session_id": session_id,
        }
    )

