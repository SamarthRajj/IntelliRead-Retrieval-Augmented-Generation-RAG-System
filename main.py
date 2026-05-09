import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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
    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>IntelliRead</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 32px; line-height: 1.5; }
      .card { max-width: 820px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px 22px; }
      code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
      a { color: #2563eb; text-decoration: none; }
      a:hover { text-decoration: underline; }
      ul { margin: 10px 0 0 18px; }
      label { display:block; margin-top: 12px; font-weight: 600; }
      input[type="text"] { width: 100%; box-sizing: border-box; padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 10px; }
      input[type="file"] { width: 100%; }
      button { margin-top: 12px; padding: 10px 14px; border: 1px solid #111827; background: #111827; color: white; border-radius: 10px; cursor: pointer; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .muted { color: #6b7280; font-size: 14px; }
      pre { white-space: pre-wrap; background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; border-radius: 12px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>IntelliRead</h2>
      <p class="muted">Upload PDFs, ask a question, get an answer.</p>

      <label for="pdfs">PDF files</label>
      <input id="pdfs" type="file" multiple accept="application/pdf" />

      <label for="question">Question</label>
      <input id="question" type="text" placeholder="Ask a question about your documents..." />

      <button id="askBtn">Ask</button>

      <div style="margin-top:16px">
        <div class="muted" id="status"></div>
        <pre id="answer" style="display:none"></pre>
      </div>

      <hr style="border:none;border-top:1px solid #e5e7eb;margin:18px 0" />
      <div class="muted">
        <a href="/docs">API docs</a> · <a href="/health">health</a>
      </div>
    </div>
    <script>
      const pdfsEl = document.getElementById('pdfs');
      const qEl = document.getElementById('question');
      const btn = document.getElementById('askBtn');
      const statusEl = document.getElementById('status');
      const answerEl = document.getElementById('answer');

      function setBusy(busy, msg) {
        btn.disabled = busy;
        statusEl.textContent = msg || '';
      }

      btn.addEventListener('click', async () => {
        const files = pdfsEl.files;
        const question = (qEl.value || '').trim();
        answerEl.style.display = 'none';
        answerEl.textContent = '';

        if (!files || files.length === 0) {
          setBusy(false, 'Please select at least one PDF.');
          return;
        }
        if (!question) {
          setBusy(false, 'Please enter a question.');
          return;
        }

        const fd = new FormData();
        fd.append('question', question);
        for (const f of files) fd.append('pdfs', f, f.name);

        try {
          setBusy(true, 'Thinking...');
          const res = await fetch('/chat', { method: 'POST', body: fd });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            throw new Error(data.detail || ('Request failed: ' + res.status));
          }
          answerEl.textContent = data.answer || '(empty answer)';
          answerEl.style.display = 'block';
          setBusy(false, '');
        } catch (e) {
          setBusy(false, String(e && e.message ? e.message : e));
        }
      });
    </script>
  </body>
</html>
""".strip()
    )


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

