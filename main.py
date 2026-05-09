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
      :root{
        --bg0:#0b1220;
        --bg1:#0f172a;
        --card:#0b1220cc;
        --stroke:#253049;
        --text:#e5e7eb;
        --muted:#9ca3af;
        --brand:#60a5fa;
        --brand2:#22c55e;
        --danger:#ef4444;
        --shadow: 0 20px 70px rgba(0,0,0,.45);
        --radius: 16px;
      }
      *{ box-sizing:border-box; }
      body{
        margin:0;
        min-height:100vh;
        color:var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
        line-height:1.45;
        background:
          radial-gradient(1100px 600px at 20% -10%, rgba(96,165,250,.35), transparent 60%),
          radial-gradient(900px 520px at 85% 0%, rgba(34,197,94,.25), transparent 55%),
          linear-gradient(180deg, var(--bg1), var(--bg0));
        padding: 40px 18px;
      }
      a{ color:var(--brand); text-decoration:none; }
      a:hover{ text-decoration:underline; }

      .wrap{ max-width: 980px; margin: 0 auto; }
      .topbar{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap: 12px;
        margin-bottom: 18px;
      }
      .brand{
        display:flex;
        align-items:center;
        gap:10px;
      }
      .logo{
        width:34px;height:34px;border-radius:10px;
        background: linear-gradient(135deg, rgba(96,165,250,.95), rgba(34,197,94,.75));
        box-shadow: 0 10px 30px rgba(96,165,250,.2);
      }
      .brand h1{ font-size: 18px; margin:0; letter-spacing:.2px; }
      .pill{
        border:1px solid var(--stroke);
        background: rgba(255,255,255,.04);
        color: var(--muted);
        padding: 7px 10px;
        border-radius: 999px;
        font-size: 13px;
        display:flex;
        gap:10px;
        align-items:center;
        white-space:nowrap;
      }
      .dot{ width:8px;height:8px;border-radius:999px;background:var(--brand2); box-shadow:0 0 0 3px rgba(34,197,94,.15); }

      .card{
        border: 1px solid var(--stroke);
        background: rgba(6,10,20,.65);
        backdrop-filter: blur(10px);
        border-radius: var(--radius);
        padding: 18px;
        box-shadow: var(--shadow);
      }
      .hero{
        padding: 18px 18px 6px;
      }
      .hero h2{ font-size: 26px; margin: 0 0 6px; letter-spacing: .2px; }
      .muted{ color: var(--muted); font-size: 14px; }

      .grid{
        display:grid;
        grid-template-columns: 1.2fr .8fr;
        gap: 16px;
        padding: 0 18px 18px;
      }
      @media (max-width: 880px){
        .grid{ grid-template-columns: 1fr; }
      }

      label{ display:block; margin-top: 10px; font-weight: 600; font-size: 13px; color: #d1d5db; }

      .field{
        margin-top: 10px;
        padding: 14px;
        border: 1px solid var(--stroke);
        border-radius: 14px;
        background: rgba(255,255,255,.03);
      }
      input[type="text"]{
        width: 100%;
        padding: 12px 12px;
        border: 1px solid var(--stroke);
        border-radius: 12px;
        background: rgba(2,6,23,.55);
        color: var(--text);
        outline: none;
      }
      input[type="text"]:focus{
        border-color: rgba(96,165,250,.7);
        box-shadow: 0 0 0 4px rgba(96,165,250,.15);
      }
      input[type="file"]{ width: 100%; color: var(--muted); }
      .row{ display:flex; gap: 12px; align-items:center; flex-wrap:wrap; margin-top: 12px; }

      button{
        appearance:none;
        border: 1px solid rgba(96,165,250,.35);
        background: linear-gradient(135deg, rgba(96,165,250,.95), rgba(96,165,250,.55));
        color: #06121f;
        font-weight: 700;
        padding: 11px 14px;
        border-radius: 12px;
        cursor:pointer;
        transition: transform .06s ease, filter .12s ease, opacity .12s ease;
      }
      button:hover{ filter: brightness(1.05); }
      button:active{ transform: translateY(1px); }
      button:disabled{ opacity: .55; cursor:not-allowed; }
      .btn-secondary{
        background: rgba(255,255,255,.04);
        color: var(--text);
        border-color: var(--stroke);
        font-weight: 600;
      }

      .status{
        margin-top: 10px;
        font-size: 14px;
        color: var(--muted);
        min-height: 20px;
        display:flex;
        align-items:center;
        gap: 10px;
      }
      .spinner{
        width: 14px; height: 14px;
        border-radius: 999px;
        border: 2px solid rgba(255,255,255,.22);
        border-top-color: rgba(255,255,255,.8);
        animation: spin 0.8s linear infinite;
      }
      @keyframes spin{ to{ transform: rotate(360deg); } }

      .answer{
        margin-top: 10px;
        white-space: pre-wrap;
        background: rgba(2,6,23,.55);
        border: 1px solid var(--stroke);
        padding: 14px;
        border-radius: 14px;
      }
      .err{ color: #fecaca; }
      .kvs{ display:grid; gap:10px; }
      .kv{
        border:1px solid var(--stroke);
        background: rgba(255,255,255,.03);
        border-radius: 14px;
        padding: 12px;
      }
      .kv b{ display:block; font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }
      .kv code{ background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); padding: 2px 6px; border-radius: 999px; color: #e5e7eb; }
      .footer{
        margin-top: 16px;
        display:flex;
        justify-content:space-between;
        gap: 10px;
        flex-wrap:wrap;
        color: var(--muted);
        font-size: 13px;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div class="brand">
          <div class="logo" aria-hidden="true"></div>
          <div>
            <h1>IntelliRead</h1>
            <div class="muted">Ask questions across multiple PDFs</div>
          </div>
        </div>
        <div class="pill"><span class="dot"></span><span>Online</span></div>
      </div>

      <div class="card">
        <div class="hero">
          <h2>PDF Q&amp;A</h2>
          <div class="muted">Upload documents, then ask a question. Answers are generated from the uploaded text.</div>
        </div>

        <div class="grid">
          <div class="field">
            <label for="pdfs">Documents (PDF)</label>
            <input id="pdfs" type="file" multiple accept="application/pdf" />
            <div class="muted" id="fileHint" style="margin-top:8px">No files selected.</div>

            <label for="question" style="margin-top:14px">Question</label>
            <input id="question" type="text" placeholder="e.g., What is the main conclusion in the report?" />

            <div class="row">
              <button id="askBtn">Ask question</button>
              <button id="clearBtn" class="btn-secondary" type="button">Clear</button>
              <div class="muted">Tip: try specific questions.</div>
            </div>

            <div class="status" id="status"></div>
            <div id="answer" class="answer" style="display:none"></div>
          </div>

          <div class="kvs">
            <div class="kv">
              <b>Endpoint</b>
              <div><code>POST /chat</code> (multipart)</div>
            </div>
            <div class="kv">
              <b>Fields</b>
              <div class="muted"><code>question</code> (text) · <code>pdfs</code> (files)</div>
            </div>
            <div class="kv">
              <b>Docs</b>
              <div><a href="/docs">Open Swagger UI</a></div>
            </div>
            <div class="kv">
              <b>Status</b>
              <div><a href="/health">Health check</a></div>
            </div>
          </div>
        </div>

        <div class="footer">
          <div>Built for deployed use on Vercel.</div>
          <div class="muted">If answers are missing, check your <code>OPENAI_API_KEY</code>.</div>
        </div>
      </div>
    </div>
    <script>
      const pdfsEl = document.getElementById('pdfs');
      const qEl = document.getElementById('question');
      const btn = document.getElementById('askBtn');
      const statusEl = document.getElementById('status');
      const answerEl = document.getElementById('answer');
      const clearBtn = document.getElementById('clearBtn');
      const fileHint = document.getElementById('fileHint');

      function setBusy(busy, msg) {
        btn.disabled = busy;
        if (!msg) {
          statusEl.innerHTML = '';
          return;
        }
        if (busy) {
          statusEl.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>' + msg + '</span>';
        } else {
          statusEl.innerHTML = '<span>' + msg + '</span>';
        }
      }

      function setError(msg) {
        statusEl.innerHTML = '<span class="err">' + msg + '</span>';
      }

      function updateFileHint() {
        const files = pdfsEl.files;
        if (!files || files.length === 0) {
          fileHint.textContent = 'No files selected.';
          return;
        }
        const names = Array.from(files).map(f => f.name);
        const shown = names.slice(0, 3).join(', ');
        fileHint.textContent = names.length <= 3 ? shown : (shown + ` (+${names.length - 3} more)`);
      }

      pdfsEl.addEventListener('change', updateFileHint);

      btn.addEventListener('click', async () => {
        const files = pdfsEl.files;
        const question = (qEl.value || '').trim();
        answerEl.style.display = 'none';
        answerEl.textContent = '';

        if (!files || files.length === 0) {
          setError('Please select at least one PDF.');
          return;
        }
        if (!question) {
          setError('Please enter a question.');
          return;
        }

        const fd = new FormData();
        fd.append('question', question);
        for (const f of files) fd.append('pdfs', f, f.name);

        try {
          setBusy(true, 'Working…');
          const res = await fetch('/chat', { method: 'POST', body: fd });
          const data = await res.json().catch(() => ({}));
          if (!res.ok) {
            throw new Error(data.detail || ('Request failed: ' + res.status));
          }
          answerEl.textContent = data.answer || '(No answer returned.)';
          answerEl.style.display = 'block';
          setBusy(false, '');
        } catch (e) {
          setError(String(e && e.message ? e.message : e));
        }
      });

      clearBtn.addEventListener('click', () => {
        qEl.value = '';
        pdfsEl.value = '';
        updateFileHint();
        setBusy(false, '');
        answerEl.style.display = 'none';
        answerEl.textContent = '';
      });

      updateFileHint();
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

