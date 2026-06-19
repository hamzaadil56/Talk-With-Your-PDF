# Talk With Your PDF

Upload a PDF and hold a conversation with it. Answers are **retrieval-augmented** —
drawn from the pages you uploaded, cited with page numbers, and generated in seconds.

A production-grade rebuild of the original Streamlit prototype, designed to run as a
**single Vercel project**: a **Next.js** frontend and a **FastAPI** backend, with a
serverless-native RAG pipeline.

## Architecture

```
Next.js (landing + chat)  ──/api/*──►  FastAPI (Python serverless function)
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                                     ▼
                Upstash Vector                              Groq
        (server-side embeddings + storage)        (llama3-70b-8192 answers)
```

- **Embeddings run server-side in Upstash Vector** — no `torch`/`sentence-transformers`,
  so the function bundle fits comfortably within Vercel's size limit.
- **Vectors persist in Upstash**, namespaced per document — no in-memory FAISS to lose
  between stateless invocations.
- **Groq** generates the final answer from the retrieved passages.

### Project layout

```
api/
  index.py            # Vercel entrypoint — exposes the FastAPI app
  server/             # backend package (single function, not split)
    main.py           # app, middleware, logging, exception handlers
    config.py         # env validation (fail-fast)
    rag.py            # ingest + retrieve + answer
    rate_limit.py     # optional Upstash Redis rate limiting
    routers/          # /api/documents, /api/chat
app/                  # Next.js App Router (landing page + /chat)
components/           # Dropzone, MessageBubble
lib/api.ts            # typed client for /api
requirements.txt      # lean Python deps (no torch)
vercel.json           # function maxDuration + includeFiles
next.config.ts        # /api dev proxy → uvicorn
```

## Prerequisites

1. **Groq API key** — https://console.groq.com/keys
2. **Upstash Vector index** — https://console.upstash.com/vector
   Create the index **with an embedding model selected** (e.g. `mixedbread` /
   `bge` — any built-in model). This is what lets the app send raw text and have
   Upstash embed it. Copy the **REST URL** and **REST token**.
3. *(Optional)* **Upstash Redis** for rate limiting — https://console.upstash.com/redis

## Local development

```bash
# 1. Backend deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Frontend deps
npm install

# 3. Env
cp .env.example .env   # fill in GROQ_API_KEY + UPSTASH_VECTOR_REST_*

# 4. Run both (two terminals)
npm run api            # FastAPI on http://127.0.0.1:8000
npm run dev            # Next.js on http://localhost:3000
```

Open http://localhost:3000. In dev, `next.config.ts` proxies `/api/*` to uvicorn.

Health check: `curl http://127.0.0.1:8000/api/health`
API docs: http://127.0.0.1:8000/api/docs

## Deploy to Vercel

Both frontend and backend ship as **one** Vercel project.

1. Import the repo into Vercel (framework preset: **Next.js**, auto-detected).
2. Add environment variables (Production **and** Preview):
   - `GROQ_API_KEY`
   - `UPSTASH_VECTOR_REST_URL`
   - `UPSTASH_VECTOR_REST_TOKEN`
   - *(optional)* `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`
3. Deploy. Vercel installs `requirements.txt`, builds `api/index.py` as a Python
   function, and serves both `/` and `/api/*` from one domain (same-origin, no CORS).

```bash
vercel          # preview
vercel --prod   # production
```

## Endpoints

| Method | Path             | Description                                  |
| ------ | ---------------- | -------------------------------------------- |
| POST   | `/api/documents` | Upload a PDF; returns `document_id`          |
| POST   | `/api/chat`      | `{document_id, question}` → grounded answer  |
| GET    | `/api/health`    | Liveness + config status                     |

## Known limits (this iteration)

- **Upload ≤ ~4 MB** — Vercel's serverless request-body limit. Larger files would
  use Vercel Blob client-upload (a documented future step).
- **Text-based PDFs only** — scanned/image-only PDFs have no extractable text (no OCR).
- **Non-streaming answers** — Vercel's Python runtime doesn't stream reliably, so the
  full answer returns in one response. Token streaming is a planned enhancement.

## Out of scope (follow-ups)

User accounts, persistent chat history, a multi-document library, token streaming,
large-file uploads via Vercel Blob.
