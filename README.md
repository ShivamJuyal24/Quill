# 📄 AI PDF Chat Application (AskPdf)

A production-oriented full-stack application for asking questions about PDF documents. AskPdf combines Clerk authentication, Supabase Storage, PostgreSQL with pgvector, Redis-backed background jobs, Gemini embeddings, and Groq/Gemini chat generation in a clean RAG architecture.

The application has a public landing page, authenticated document-chat workspace, asynchronous PDF processing, semantic retrieval, grounded answers, and a Groq-first chat fallback to Gemini.

## 🚀 Project Status

- ✅ **Phase 1** – Core API & Authentication
- ✅ **Phase 2** – Background Queue (BullMQ + Redis)
- ✅ **Phase 3** – PDF Parsing & Text Chunking
- ✅ **Phase 4** – Database & Storage Integration
- ✅ **Phase 5** – Vector Embeddings with pgvector
- ✅ **Phase 6** – Chat with PDF and Semantic Retrieval
- ✅ **Phase 7** – Landing Page, Protected Workspace & Processing Status UX
- ✅ **Phase 8** – Production Deployment, Monitoring & Automated Tests

## 🧠 What This Project Demonstrates

- Public product landing page with Clerk sign-in and sign-up flows
- Protected document-chat workspace for authenticated users
- JWT-based authentication using Clerk on both client and server
- PDF upload validation using Multer and PDF magic bytes
- Private document storage using Supabase Storage
- Asynchronous document processing using BullMQ and Redis
- PDF text extraction and sentence-aware chunking
- 3072-dimensional Gemini embeddings stored with pgvector
- Hybrid retrieval: vector (semantic) + full-text (lexical) search fused with Reciprocal Rank Fusion (RRF)
- Grounded RAG answers with numbered context citations
- Groq-first chat generation with Gemini fallback
- Per-user document and chat isolation
- Processing status polling in the frontend
- Markdown-rendered assistant responses
- Structured API, worker, service, queue, and utility layers

## 🏗️ Architecture

```
Public visitor
   │
   ├── Landing page (/)
   ├── Clerk sign-in (/sign-in)
   └── Clerk sign-up (/sign-up)
            │
            ▼
Authenticated user
   │
   └── Document workspace (/chat)
          │
          ├── Upload PDF
          │      │
          │      ├── Express API
          │      ├── Clerk authentication
          │      ├── Multer validation
          │      ├── Supabase Storage
          │      ├── PostgreSQL document record
          │      └── BullMQ document job
          │
          └── Ask question
                 │
                 ├── Gemini question embedding
                 ├── PostgreSQL exact vector search
                 ├── Top-K document chunks
                 ├── Groq chat generation
                 ├── Gemini fallback if Groq fails
                 └── Persisted chat history

Upstash Redis ──► BullMQ ──► Document Worker
                                  │
                                  ├── Download PDF
                                  ├── Extract text
                                  ├── Chunk text
                                  ├── Generate embeddings
                                  └── Store vectors
```

## 🏗️ Tech Stack

### Client

- Next.js 16 with App Router and Turbopack
- React 19
- TypeScript
- Tailwind CSS
- Clerk Next.js SDK
- React Markdown
- Lucide React

### Server

- Node.js 20+
- Express 5
- Prisma
- PostgreSQL with pgvector
- Multer
- Supabase Storage
- Upstash Redis
- BullMQ

### Authentication

- Clerk session authentication
- Clerk middleware on the Next.js client
- Clerk request authentication on the Express API
- Per-user document, chunk, and chat ownership checks

### AI

- Gemini Embedding 2 for document and question embeddings
- Groq `llama-3.3-70b-versatile` as the primary chat model
- Gemini chat model as the fallback provider

## 📁 Project Structure

```
client/
├── app/
│   ├── chat/page.tsx              # Protected document workspace route
│   ├── sign-in/page.tsx           # Clerk sign-in route
│   ├── sign-up/page.tsx           # Clerk sign-up route
│   ├── page.tsx                   # Public landing route
│   ├── layout.tsx                 # Clerk provider and global layout
│   └── globals.css
├── components/
│   ├── LandingPage.tsx            # Public product page
│   ├── Navbar.tsx                 # Navigation and auth controls
│   └── SplitPage.tsx              # Upload and chat workspace
├── lib/api.ts                     # Shared API helpers
└── proxy.ts                       # Clerk route protection

server/
├── index.js                       # Express API process
├── worker.js                      # BullMQ worker process
└── src/
    ├── config/                    # Environment, Clerk, Prisma, Redis, Supabase
    ├── controllers/               # Upload, document, and chat handlers
    ├── middleware/                # Auth, upload, rate limit, and errors
    ├── queues/                   # BullMQ producers
    ├── routes/                   # API route definitions
    ├── services/                 # Gemini, Groq, and storage integrations
    ├── utils/                    # Logging, errors, and chunking
    ├── workers/                  # PDF processing worker
    └── prisma/                   # Schema and migrations

docker/
└── docker-compose.yml             # Local PostgreSQL development service
```

## 🔐 Authentication Flow (Clerk)

1. A visitor opens the public landing page.
2. Clerk handles sign-in or sign-up.
3. Authenticated users are redirected to `/chat`.
4. The client obtains a Clerk session token with `getToken()`.
5. The token is sent as `Authorization: Bearer <token>` to the API.
6. Express verifies the request through Clerk middleware.
7. The authenticated Clerk `userId` scopes documents and conversations.

❌ Unauthenticated API requests return `401 Unauthorized`.

## 📤 File Upload & Processing Flow

1. An authenticated user selects or drops a PDF.
2. The client uploads it as multipart field `file`.
3. The API validates the MIME type, file size, and PDF magic bytes.
4. The PDF is uploaded to a private Supabase Storage bucket.
5. A document row is created in PostgreSQL with status `UPLOADED`.
6. A BullMQ job is added to Redis.
7. The API returns immediately with `201 Created`.
8. The client polls the document status endpoint every two seconds.
9. The UI shows the processing lifecycle:

```
Uploading
   ↓
Uploaded - preparing document
   ↓
Creating and storing embeddings
   ↓
Ready for questions
```

The chat input remains disabled until the document reaches `PROCESSED`. Failed processing is surfaced in the workspace with the stored error message.

## ⚙️ Background Processing (BullMQ + Redis)

- Redis acts as the job broker and rate-limit store.
- BullMQ manages retries and exponential backoff.
- The API and worker run as separate processes.
- The worker downloads PDFs from Supabase Storage.
- Extracted text is split into sentence-aware chunks.
- Embeddings are generated in batches.
- Failed jobs update the document to `FAILED` with an error message.
- Retried jobs clear previous chunks before reprocessing to avoid duplicates.

### Example Worker Logs

```
[job 12 | doc ...] started (attempt 1/3)
[job 12 | doc ...] status set to PROCESSING
[job 12 | doc ...] PDF downloaded; extracting text
[job 12 | doc ...] extracted 3733 chars -> 4 chunks
[job 12 | doc ...] generating embeddings for 4 chunks
[job 12 | doc ...] embeddings stored successfully
[job 12 | doc ...] completed successfully; status set to PROCESSED
```

✔ Upload API remains responsive  
✔ Heavy processing runs asynchronously  
✔ Users cannot chat with incomplete embeddings

## 🧠 PDF Processing & Embeddings

1. The worker downloads the PDF from private storage.
2. `pdf-parse` extracts readable text.
3. `chunkText` creates overlapping, sentence-aware chunks.
4. Gemini Embedding 2 generates one 3072-dimensional vector per chunk.
5. Vectors are stored in the PostgreSQL `vector(3072)` column.
6. Chat questions use the same embedding model.
7. PostgreSQL retrieves the most similar chunks using cosine distance.

HNSW is not used because pgvector limits HNSW indexes to 2000 dimensions. The current implementation uses exact vector search, which is appropriate for the current MVP scale but should be revisited as the document corpus grows.

⚠️ Embedding models and vector dimensions must remain compatible. If the embedding model changes, all stored document embeddings must be regenerated and the database column must match the new dimension.

## 💬 Chat & Model Fallback

The chat flow retrieves relevant chunks and sends only those excerpts, the question, and recent conversation history to the generation model.

1. Groq is tried first using the configured chat model.
2. Groq retries temporary failures up to three times.
3. Gemini is used when Groq is unavailable or fails.
4. The answer is constrained to the retrieved document context.
5. The assistant response is stored in chat history and rendered as Markdown.

Supported citations are numbered context labels such as `[1]` and `[2]`. The prompt explicitly prevents invented page numbers, line numbers, URLs, or unsupported citation formats.

## 🗄️ Database Schema

### `documents`

Stores uploaded document metadata and processing state.

- `id`
- `user_id`
- `original_name`
- `file_name`
- `file_path`
- `file_size`
- `status` (`UPLOADED`, `PROCESSING`, `PROCESSED`, `FAILED`)
- `error_message`
- `created_at`
- `updated_at`

### `document_chunks`

Stores extracted text and vector embeddings.

- `id`
- `document_id`
- `chunk_index`
- `content`
- `embedding` (`vector(3072)`)
- `created_at`

### `chat_messages`

Stores the conversation for each owned document.

- `id`
- `document_id`
- `user_id`
- `role` (`USER` or `ASSISTANT`)
- `content`
- `created_at`

## 🔌 API

Base URL: `/api/v1`. All routes except `/health` require a Clerk bearer token.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Public readiness probe for PostgreSQL, Redis, BullMQ, Supabase Storage, Gemini, and Groq |
| GET | `/me` | Returns the authenticated Clerk user ID |
| POST | `/upload` | Uploads a PDF and queues processing |
| GET | `/documents` | Lists the authenticated user's documents |
| GET | `/documents/:id` | Returns status, errors, and chunk/message counts |
| GET | `/documents/:id/messages` | Returns document chat history |
| DELETE | `/documents/:id` | Deletes the document, chunks, messages, and stored PDF |
| POST | `/chat` | Retrieves context and generates a grounded answer |

### Upload

Multipart field:

```text
file=<PDF file>
```

Default limit: 20 MB. The API accepts PDF files only and verifies the `%PDF-` file signature.

### Chat

```json
{
  "documentId": "document-id",
  "message": "What does this document say about Docker?"
}
```

The document must be `PROCESSED` before chat is available.

## ⚙️ Environment Variables

Create `server/.env` from [server/.env.example](server/.env.example).

```env
NODE_ENV=development
PORT=5000
CLIENT_URL=http://localhost:3000

DATABASE_URL=postgresql://...
DIRECT_URL=postgresql://...

SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
SUPABASE_STORAGE_BUCKET=documents

UPSTASH_REDIS_URL=rediss://...

GEMINI_API_KEY=<api-key>
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_CHAT_MODEL=gemini-3.5-flash

GROQ_API_KEY=<api-key>
GROQ_CHAT_MODEL=llama-3.3-70b-versatile

CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
```

The client requires:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
CLERK_SECRET_KEY=sk_...
```

The client appends `/api/v1` to `NEXT_PUBLIC_API_URL`. Keep secret keys out of browser-exposed variables and never commit `.env` files.

## 🛠️ Local Setup

### 1. Install dependencies

```bash
cd server
npm install

cd ../client
npm install
```

### 2. Configure external services

Create or connect the following services:

- Supabase project with PostgreSQL and a private Storage bucket named `documents`
- Upstash Redis database
- Clerk application
- Google AI Studio API key
- Groq API key for chat fallback

### 3. Apply database migrations

```bash
cd server
npm run db:migrate
npm run db:generate
```

### 4. Start the API

```bash
cd server
npm run dev
```

The API runs on `http://localhost:5000`.

### 5. Start the document worker

Use a second terminal:

```bash
cd server
npm run dev:worker
```

### 6. Start the client

Use a third terminal:

```bash
cd client
npm run dev
```

The client runs on `http://localhost:3000`.

## 🚢 Production Deployment

Deploy the client, API, and worker as separate services.

### Client service

```bash
npm run build
npm start
```

Required variables:

```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
CLERK_SECRET_KEY=sk_live_...
```

### API service

```bash
npm run db:migrate
npm start
```

Set `CLIENT_URL` to the deployed client origin and configure every server environment variable in the hosting provider. Do not place secrets in source control.

### Worker service

```bash
npm run start:worker
```

The API and worker must share the same PostgreSQL, Redis, Supabase, Gemini, Groq, and Clerk configuration.

### Production checklist

- Rotate credentials that were exposed during development.
- Use production Clerk keys from the same Clerk instance.
- Configure the production client origin in `CLIENT_URL`.
- Create the private `documents` Storage bucket.
- Apply Prisma migrations before accepting traffic.
- Verify `/api/v1/health` returns `200`.
- Test upload, processing, chat, deletion, and Groq-to-Gemini fallback.
- Configure logs and alerts for worker failures.
- Run dependency and vulnerability audits.

## 🧪 Validation

Client checks:

```bash
cd client
npm run lint
npm run build
```

Server checks:

```bash
cd server
npx prisma validate
node --check index.js
node --check src/services/gemini.service.js
node --check src/workers/documentWorker.js
```

Automated tests for authentication, ownership isolation, upload validation, worker retries, and the complete upload-to-chat flow are still recommended before production launch.

## 📊 Evaluation

Retrieval and answer quality were measured offline on a gold dataset of **50 questions** with known relevant chunks (`evals/rag_eval.json`), using Gemini `gemini-embedding-2` (3072-dim) embeddings and Groq `openai/gpt-oss-120b` for answer generation. Last evaluated: **2026-10-25**.

### Retrieval: vector-only vs hybrid (vector + keyword + RRF)

| Metric | Vector-only (before) | Hybrid (after) | Hybrid + Rerank (top-4) | Change |
|---|---|---|---|---|
| **hit@1** | 0.540 | **0.640** | **0.820** | **+18.5%** |
| **hit@3** | 0.800 | **0.860** | **0.940** | +7.5% |
| **hit@5** | 0.900 | **0.920** | **0.940** | +2.2% |
| **hit@8** | 0.920 | **0.920** | **0.940** | 0.0% |
| **MRR@8** | 0.687 | **0.753** | **0.877** | **+9.6%** |
| **recall@8** | 0.862 | 0.853 | 0.858 | -1.0% |
| **precision@8** | 0.172 | 0.170 | 0.165 | -1.4% |

Hybrid search = vector search (cosine similarity over `embedding`) **+** full-text keyword search (PostgreSQL `tsvector` + GIN index) fused with **Reciprocal Rank Fusion (RRF)**. The most relevant chunk is far more likely to appear at rank 1 (hit@1 +18.5%), and overall ranking quality improves (MRR +9.6%), with no regression on hit@8.

### Retrieval: + Cohere rerank (top-4)

| Metric | Hybrid (before) | Hybrid + Rerank (after) | Change |
|---|---|---|---|
| **hit@1** | 0.640 | **0.820** | **+28.1%** |
| **hit@3** | 0.860 | **0.940** | +9.3% |
| **hit@5** | 0.920 | **0.940** | +2.2% |
| **hit@8** | 0.920 | **0.940** | +2.2% |
| **MRR@8** | 0.753 | **0.877** | **+16.4%** |
| **recall@8** | 0.853 | 0.858 | +0.6% |

After RRF fusion, the top candidates are re-scored by a Cohere cross-encoder reranker (`rerank-english-v3.0`), returning the top-4 chunks. This lifts MRR@8 from 0.753 to 0.877 (+16.4%) and hit@1 from 0.640 to 0.820. Reranking is applied in `chat.controller.js` via `reranker.service.js`; if `COHERE_API_KEY` is unset or the call fails, it falls back to the fused order.

Hybrid search is enabled by default (`HYBRID_SEARCH_ENABLED=true`, `RRF_K=60`). Set `HYBRID_SEARCH_ENABLED=false` for vector-only.

### Answer quality: Ragas

Answer quality was scored with **Ragas 0.4.3** on the same 50-question dataset at `top_k=8`, using Gemini `gemini-embedding-2` embeddings and Groq `openai/gpt-oss-120b` for generation and judging. All 50 answers were generated by Groq; no Gemini fallback was used.

| Metric | Mean (0–1) | Mean (%) | Median | Min | Max | Valid n |
|---|---:|---:|---:|---:|---:|---:|
| **Faithfulness** | 0.9162 | **91.62%** | 1.0000 | 0.0000 | 1.0000 | 50 |
| **Answer relevancy** | 0.8755 | **87.55%** | 0.9405 | 0.0000 | 1.0000 | 50 |
| **Context precision** | 0.7019 | **70.19%** | 0.6833 | 0.1250 | 1.0000 | 50 |
| **Context recall** | 0.8733 | **87.33%** | 1.0000 | 0.0000 | 1.0000 | 50 |
| **Answer correctness** | 0.7277 | **72.77%** | 0.7382 | 0.2252 | 0.9812 | 50 |

All five metrics have **50/50 valid scores**. The mean of the five metric averages is **0.8189 (81.89%)**; this is an informal roll-up, not an official Ragas metric.

> The dataset contains **47 answerable and 3 unanswerable** questions (Q48–Q50). Ragas scores noncommittal answers at zero relevancy, so the all-50 averages above are conservative. On the 47 answerable questions, the means are: faithfulness 96.40%, answer relevancy 93.14%, context precision 72.01%, context recall 92.91%, and answer correctness 73.93%.

### How to run the evaluations

```bash
cd evals
pip install -r requirements.txt
python retrieval_compare.py        # vector-only vs hybrid comparison
python ragas_eval.py --skip-ragas  # generate answers + retrieval metrics
python ragas_score.py              # score answers with Ragas
python llm_judge_eval.py           # optional LLM-as-judge (separate output)
```

Results are written to `evals/retrieval_compare_results.json`, `evals/ragas_results.json`, `evals/ragas_metrics_results.json`, and optionally `evals/llm_judge_results.json`.

## 🛑 Important Notes

- `.env` files are ignored by Git and must never be committed.
- Supabase service-role keys are server-only secrets.
- Clerk publishable keys may be public; Clerk secret keys must remain server-side.
- Uploaded PDFs are stored in a private Supabase Storage bucket.
- Every storage path and database query is scoped to the authenticated user.
- Scanned/image-only PDFs are rejected because they contain no extractable text.
- The worker must be running for uploaded documents to become chat-ready.
- A document can be deleted only by its owner.
- The current 3072-dimensional vector search is exact and may need an indexing strategy at larger scale.

## 📌 Summary

AskPdf is a complete full-stack RAG application for private PDF question answering. It now includes a public landing experience, Clerk-protected workspace, asynchronous document ingestion, 3072-dimensional Gemini embeddings, PostgreSQL retrieval, Groq-first chat generation with Gemini fallback, and clear processing states from upload through readiness.
