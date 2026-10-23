# Architecture

This document describes the architecture of **Quill** (AskPdf): a full-stack RAG application that lets authenticated users upload a PDF and ask grounded questions about it. It covers the tech stack, folder structure, data stores, and the end-to-end API/data flows.

> Complements [`README.md`](README.md), which focuses on setup and product features. This file focuses on how the system is built and how data moves through it.

---

## 1. System overview

Quill is a **three-tier, multi-process** application:

- **Client** — Next.js 16 (App Router) SPA/SSR frontend with Clerk auth.
- **API server** — Express 5 REST API (`/api/v1`) that authenticates requests, validates uploads, stores files, persists metadata, enqueues jobs, and serves chat.
- **Worker** — BullMQ worker process that downloads PDFs, extracts text, chunks it, generates embeddings, and stores vectors.
- **Data plane** — Supabase Postgres (with `pgvector`), Supabase Storage (private PDF bucket), Upstash Redis (BullMQ broker + rate-limit store).

AI providers: Google Gemini (embeddings + fallback chat) and Groq (primary chat).

```
Browser (Next.js)
   |
   | HTTPS + Clerk session token (Bearer JWT)
   v
Express API (/api/v1)  ---->  Supabase Storage (PDFs)
   |    |    \---->  Postgres (documents, chunks, messages)
   |    \---->  Upstash Redis (BullMQ queue + rate limits)
   |
   \-- enqueues "process-document" job
            |
            v
      BullMQ Worker  ---->  Supabase Storage (download PDF)
            |              ---->  pdf-parse (text)
            |              ---->  chunkText (chunks)
            |              ---->  Gemini Embedding 2 (vectors)
            \--------------->  Postgres (document_chunks.embedding vector(3072))
```

---

## 2. Tech stack

### Client (`client/`)

| Concern | Technology |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack) |
| UI | React 19, TypeScript |
| Styling | Tailwind CSS 4, PostCSS |
| Auth | `@clerk/nextjs` (ClerkProvider, middleware, `useAuth`) |
| Markdown | `react-markdown` |
| Icons | `lucide-react` |
| Lint | ESLint 9 (`eslint-config-next`) |

### Server (`server/`)

| Concern | Technology |
|---|---|
| Runtime | Node.js 20+ (ESM, `"type": "module"`) |
| Framework | Express 5 |
| ORM | Prisma 6 (`@prisma/client`), schema at `src/prisma/schema.prisma` |
| Database | PostgreSQL + `pgvector` (Supabase) |
| File upload | Multer (in-memory storage) |
| Object storage | `@supabase/supabase-js` (service-role client) |
| Queue | BullMQ 5 over `ioredis` |
| Redis | Upstash Redis (TLS) |
| Auth | `@clerk/express` (`clerkMiddleware`, `getAuth`) |
| Security | `helmet`, `cors` |
| Logging | `morgan`, custom `logger` |
| PDF parsing | `pdf-parse` (`PDFParse`) |
| AI | `@google/generative-ai` (Gemini), Groq via `fetch` (OpenAI-compatible) |
| Dev | `nodemon` |

### Infrastructure

| Component | Role |
|---|---|
| Supabase Postgres | Primary relational DB + `pgvector` (3072-dim) |
| Supabase Storage | Private `documents` bucket for raw PDFs |
| Upstash Redis | BullMQ broker + fixed-window rate limiter |
| Clerk | Identity provider / JWT session tokens |
| Google AI Studio | Gemini Embedding 2 + Gemini chat (fallback) |
| Groq | Primary chat model (`llama-3.3-70b-versatile`) |
| Docker Compose | Local Postgres 15 for development (`docker/`) |

---

## 3. Repository layout

```
Quill/
├── ARCHITECTURE.md
├── README.md
├── .gitignore
├── docker/
│   └── docker-compose.yml          # Local Postgres 15
├── client/                         # Next.js app
│   ├── app/
│   │   ├── page.tsx                # Public landing (redirects authed users to /chat)
│   │   ├── layout.tsx              # ClerkProvider + Navbar + fonts
│   │   ├── globals.css
│   │   ├── chat/page.tsx           # Protected workspace (server component, auth check)
│   │   ├── sign-in/page.tsx        # Clerk SignIn
│   │   └── sign-up/page.tsx        # Clerk SignUp
│   ├── components/
│   │   ├── LandingPage.tsx         # Marketing page
│   │   ├── Navbar.tsx              # Auth-aware nav
│   │   └── SplitPage.tsx           # Upload + chat workspace (client)
│   ├── lib/
│   │   └── api.ts                  # Shared fetch helpers
│   ├── proxy.ts                    # Clerk route protection (protects /chat)
│   ├── next.config.ts
│   ├── tsconfig.json
│   └── package.json
└── server/                         # Express API + worker
    ├── index.js                    # API process entry (also boots a worker instance)
    ├── worker.js                   # Worker-only process entry
    ├── package.json
    ├── .env.example
    └── src/
        ├── app.js                  # Express app factory (middleware + routes)
        ├── config/
        │   ├── env.js              # Env loading + validation + tunables
        │   ├── clerk.js            # Re-exports Clerk middleware
        │   ├── prisma.js           # PrismaClient singleton
        │   ├── redis.js            # ioredis connections (redis, bullConnection)
        │   └── supabase.js         # Service-role Supabase client
        ├── controllers/
        │   ├── upload.controller.js
        │   ├── document.controller.js
        │   └── chat.controller.js
        ├── middleware/
        │   ├── auth.middleware.js  # protectRoute (Clerk)
        │   ├── upload.middleware.js # Multer + logging
        │   ├── rateLimit.middleware.js
        │   └── error.middleware.js  # notFound + errorHandler
        ├── queues/
        │   └── documentQueue.js    # BullMQ Queue (producer)
        ├── routes/
        │   ├── index.routes.js     # /api/v1 router composition
        │   ├── health.routes.js    # GET /health
        │   ├── upload.routes.js    # POST /upload
        │   ├── document.routes.js  # /documents
        │   └── chat.routes.js      # POST /chat
        ├── services/
        │   ├── gemini.service.js   # embeddings + Groq/Gemini chat generation
        │   └── storage.service.js  # Supabase Storage upload/download/delete
        ├── utils/
        │   ├── chunkText.js        # sentence-aware chunking
        │   ├── errors.js           # ApiError
        │   ├── logger.js
        │   └── timed.js            # timing wrapper
        ├── workers/
        │   └── documentWorker.js   # BullMQ Worker (consumer)
        └── prisma/
            ├── schema.prisma
            └── migrations/
```

---

## 4. Runtime processes

There are two runnable entry points:

| Entry | Command | Purpose |
|---|---|---|
| `server/index.js` | `npm run dev` / `npm start` | Boots Express HTTP server **and** a BullMQ worker in the same process (guarded by a `globalThis` singleton). |
| `server/worker.js` | `npm run dev:worker` / `npm run start:worker` | Worker-only process. |

Both can run simultaneously; BullMQ distributes jobs across all consumers. The worker instance is deduplicated within a process via `Symbol.for('askpdf.documentWorker')` to avoid double registration under nodemon.

The API process also runs a worker by default, so a single `npm run dev` is enough for local development; a dedicated worker is recommended for production.

---

## 5. Data stores

### 5.1 PostgreSQL (Supabase) — primary database

Prisma datasource uses `DATABASE_URL` (transaction pooler) for runtime and `DIRECT_URL` (session pooler) for migrations. Extension `vector` is enabled.

#### `documents`

| Column | Type | Notes |
|---|---|---|
| `id` | uuid (PK) | generated client-side (`randomUUID`) on upload |
| `user_id` | text | Clerk `userId`; scoped on every query |
| `original_name` | text | uploaded filename |
| `file_name` | text | `{documentId}.pdf` |
| `file_path` | text | Supabase Storage path `{userId}/{documentId}.pdf` |
| `file_size` | int | bytes |
| `status` | enum | `UPLOADED` → `PROCESSING` → `PROCESSED` / `FAILED` |
| `error_message` | text (nullable) | set on failure |
| `created_at` | timestamptz | |
| `updated_at` | timestamptz | |

Index: `(user_id, created_at)`.

#### `document_chunks`

| Column | Type | Notes |
|---|---|---|
| `id` | uuid (PK) | |
| `document_id` | uuid (FK → documents, cascade) | |
| `chunk_index` | int | ordinal within document |
| `content` | text | chunk text |
| `embedding` | `vector(3072)` (nullable, `Unsupported`) | written via raw SQL; HNSW not used (pgvector HNSW cap = 2000 dims) |
| `content_tsv` | `tsvector` (nullable, `Unsupported`) | generated column: `to_tsvector('english', content)`; used for full-text keyword search |
| `created_at` | timestamptz | |

Index: `(document_id)`, GIN index on `(content_tsv)`.

#### `chat_messages`

| Column | Type | Notes |
|---|---|---|
| `id` | uuid (PK) | |
| `document_id` | uuid (FK → documents, cascade) | |
| `user_id` | text | Clerk `userId` |
| `role` | enum | `USER` / `ASSISTANT` |
| `content` | text | |
| `created_at` | timestamptz | |

Index: `(document_id, created_at)`.

Cascade deletes remove chunks and messages when a document is deleted.

### 5.2 Redis (Upstash) — broker + rate limiter

Two `ioredis` connections (TLS, `maxRetriesPerRequest: null`, `enableReadyCheck: false`):

- `redis` — general commands (health, rate limiting).
- `bullConnection` — dedicated BullMQ connection (blocking mode).

Used for:

- BullMQ queue `document-queue` / job `process-document`.
- Fixed-window rate limiter keys `rl:{prefix}:{identity}` (INCR + EXPIRE via Lua).
- Health probe (`PING`).

### 5.3 Supabase Storage — object store

- Private bucket `documents` (default name, configurable via `SUPABASE_STORAGE_BUCKET`).
- Object key: `{userId}/{documentId}.pdf`.
- Server uses service-role key (bypasses RLS); ownership is enforced in application code by scoping paths/queries to `userId`.
- Operations: `uploadPdf`, `downloadPdf`, `deletePdf`.

---

## 6. Authentication & authorization

- **Provider:** Clerk.
- **Client:** `ClerkProvider` in `app/layout.tsx`; `proxy.ts` (Clerk middleware) protects `/chat(.*)` via `auth.protect()`. Client obtains a session token with `useAuth().getToken()` and sends `Authorization: Bearer <token>`.
- **Server:** `clerkMiddleware` (from `@clerk/express`) runs globally in `src/app.js` and populates `req.auth`. `protectRoute` (`src/middleware/auth.middleware.js`) rejects requests with no `auth.userId` with `401`.
- **Scoping:** Every DB query filters by `userId` (documents, chat messages). Storage paths are prefixed with `userId`. There is no cross-user access path.

Unauthenticated API requests → `401 Unauthorized`.

---

## 7. HTTP API

Base path: `/api/v1`. All routes except `/health` (and `/me` requires auth) are protected.

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | public | Readiness probe for Postgres + Redis (200/503) |
| GET | `/me` | bearer | Returns `{ userId }` |
| POST | `/upload` | bearer | Uploads a PDF (multipart `file`), queues processing; 201 |
| GET | `/documents` | bearer | List caller's documents |
| GET | `/documents/:id` | bearer | Status, error, chunk/message counts |
| GET | `/documents/:id/messages` | bearer | Chat history (asc, max 200) |
| DELETE | `/documents/:id` | bearer | Delete doc + cascade + storage cleanup |
| POST | `/chat` | bearer | Retrieve context + generate grounded answer |

### Middleware pipeline (per request)

`helmet` → `cors` (preflight handled before auth) → `morgan` → `clerkMiddleware` → `express.json({limit:1mb})` → router → `notFoundHandler` / `errorHandler`.

Route-level: `protectRoute`, `rateLimit` (upload: 10/hour, chat: 20/min, per user), `uploadPdfFile` (Multer, in-memory, 20 MB, PDF mimetype), controller.

Errors are normalized by `errorHandler`: Multer errors → 400 with size message; `ApiError` carries status; 5xx hide message in prod.

---

## 8. API flows

### 8.1 Upload & processing flow

1. Client selects/drops a PDF → `POST /api/v1/upload` with multipart `file` + Bearer token.
2. `uploadPdfFile` (Multer) parses into memory; validates mimetype `application/pdf` and size ≤ `MAX_FILE_SIZE_MB` (default 20).
3. `uploadFile` controller:
   - validates buffer non-empty and PDF magic bytes `%PDF-` (content-based, not just mimetype).
   - `userId = req.auth.userId`, `documentId = randomUUID()`, `storagePath = {userId}/{documentId}.pdf`.
   - `uploadPdf(storagePath, buffer)` → Supabase Storage.
   - `prisma.document.create` with status default `UPLOADED`.
   - `enqueueDocumentJob({documentId, userId, storagePath})` → BullMQ (3 attempts, exponential backoff 15s).
   - returns `201 { id, status: 'UPLOADED', originalName, createdAt }`.
   - On queue failure: marks document `FAILED`, deletes stored PDF, returns `503`.
   - On any later failure: deletes stored PDF (best-effort), rethrows.
4. Client polls `GET /documents/:id` every 2s until `PROCESSED`/`FAILED`, updating UI status:
   - `UPLOADED` → "Uploaded - preparing document"
   - `PROCESSING` → "Creating and storing embeddings"
   - `PROCESSED` → "Ready for questions"
   - `FAILED` → error message.
5. Chat input stays disabled unless status is `PROCESSED`.

### 8.2 Background processing flow (worker)

`processDocument(job)`:

1. Mark document `PROCESSING`, clear `errorMessage`.
2. Download PDF from Supabase Storage → `Buffer`.
3. `pdf-parse` extracts text; reject if < 20 non-whitespace chars (scanned/image PDFs).
4. `chunkText(text)` → sentence-aware chunks (max 1200 chars, 200 overlap); hard-split oversized sentences; enforce `MAX_CHUNKS_PER_DOCUMENT` (default 4000).
5. If retry (`attemptsMade > 0`), delete existing chunks for the document (avoid duplicates).
6. Persist chunks (`documentChunk.createMany`, batches of 200) **concurrently** with embedding generation (`embedTexts(chunks)` in batches of 100).
7. `storeEmbeddings`: batched `UPDATE document_chunks SET embedding = ...::vector` via `jsonb_each_text`.
8. Mark document `PROCESSED`.
9. On final failure: worker `failed` handler sets document `FAILED` with truncated error message.

Progress reported via `job.updateProgress` (30/55/80/100).

### 8.3 Chat (RAG) flow

`POST /api/v1/chat` with `{ documentId, message }`:

1. Validate inputs; enforce `MAX_QUESTION_LENGTH` (4000).
2. Ownership + readiness check: `document.findFirst({id, userId})`; 404 if missing; 409 if not `PROCESSED` (or `FAILED` with message).
3. In parallel:
   - `embedTexts([message])` → question vector (pgvector literal).
   - fetch last 10 chat messages (desc) for history.
4. Hybrid search (vector + full-text, fused with RRF):
   - **Vector search** — cosine distance over `embedding` (3072-dim Gemini vectors):
     ```sql
     SELECT id, content, 1 - (embedding <=> $q::vector) AS similarity
     FROM document_chunks
     WHERE document_id = $id AND embedding IS NOT NULL
     ORDER BY embedding <=> $q::vector
     LIMIT $topK * 2
     ```
   - **Keyword search** — PostgreSQL full-text search on `content_tsv` (GIN index):
     ```sql
     SELECT id, content, ts_rank(content_tsv, plainto_tsquery('english', $q)) AS rank
     FROM document_chunks
     WHERE document_id = $id AND content_tsv @@ plainto_tsquery('english', $q)
     ORDER BY rank DESC
     LIMIT $topK * 2
     ```
   - **RRF fusion** — `score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)` (default `k=60`, `TOP_K_CHUNKS=8`). Items appearing in both lists are boosted. Falls back to vector-only when `HYBRID_SEARCH_ENABLED=false`.
5. If no chunks → 503.
6. `generateAnswer({documentTitle, question, contextChunks, history})`:
   - Build numbered context `[1]..[n]`.
   - Try Groq (with retry up to 3) using `GROQ_CHAT_MODEL`; on failure or if no key, fall back to Gemini chat model.
   - System prompt constrains answers to provided context and numbered citations only.
7. Persist `USER` and `ASSISTANT` messages in a transaction.
8. Return `{ messageId, documentId, answer, sources: [{id, similarity, snippet}] }` (top 4 sources).

### 8.4 Document management flow

- `GET /documents` — list caller's docs with chunk counts.
- `GET /documents/:id` — detail + chunk/message counts.
- `GET /documents/:id/messages` — history (asc, cap 200).
- `DELETE /documents/:id` — delete DB row (cascade removes chunks/messages), then best-effort delete PDF from storage.

### 8.5 Health flow

`GET /api/v1/health` runs `SELECT 1` (Postgres) and `redis.ping()` with 3s timeouts; returns 200 `{status:'ok', dependencies:{postgres,redis}}` or 503 `{status:'degraded',...}`.

---

## 9. Embeddings & retrieval details

- **Model:** Gemini Embedding 2 (`gemini-embedding-2`), producing **3072-dimensional** vectors.
- **Storage format:** pgvector literal string `"[v1,v2,...]"` written via raw SQL (Prisma `Unsupported("vector(3072)")`).
- **Batching:** embedding API calls batched at 100 texts; worker inserts chunks in batches of 200.
- **Retrieval — hybrid:** vector search (cosine distance `<=>`) **+** full-text search (PostgreSQL `tsvector` + GIN index) fused with **Reciprocal Rank Fusion (RRF)**.
  - `score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)`, default `k=60` (`RRF_K`).
  - Fetches `TOP_K_CHUNKS * 2` from each list, fuses, returns top `TOP_K_CHUNKS` (default 8).
  - Controlled by `HYBRID_SEARCH_ENABLED` (default `true`); set to `false` for vector-only.
- **Exact vector search:** HNSW not used because pgvector HNSW max is 2000 dims; revisit with IVFFlat or model change at scale.
- **Compatibility constraint:** embedding model and `vector(3072)` column must stay in sync; changing models requires regenerating all embeddings and altering the column.
- **Question embedding** uses the same model to keep vector spaces aligned.

---

## 10. Chat generation & model fallback

- **Primary:** Groq `llama-3.3-70b-versatile` (OpenAI-compatible `/v1/chat/completions`), retried up to 3 on retryable errors (429/quota/5xx/network) with exponential backoff.
- **Fallback:** Gemini chat model (`gemini-2.0-flash` default), also retried.
- **Prompting:** system prompt restricts answers to provided context; citations only as numbered labels `[1]`, `[2]`; no invented page/line numbers/URLs. History limited to last 10 messages. Temperature 0.2, max 2048 tokens.
- **Persistence:** both user question and assistant answer stored; history returned chronologically.

---

## 11. Client architecture

- **Routing (App Router):**
  - `/` — `page.tsx` server component: redirects authenticated users to `/chat`, else renders `LandingPage`.
  - `/chat` — server component checks `auth()`; redirects unauthed to `/sign-in`; renders `SplitPage`.
  - `/sign-in`, `/sign-up` — Clerk components.
- **Protection:** `proxy.ts` Clerk middleware protects `/chat(.*)`.
- **Workspace (`SplitPage.tsx`):** client component managing upload, status polling (2s), and chat. Uses `useAuth().getToken()` for Bearer tokens. Calls `NEXT_PUBLIC_API_URL/api/v1/...`.
- **Shared helpers:** `lib/api.ts` (`apiRequest`, `uploadFile`, `sendMessage`) — note these append endpoints without `/api/v1` in some paths; the workspace uses `/api/v1` directly.
- **Rendering:** assistant messages rendered with `react-markdown`.

---

## 12. Configuration & environment

Server required env (validated at boot in `src/config/env.js`, missing → exit 1):

`DATABASE_URL`, `DIRECT_URL`, `UPSTASH_REDIS_URL`, `GEMINI_API_KEY`, `CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.

Optional with defaults: `PORT` (5000), `CLIENT_URL` (http://localhost:3000), `SUPABASE_STORAGE_BUCKET` (documents), `GEMINI_EMBEDDING_MODEL` (gemini-embedding-2), `GEMINI_CHAT_MODEL` (gemini-2.0-flash), `GROQ_API_KEY`, `GROQ_CHAT_MODEL` (llama-3.3-70b-versatile), plus tunables (`MAX_FILE_SIZE_MB=20`, `MAX_CHUNKS_PER_DOCUMENT=4000`, `WORKER_CONCURRENCY=2`, `TOP_K_CHUNKS=8`, `HYBRID_SEARCH_ENABLED=true`, `RRF_K=60`, `RATE_LIMIT_UPLOAD_PER_HOUR=10`, `RATE_LIMIT_CHAT_PER_MIN=20`).

Client env: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`. Client appends `/api/v1` to base URL in workspace calls.

---

## 13. Local development

1. Start Postgres (optional, for local): `docker-compose up` in `docker/` (or use Supabase).
2. `cd server && npm install && npm run db:migrate && npm run db:generate`.
3. Create `server/.env` from `.env.example`.
4. `cd server && npm run dev` (API + worker).
5. Optionally `npm run dev:worker` in another terminal.
6. `cd client && npm install && npm run dev` (port 3000).

---

## 14. Key design decisions

- **Async ingestion:** uploads return immediately (201) and heavy PDF processing runs in BullMQ; UI polls status. Keeps API responsive.
- **Hybrid retrieval:** vector search (semantic) + full-text search (lexical) fused with RRF. Keyword search catches exact terms/acronyms that embeddings miss; RRF boosts chunks ranked highly by both.
- **Exact vector search:** 3072-dim embeddings exceed pgvector HNSW's 2000-dim limit, so exact scan is used (suitable for MVP scale).
- **Groq-first with Gemini fallback:** optimizes cost/latency while retaining availability.
- **Per-user isolation:** all queries and storage paths scoped by Clerk `userId`.
- **In-memory uploads:** Multer memory storage avoids local disk; files stream straight to Supabase.
- **Idempotent retries:** worker clears prior chunks on retry to avoid duplicates.
- **Fails-open rate limiter:** limiter outages don't take down the API.
- **Best-effort storage cleanup:** DB is source of truth; storage deletes don't block document deletion.

---

## 15. Evaluation results (offline, 50-question gold set)

| Metric | Vector-only | Hybrid | Change |
|---|---|---|---|
| hit@1 | 0.540 | 0.640 | +18.5% |
| hit@3 | 0.800 | 0.860 | +7.5% |
| hit@8 | 0.920 | 0.920 | 0.0% |
| MRR@8 | 0.687 | 0.753 | +9.6% |

Hybrid search = vector search + PostgreSQL full-text search (tsvector + GIN) fused with Reciprocal Rank Fusion (RRF, k=60). Full results in `evals/retrieval_compare_results.json`.

## 16. Known gaps / notes

- `client/lib/api.ts` helpers call `/upload` and `/chat` without `/api/v1` in some paths; the active workspace (`SplitPage`) uses `/api/v1/...`. Align if using helpers.
- Embedding column is `Unsupported` in Prisma; all vector access is raw SQL.
- Automated tests are not yet implemented (recommended before production).
- Changing embedding model requires regenerating all embeddings and altering `vector(3072)`.
- Worker runs inside the API process by default; for production, run `start:worker` separately and consider disabling in-process worker.
- A neural cross-encoder re-ranker (after RRF) was considered but not implemented — could further improve top-of-list precision at the cost of latency.