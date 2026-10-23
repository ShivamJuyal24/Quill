# Development Rules

Rules that must be followed for every change in this repository. These exist to keep the codebase stable, minimal, and production-ready.

## 1. Dependencies

- **No unnecessary dependencies.** Do not add a package unless it is required to implement the feature and no existing dependency covers it. Prefer the standard library / already-installed packages.
- Before adding a dependency, check `package.json` and `node_modules` for an existing equivalent.
- Remove unused imports/dependencies introduced by a change.
- Pin versions consistent with the existing stack (Node 20+, Express 5, Next.js 16, Prisma 6, React 19). Do not introduce major-version upgrades without explicit approval.
- Never commit `node_modules/`, `.env`, build artifacts, or `.next/` (see `.gitignore`).

## 2. API stability

- **Do not break existing APIs.** Preserve the `/api/v1` route paths, request/response shapes, and status codes listed in `ARCHITECTURE.md` unless the user explicitly requests a breaking change.
- Additive changes (new optional fields, new endpoints) are preferred over modifications to existing contracts.
- If a breaking change is unavoidable, document it in the PR/commit and update clients in the same change.
- Keep error responses in the `{ error: string }` shape and appropriate HTTP status codes.
- Maintain Clerk Bearer-token auth and per-user scoping on every protected route.

## 3. Testing & validation

- **Test after every change.** Run the relevant checks before considering work done:
  - Client: `cd client && npm run lint && npm run build`.
  - Server: `cd server && npx prisma validate && node --check index.js` plus `node --check` on any touched file under `src/`.
  - Database: run migrations (`npm run db:migrate` / `db:migrate:dev`) and regenerate the client (`npm run db:generate`) when schema changes.
- Manually exercise the affected flow (upload → processing → chat) when changing ingestion or retrieval.
- Do not mark a task complete if checks fail; fix or report the failure.

## 4. Architecture & data integrity

- **Vector dimension compatibility:** embeddings must remain 3072-dimensional and match the `vector(3072)` column and `GEMINI_EMBEDDING_MODEL`. Changing models requires regenerating all embeddings and altering the column.
- **pgvector constraints:** do not add HNSW indexes (max 2000 dims); keep exact search unless a compatible indexing strategy is introduced.
- **Ownership scoping:** every document/chunk/message query and storage path must be scoped to the authenticated `userId`. Never allow cross-user access.
- **Worker idempotency:** retries must clear prior chunks before reprocessing to avoid duplicates.
- **Status lifecycle:** documents transition `UPLOADED → PROCESSING → PROCESSED | FAILED`; do not skip states or leave orphaned storage objects.
- **Cascade deletes:** rely on `onDelete: Cascade`; clean up Supabase Storage best-effort without blocking deletion.

## 5. Security

- Never commit secrets or `.env` files. Keep service-role keys server-only.
- Validate file content by magic bytes, not just mimetype; enforce size limits.
- Do not bypass `protectRoute` or Clerk auth on protected endpoints.
- Do not expose Supabase service-role key to the client.
- Rate limits must remain in place for upload and chat.

## 6. Code style & consistency

- Follow existing patterns: ESM imports, controller/service/middleware separation, `ApiError` for HTTP errors, `logger`/`timed` for observability.
- Use the existing config singletons (`prisma`, `redis`, `supabase`) — do not create new clients.
- Keep changes minimal and focused; avoid unrelated refactors.
- Preserve file naming and folder conventions under `src/`.
- Do not edit generated files (`prisma client`, `.next/`, lockfiles) by hand; regenerate via scripts.

## 7. Reliability

- Handle external-service failures gracefully (Supabase, Redis, Gemini, Groq) and surface meaningful errors.
- Background jobs must update document status to `FAILED` with an error message on final failure.
- Rate limiter and health checks must fail open / report degraded state rather than crash the process.
- Ensure graceful shutdown paths remain intact (SIGTERM/SIGINT handlers).

## 8. Documentation

- Update `ARCHITECTURE.md` and `README.md` when changing architecture, schema, env vars, or API contracts.
- Keep code comments accurate; remove stale comments.
- Do not add documentation files beyond what is requested unless useful.