-- Change embedding dimension from 3072 to 768 for Gemini text-embedding-004.
-- Existing 3072-dimensional vectors are incompatible and are cleared.
-- After migration, reprocess all documents to regenerate embeddings.

-- pgvector HNSW indexes support up to 2000 dimensions; 768-dim vectors qualify.
-- Drop any pre-existing index before altering the column type.
DROP INDEX IF EXISTS "document_chunks_embedding_hnsw_idx";

UPDATE "document_chunks"
SET "embedding" = NULL
WHERE "embedding" IS NOT NULL;

ALTER TABLE "document_chunks"
  ALTER COLUMN "embedding" TYPE vector(768)
  USING NULL::vector(768);

-- HNSW index for approximate nearest-neighbor search on cosine distance.
CREATE INDEX "document_chunks_embedding_hnsw_idx"
  ON "document_chunks" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
