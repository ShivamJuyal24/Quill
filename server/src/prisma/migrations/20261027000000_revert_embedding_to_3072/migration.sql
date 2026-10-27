-- Revert embedding dimension from 768 back to 3072 for Gemini gemini-embedding-001.
-- Existing 768-dimensional vectors are incompatible and are cleared.

DROP INDEX IF EXISTS "document_chunks_embedding_hnsw_idx";

ALTER TABLE "document_chunks"
  ALTER COLUMN "embedding" TYPE vector(3072)
  USING NULL::vector(3072);

-- HNSW is not used because pgvector limits HNSW indexes to 2000 dimensions.
