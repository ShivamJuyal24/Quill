-- Add full-text search support to document_chunks for hybrid retrieval.
-- A generated tsvector column is populated automatically from `content`,
-- and a GIN index enables fast lexical search via plainto_tsquery / ts_rank.

ALTER TABLE "document_chunks"
  ADD COLUMN "content_tsv" tsvector
    GENERATED ALWAYS AS (to_tsvector('english', "content")) STORED;

CREATE INDEX "document_chunks_content_tsv_gin_idx"
  ON "document_chunks" USING GIN ("content_tsv");
