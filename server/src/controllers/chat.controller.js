import { prisma } from '../config/prisma.js';
import { env } from '../config/env.js';
import { embedTexts, generateAnswer } from '../services/gemini.service.js';
import { ApiError } from '../utils/errors.js';
import { timed } from '../utils/timed.js';
import { logger } from '../utils/logger.js';

const MAX_QUESTION_LENGTH = 4000;

/**
 * Reciprocal Rank Fusion (RRF) for hybrid search.
 *
 * Combines two ranked result lists (vector + keyword) into a single ranked list.
 * Score for each item = sum(1 / (k + rank)) across all lists it appears in.
 * Items appearing in both lists get boosted; k controls how much weight is given
 * to top-ranked items vs. lower-ranked ones.
 *
 * @param {Array} vectorList - vector search results (already ranked)
 * @param {Array} keywordList - keyword search results (already ranked)
 * @param {number} limit - max number of fused results to return
 * @param {number} k - RRF constant (default 60)
 * @returns {Array} fused and re-ranked results, truncated to `limit`
 */
function fuseRRF(vectorList, keywordList, limit, k = 60) {
  const scores = new Map();

  const add = (list, sourceLabel) => {
    for (let i = 0; i < list.length; i += 1) {
      const item = list[i];
      const id = item.id;
      if (!scores.has(id)) {
        scores.set(id, { ...item, _rrfScore: 0, _source: [] });
      }
      const entry = scores.get(id);
      entry._rrfScore += 1 / (k + i + 1);
      if (!entry._source.includes(sourceLabel)) {
        entry._source.push(sourceLabel);
      }
    }
  };

  add(vectorList, 'vector');
  add(keywordList, 'keyword');

  return Array.from(scores.values())
    .sort((a, b) => b._rrfScore - a._rrfScore)
    .slice(0, limit)
    .map((entry) => {
      const { _rrfScore, _source, ...rest } = entry;
      return { ...rest, _rrfScore: Number(_rrfScore.toFixed(4)), _source };
    });
}

export const chat = async (req, res, prismaInstance = prisma) => {
  // Express invokes route handlers as fn(req, res, next), so the third
  // argument is the `next` function in production. Fall back to the module
  // Prisma client unless a test explicitly injected a real client object.
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  const requestId = req.headers['x-request-id'];
  const startedAt = process.hrtime.bigint();
  res.once('finish', () => {
    logger.info('Timing', {
      label: 'chat.total',
      requestId,
      durationMs: Number((Number(process.hrtime.bigint() - startedAt) / 1_000_000).toFixed(2)),
      statusCode: res.statusCode,
    });
  });
  const { documentId, message } = req.body ?? {};

  if (!documentId || typeof documentId !== 'string') {
    throw new ApiError(400, '"documentId" is required.');
  }
  if (!message || typeof message !== 'string' || !message.trim()) {
    throw new ApiError(400, '"message" is required.');
  }
  if (message.length > MAX_QUESTION_LENGTH) {
    throw new ApiError(400, `Question is too long (max ${MAX_QUESTION_LENGTH} characters).`);
  }

  const userId = req.userId;

  // Ownership + readiness check — a not-yet-processed document has no
  // embeddings to retrieve from yet.
  const document = await timed('chat.document-ownership-check', () => prismaInstance.document.findFirst({
    where: { id: documentId, userId },
    select: { id: true, originalName: true, status: true, errorMessage: true },
  }), { requestId, documentId });
  if (!document) throw new ApiError(404, 'Document not found.');
  if (document.status !== 'PROCESSED') {
    throw new ApiError(
      409,
      document.status === 'FAILED'
        ? `This document failed to process: ${document.errorMessage ?? 'unknown error'}`
        : 'This document is still processing. Please try again shortly.'
    );
  }

  // Embedding and history are independent after the ownership check.
  const [questionVectors, history] = await Promise.all([
    timed('chat.question-embedding', () => embedTexts([message.trim()]), { requestId, documentId }),
    timed('chat.history-fetch', () => prismaInstance.chatMessage.findMany({
      where: { documentId, userId },
      orderBy: { createdAt: 'desc' },
      take: 10,
      select: { role: true, content: true },
    }), { requestId, documentId, historyLimit: 10 }),
  ]);
  const [questionVector] = questionVectors;

  // 2. Retrieve chunks via hybrid search: vector (semantic) + full-text (lexical),
  //    fused with Reciprocal Rank Fusion (RRF). Falls back to vector-only when
  //    hybrid search is disabled or full-text search returns nothing.
  const vectorChunks = await timed('chat.vector-search', () => prismaInstance.$queryRaw`
    SELECT id, content, 1 - (embedding <=> ${questionVector}::vector) AS similarity
    FROM document_chunks
    WHERE document_id = ${documentId} AND embedding IS NOT NULL
    ORDER BY embedding <=> ${questionVector}::vector
    LIMIT ${env.topKChunks * 2}
  `, { requestId, documentId, topK: env.topKChunks * 2 });

  let keywordChunks = [];
  if (env.hybridSearchEnabled === 'true') {
    keywordChunks = await timed('chat.keyword-search', () => prismaInstance.$queryRaw`
      SELECT id, content, ts_rank(content_tsv, plainto_tsquery('english', ${message.trim()})) AS rank
      FROM document_chunks
      WHERE document_id = ${documentId} AND content_tsv @@ plainto_tsquery('english', ${message.trim()})
      ORDER BY rank DESC
      LIMIT ${env.topKChunks * 2}
    `, { requestId, documentId, topK: env.topKChunks * 2 });
  }

  const chunks = fuseRRF(vectorChunks, keywordChunks, env.topKChunks, env.rrfK);

  if (chunks.length === 0) {
    throw new ApiError(503, 'No searchable content is available for this document yet. Please try again shortly.');
  }

  // 3. Ground the answer in the retrieved context, with recent conversation history.
  const answer = await timed('chat.generation', () => generateAnswer({
    documentTitle: document.originalName,
    question: message.trim(),
    contextChunks: chunks.map((chunk) => chunk.content),
    history: history.reverse(),
  }), { requestId, documentId, contextChunkCount: chunks.length });

  // 4. Persist the exchange so the conversation survives reloads.
  const [, assistantMessage] = await timed('chat.message-persistence', () => prismaInstance.$transaction([
    prismaInstance.chatMessage.create({
      data: { documentId, userId, role: 'USER', content: message.trim() },
    }),
    prismaInstance.chatMessage.create({
      data: { documentId, userId, role: 'ASSISTANT', content: answer },
    }),
  ]), { requestId, documentId });

  res.json({
    messageId: assistantMessage.id,
    documentId,
    answer,
    sources: chunks.slice(0, 4).map((chunk) => ({
      id: chunk.id,
      similarity: Number(chunk.similarity),
      snippet: chunk.content.slice(0, 200),
    })),
  });
};
