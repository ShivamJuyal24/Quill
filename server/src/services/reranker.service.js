import { CohereClient } from 'cohere-ai';
import { env } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { timed } from '../utils/timed.js';

const cohere = env.cohereApiKey ? new CohereClient({ token: env.cohereApiKey }) : null;

export async function rerankChunks(query, chunks, topK) {
  if (!cohere || chunks.length === 0) return chunks.slice(0, topK);

  try {
    const response = await timed('rerank.request', () =>
      cohere.rerank({
        model: env.cohereRerankModel,
        query,
        documents: chunks.map((chunk) => ({ text: chunk.content })),
        topN: topK,
      }),
    );

    const ranked = (response.results || [])
      .map((result) => chunks[result.index])
      .filter(Boolean);

    return ranked.length > 0 ? ranked : chunks.slice(0, topK);
  } catch (error) {
    logger.warn('Rerank failed, falling back to fused order:', error?.message);
    return chunks.slice(0, topK);
  }
}
