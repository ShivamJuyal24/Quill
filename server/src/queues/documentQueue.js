import { Queue } from 'bullmq';
import { bullConnection } from '../config/redis.js';

export const QUEUE_NAME = 'document-queue';
export const JOB_NAME = 'process-document';

/** In test mode, use a mock queue to avoid creating real Redis connections. */
export const documentQueue =
  process.env.NODE_ENV === 'test'
    ? { add: async () => ({ id: 'mock-job-id' }), close: async () => {} }
    : new Queue(QUEUE_NAME, { connection: bullConnection });

export function enqueueDocumentJob(data) {
  return documentQueue.add(JOB_NAME, data, {
    attempts: 3,
    backoff: { type: 'exponential', delay: 15_000 },
    removeOnComplete: { count: 200 },
    removeOnFail: { count: 500 },
  });
}
