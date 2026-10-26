import { Router } from 'express';
import { prisma } from '../config/prisma.js';
import { redis, bullConnection } from '../config/redis.js';
import { supabase } from '../config/supabase.js';
import { env } from '../config/env.js';

export const healthRouter = Router();

/**
 * Runs a promise with a timeout and returns { ok, latencyMs }.
 */
async function probe(promise, ms) {
  const start = process.hrtime.bigint();
  try {
    await Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), ms)),
    ]);
    return { ok: true, latencyMs: Number((Number(process.hrtime.bigint() - start) / 1_000_000).toFixed(2)) };
  } catch {
    return { ok: false, latencyMs: null };
  }
}

/**
 * Extracts the ok flag from a Promise.allSettled result.
 * probe() already returns { ok, latencyMs } on fulfillment.
 */
function settledOk(result) {
  return result.status === 'fulfilled' && result.value.ok;
}

/**
 * Lists the Supabase storage bucket. Unlike a plain fetch, Supabase's
 * .list() resolves with { data, error } instead of throwing on errors,
 * so we reject explicitly when error is present.
 */
async function supabaseStorageProbe() {
  const { data, error } = await supabase.storage.from(env.supabaseBucket).list({ limit: 1 });
  if (error) throw error;
  return data;
}

healthRouter.get('/health', async (_req, res) => {
  // All dependency probes run in parallel so a slow provider cannot block
  // the readiness probe.
  const [postgres, redisResult, bullmq, storage, geminiProbe, groqProbe] =
    await Promise.allSettled([
      probe(prisma.$queryRaw`SELECT 1`, 3000),
      probe(redis.ping(), 3000),
      probe(bullConnection.ping(), 3000),
      probe(supabaseStorageProbe(), 5000),
      probe(
        fetch(`https://generativelanguage.googleapis.com/v1beta/models/${env.geminiEmbeddingModel}`, {
          headers: { 'x-goog-api-key': env.geminiApiKey },
        }).then((r) => (r.ok ? r : Promise.reject(new Error(`HTTP ${r.status}`)))),
        3000,
      ),
      probe(
        fetch('https://api.groq.com/openai/v1/models', {
          headers: { Authorization: `Bearer ${env.groqApiKey}` },
        }).then((r) => (r.ok ? r : Promise.reject(new Error(`HTTP ${r.status}`)))),
        3000,
      ),
    ]);

  const postgresOk = settledOk(postgres);
  const redisOk = settledOk(redisResult);
  const bullmqOk = settledOk(bullmq);
  const storageOk = settledOk(storage);
  const geminiOk = settledOk(geminiProbe);
  const groqOk = settledOk(groqProbe);

  // Critical dependencies determine overall health.
  const criticalOk = postgresOk && redisOk && bullmqOk && storageOk;

  const dependencies = {
    postgres: postgresOk,
    redis: redisOk,
    bullmq: bullmqOk,
    supabase_storage: storageOk,
    gemini: Boolean(env.geminiApiKey) && geminiOk,
    groq: Boolean(env.groqApiKey) && groqOk,
  };

  const details = {
    postgres: postgresOk ? { latencyMs: postgres.value.latencyMs } : null,
    redis: redisOk ? { latencyMs: redisResult.value.latencyMs } : null,
    bullmq: bullmqOk ? { latencyMs: bullmq.value.latencyMs } : null,
    supabase_storage: storageOk ? { latencyMs: storage.value.latencyMs } : null,
    gemini: {
      configured: Boolean(env.geminiApiKey),
      healthy: geminiOk,
      latencyMs: geminiProbe.status === 'fulfilled' ? geminiProbe.value.latencyMs : null,
    },
    groq: {
      configured: Boolean(env.groqApiKey),
      healthy: groqOk,
      latencyMs: groqProbe.status === 'fulfilled' ? groqProbe.value.latencyMs : null,
    },
  };

  const healthy = criticalOk;
  const statusCode = healthy ? 200 : 503;

  res.status(statusCode).json({
    status: healthy ? 'ok' : 'degraded',
    dependencies,
    details,
    timestamp: new Date().toISOString(),
    uptimeSeconds: Math.round(process.uptime()),
  });
});
