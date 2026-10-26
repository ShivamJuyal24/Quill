import IORedis from 'ioredis';
import { env } from './env.js';
import { logger } from '../utils/logger.js';

// Upstash requires TLS and BullMQ requires these flags over it.
const baseOptions = {
  maxRetriesPerRequest: null,
  enableReadyCheck: false,
};

/** Mock Redis connections for test mode (avoids real network connections). */
const makeMockRedis = () => ({
  ping: async () => 'PONG',
  quit: async () => 'OK',
  disconnect: async () => {},
  on: () => {},
  once: () => {},
  off: () => {},
  eval: async () => 1,
  evalsha: async () => 1,
  incr: async () => 1,
  expire: async () => 1,
  get: async () => null,
  set: async () => 'OK',
  del: async () => 0,
  keys: async () => [],
});

/** General-purpose connection: rate limiting, health checks, caching. */
export const redis =
  process.env.NODE_ENV === 'test' ? makeMockRedis() : new IORedis(env.redisUrl, baseOptions);

/**
 * Dedicated connection for BullMQ. The worker holds its connection in
 * blocking mode while waiting for jobs, so it must not be shared with
 * regular commands.
 */
export const bullConnection =
  process.env.NODE_ENV === 'test' ? makeMockRedis() : new IORedis(env.redisUrl, baseOptions);

for (const [name, connection] of [
  ['redis', redis],
  ['bull', bullConnection],
]) {
  connection.on('error', (err) => logger.error(`Redis [${name}] error:`, err.message));
}
