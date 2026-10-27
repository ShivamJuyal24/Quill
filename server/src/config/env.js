import dotenv from 'dotenv';

dotenv.config();

const REQUIRED_VARS = [
  'DATABASE_URL',
  'DIRECT_URL',
  'UPSTASH_REDIS_URL',
  'GEMINI_API_KEY',
  'CLERK_PUBLISHABLE_KEY',
  'CLERK_SECRET_KEY',
  'SUPABASE_URL',
  'SUPABASE_SERVICE_ROLE_KEY',
];

const missing = REQUIRED_VARS.filter((name) => !process.env[name] || !process.env[name].trim());

if (missing.length > 0) {
  console.error('❌ Missing required environment variables:', missing.join(', '));
  console.error('   Copy .env.example to .env and fill in every value before starting the server.');
  process.exit(1);
}

const toInt = (value, fallback) => {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};

export const env = {
  nodeEnv: process.env.NODE_ENV ?? 'development',
  isProd: process.env.NODE_ENV === 'production',
  port: toInt(process.env.PORT, 5000),

  // Comma-separated list of allowed browser origins (the Next.js client).
  clientOrigins: (process.env.CLIENT_URL ?? 'http://localhost:3000')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean),

  // Supabase Postgres (Prisma). DATABASE_URL goes through the transaction
  // pooler; DIRECT_URL is the session-mode connection used for migrations.
  databaseUrl: process.env.DATABASE_URL,
  directUrl: process.env.DIRECT_URL,

  // Upstash Redis (BullMQ + rate limiting).
  redisUrl: process.env.UPSTASH_REDIS_URL,

  // Supabase Storage (PDF files). Service-role key: server-side only.
  supabaseUrl: process.env.SUPABASE_URL,
  supabaseServiceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY,
  supabaseBucket: process.env.SUPABASE_STORAGE_BUCKET ?? 'documents',

   // Gemini embeddings must remain dimension-compatible with vector(3072).
  geminiApiKey: process.env.GEMINI_API_KEY,
  geminiEmbeddingModel: process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-001',
  geminiChatModel: process.env.GEMINI_CHAT_MODEL ?? 'gemini-2.0-flash',
   groqApiKey: process.env.GROQ_API_KEY,
   groqChatModel: process.env.GROQ_CHAT_MODEL ?? 'llama-3.3-70b-versatile',
   cohereApiKey: process.env.COHERE_API_KEY,
   cohereRerankModel: process.env.COHERE_RERANK_MODEL ?? 'rerank-english-v3.0',

  // Clerk
  clerkPublishableKey: process.env.CLERK_PUBLISHABLE_KEY,
  clerkSecretKey: process.env.CLERK_SECRET_KEY,

  // Tunables
  maxFileSizeMb: toInt(process.env.MAX_FILE_SIZE_MB, 20),
  maxChunksPerDocument: toInt(process.env.MAX_CHUNKS_PER_DOCUMENT, 4000),
  workerConcurrency: toInt(process.env.WORKER_CONCURRENCY, 2),
  topKChunks: toInt(process.env.TOP_K_CHUNKS, 8),
  hybridSearchEnabled: process.env.HYBRID_SEARCH_ENABLED ?? 'true',
  rrfK: toInt(process.env.RRF_K, 60),
  rateLimitUploadPerHour: toInt(process.env.RATE_LIMIT_UPLOAD_PER_HOUR, 10),
  rateLimitChatPerMin: toInt(process.env.RATE_LIMIT_CHAT_PER_MIN, 20),
};
