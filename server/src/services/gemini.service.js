import { GoogleGenerativeAI } from '@google/generative-ai';
import { env } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { timed } from '../utils/timed.js';

const genAI = new GoogleGenerativeAI(env.geminiApiKey);

const embeddingModel = genAI.getGenerativeModel({ model: env.geminiEmbeddingModel });
const chatModel = genAI.getGenerativeModel({ model: env.geminiChatModel });

// Gemini accepts at most 100 embed requests per batch call.
const EMBED_BATCH_SIZE = 100;

const RETRYABLE = /429|quota|overload|resource.?exhausted|5\d\d|ECONNRESET|ETIMEDOUT|fetch failed|network/i;

async function withRetry(fn, { attempts = 3, baseDelayMs = 1500, label = 'Gemini call' } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await timed(`${label}.attempt-${attempt}`, fn, { attempt });
    } catch (error) {
      lastError = error;
      const retryable = RETRYABLE.test(String(error?.message ?? error));
      if (attempt === attempts || !retryable) break;
      const delay = baseDelayMs * 2 ** (attempt - 1);
      logger.warn(`${label} failed (attempt ${attempt}/${attempts}), retrying in ${delay}ms:`, error?.message);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}

/**
 * Embeds texts and returns pgvector literals (`"[0.1,0.2,...]"`), one per input,
 * in the same order. Each call requests 3072-dimensional output to match the
 * `vector(3072)` column. Uses `embedContent` (single) because `batchEmbedContents`
 * on v1beta does not accept a per-request `config` with `outputDimensionality`. */
export async function embedTexts(texts) {
  if (texts.length === 0) return [];

  const vectorStrings = [];
  for (const content of texts) {
    const result = await withRetry(
      () => timed('embedding.request', () =>
        embeddingModel.embedContent({
          content: { parts: [{ text: content }] },
          output_dimensionality: 3072,
        })),
      { label: 'Embedding request' },
    );

    const embedding = result?.embedding;
    if (!embedding) {
      throw new Error(`Gemini returned no embedding for input`);
    }

    vectorStrings.push(`[${embedding.values.map(Number).join(',')}]`);
  }

  return vectorStrings;
}

const SYSTEM_PROMPT = `You are a helpful assistant that answers questions about a PDF document.
Answer using ONLY the provided context excerpts. If the context does not contain
the answer, say you don't know — never invent facts. When you use an excerpt,
cite only its numbered label, for example [1] or [2]. The excerpts do not
contain page or line-number metadata, so never invent citations like
line-range references such as [1:L1-L2], page numbers, URLs, or GitHub-style
references. Use plain Markdown
and be concise and specific.`;

async function generateGroqAnswer({ contents, systemPrompt }) {
  const messages = [
    { role: 'system', content: systemPrompt },
    ...contents.map((content) => ({
      role: content.role === 'model' ? 'assistant' : content.role,
      content: content.parts.map((part) => part.text).join(''),
    })),
  ];

  const response = await timed('chat.groq.request', () => fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.groqApiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: env.groqChatModel,
      messages,
      temperature: 0.2,
      max_tokens: 2048,
    }),
  }), { model: env.groqChatModel });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`Groq request failed (${response.status}): ${data.error?.message ?? response.statusText}`);
  }

  const answer = data.choices?.[0]?.message?.content?.trim();
  if (!answer) throw new Error('Groq returned an empty answer');
  return answer;
}

/**
 * Generates an answer grounded in the retrieved context chunks.
 * `history` is an array of { role: 'USER' | 'ASSISTANT', content } in chronological order.
 */
export async function generateAnswer({ documentTitle, question, contextChunks, history = [] }) {
  const context = contextChunks
    .map((content, index) => `[${index + 1}] ${content}`)
    .join('\n\n');

  const userContent = `Document: "${documentTitle}"

Context excerpts:
${context}

Question: ${question}

Citation rule: cite only the available chunk labels [1], [2], etc. Do not
invent page numbers, line numbers, URLs, or any other citation format.`;

  const contents = [
    ...history.slice(-10).map((message) => ({
      role: message.role === 'ASSISTANT' ? 'model' : 'user',
      parts: [{ text: message.content }],
    })),
    { role: 'user', parts: [{ text: userContent }] },
  ];

  const generateGeminiAnswer = async () => {
    const result = await withRetry(
      () => timed('chat.gemini.request', () =>
        chatModel.generateContent({
          systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
          contents,
          generationConfig: { temperature: 0.2, maxOutputTokens: 2048 },
        })),
      { label: `Chat generation fallback (${env.geminiChatModel})` },
    );

    const answer = result.response?.text()?.trim();
    if (!answer) throw new Error('Gemini returned an empty answer');
    return answer;
  };

  if (!env.groqApiKey) return generateGeminiAnswer();

  try {
    return await withRetry(
      () => generateGroqAnswer({ contents, systemPrompt: SYSTEM_PROMPT }),
      { label: `Chat generation (${env.groqChatModel})` },
    );
  } catch (groqError) {
    logger.warn(
      `Groq chat failed; trying Gemini fallback (${env.geminiChatModel}):`,
      groqError?.message,
    );
    return generateGeminiAnswer();
  }
}
