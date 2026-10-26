import { describe, it, beforeEach, expect, vi } from 'vitest';

// ---- Mock external services ----
vi.mock('../../src/services/gemini.service.js', () => ({
  embedTexts: vi.fn(),
  generateAnswer: vi.fn(),
}));

import { embedTexts, generateAnswer } from '../../src/services/gemini.service.js';
import { chat } from '../../src/controllers/chat.controller.js';

// ---- Env setup ----
process.env.NODE_ENV = 'test';
process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test';
process.env.DIRECT_URL = 'postgresql://test:test@localhost:5432/test';
process.env.UPSTASH_REDIS_URL = 'redis://localhost:1/test';
process.env.GEMINI_API_KEY = 'test-key';
process.env.CLERK_PUBLISHABLE_KEY = 'pk_test';
process.env.CLERK_SECRET_KEY = 'sk_test';
process.env.SUPABASE_URL = 'https://test.supabase.co';
process.env.SUPABASE_SERVICE_ROLE_KEY = 'test-key';

// ---- Helpers ----
function createMockRes() {
  const statusMock = vi.fn();
  const jsonMock = vi.fn();
  const mockRes = {
    status: statusMock,
    json: jsonMock,
    on: vi.fn(),
    once: vi.fn(),
  };
  statusMock.mockImplementation((code) => { mockRes._statusCode = code; return mockRes; });
  jsonMock.mockImplementation((body) => { mockRes._jsonBody = body; return mockRes; });
  return mockRes;
}

// ---- Mock functions ----
let mockPrismaDocumentFindFirst;
let mockPrismaChatMessageFindMany;
let mockPrismaQueryRaw;
let mockPrismaTransaction;
let mockPrismaChatMessageCreate;

describe('chat.controller', () => {
  let req, res;

  beforeEach(() => {
    mockPrismaDocumentFindFirst = vi.fn(async () => ({
      id: 'doc_123',
      userId: 'user_123',
      originalName: 'test.pdf',
      status: 'PROCESSED',
      errorMessage: null,
    }));
    mockPrismaChatMessageFindMany = vi.fn(async () => Promise.resolve([]));
    mockPrismaChatMessageCreate = vi.fn(async () => ({ id: 'msg_1' }));
    mockPrismaQueryRaw = vi.fn(async () => Promise.resolve([
      { id: 'chunk_1', content: 'Chunk content about the document', similarity: 0.95 },
    ]));
    mockPrismaTransaction = vi.fn(async (arr) => Promise.all(arr));

    embedTexts.mockImplementation(async () => Promise.resolve(['[0.1,0.2,0.3]']));
    generateAnswer.mockImplementation(async () => Promise.resolve('The document is about testing.'));

    const mockPrisma = {
      document: { findFirst: mockPrismaDocumentFindFirst },
      chatMessage: {
        findMany: mockPrismaChatMessageFindMany,
        create: mockPrismaChatMessageCreate,
      },
      $queryRaw: mockPrismaQueryRaw,
      $transaction: mockPrismaTransaction,
    };

    res = createMockRes();

    req = {
      userId: 'user_123',
      headers: {},
      body: { documentId: 'doc_123', message: 'What is this document about?' },
      _mockPrisma: mockPrisma,
    };
  });

  async function callChat() {
    try {
      await chat(req, res, req._mockPrisma);
    } catch (e) {
      res.status(e.status).json({ error: e.message });
    }
  }

  it('returns 400 when documentId is missing', async () => {
    req.body = { message: 'Hello' };
    await callChat();
    expect(res._statusCode).toBe(400);
  });

  it('returns 400 when message is missing', async () => {
    req.body = { documentId: 'doc_123' };
    await callChat();
    expect(res._statusCode).toBe(400);
  });

  it('returns 404 when document not found', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve(null));
    await callChat();
    expect(res._statusCode).toBe(404);
  });

  it('returns 409 when document is still processing', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve({
      id: 'doc_123', userId: 'user_123', status: 'PROCESSING', errorMessage: null,
    }));
    await callChat();
    expect(res._statusCode).toBe(409);
  });

  it('returns 409 when document failed', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve({
      id: 'doc_123', userId: 'user_123', status: 'FAILED', errorMessage: 'Parsing error',
    }));
    await callChat();
    expect(res._statusCode).toBe(409);
  });

  it('returns answer when document is PROCESSED', async () => {
    await chat(req, res, req._mockPrisma);
    expect(res._jsonBody.answer).toBe('The document is about testing.');
    expect(res._jsonBody.documentId).toBe('doc_123');
  });

  it('embeds the question before vector search', async () => {
    await chat(req, res, req._mockPrisma);
    expect(embedTexts).toHaveBeenCalledWith(['What is this document about?']);
  });

  it('performs vector search with question vector', async () => {
    await chat(req, res, req._mockPrisma);
    expect(mockPrismaQueryRaw).toHaveBeenCalled();
  });

  it('persists user and assistant messages in transaction', async () => {
    await chat(req, res, req._mockPrisma);
    expect(mockPrismaTransaction).toHaveBeenCalled();
    expect(mockPrismaChatMessageCreate).toHaveBeenCalledTimes(2);
    expect(mockPrismaChatMessageCreate).toHaveBeenCalledWith({
      data: expect.objectContaining({ role: 'USER', documentId: 'doc_123', userId: 'user_123' }),
    });
    expect(mockPrismaChatMessageCreate).toHaveBeenCalledWith({
      data: expect.objectContaining({ role: 'ASSISTANT', documentId: 'doc_123', userId: 'user_123' }),
    });
  });

  it('enforces ownership - user cannot access another user document', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve(null));
    await callChat();
    expect(mockPrismaDocumentFindFirst).toHaveBeenCalledWith({
      where: { id: 'doc_123', userId: 'user_123' },
      select: { id: true, originalName: true, status: true, errorMessage: true },
    });
    expect(res._statusCode).toBe(404);
  });
});
