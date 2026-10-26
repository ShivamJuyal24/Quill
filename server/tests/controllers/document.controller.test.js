import { describe, it, beforeEach, expect, vi } from 'vitest';

// ---- Mock external services ----
vi.mock('../../src/services/storage.service.js', () => ({
  deletePdf: vi.fn(),
}));

import { deletePdf } from '../../src/services/storage.service.js';
import {
  listDocuments,
  getDocument,
  getDocumentMessages,
  deleteDocument,
} from '../../src/controllers/document.controller.js';

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
  const sendMock = vi.fn();
  const mockRes = {
    status: statusMock,
    json: jsonMock,
    send: sendMock,
    on: vi.fn(),
    once: vi.fn(),
  };
  statusMock.mockImplementation((code) => { mockRes._statusCode = code; return mockRes; });
  jsonMock.mockImplementation((body) => { mockRes._jsonBody = body; return mockRes; });
  sendMock.mockImplementation((body) => { mockRes._body = body; return mockRes; });
  return mockRes;
}

// ---- Mock functions ----
let mockPrismaDocumentFindFirst;
let mockPrismaDocumentFindMany;
let mockPrismaDocumentDelete;
let mockPrismaChatMessageFindMany;

describe('document.controller', () => {
  let req, res;

  beforeEach(() => {
    const ownedDoc = {
      id: 'doc_123',
      userId: 'user_123',
      originalName: 'test.pdf',
      status: 'PROCESSED',
      errorMessage: null,
      filePath: 'user_123/doc_123.pdf',
      _count: { chunks: 3, messages: 2 },
    };

    mockPrismaDocumentFindFirst = vi.fn(async () => ownedDoc);
    mockPrismaDocumentFindMany = vi.fn(async () => [
      { id: 'doc_1', userId: 'user_123', originalName: 'a.pdf' },
      { id: 'doc_2', userId: 'user_123', originalName: 'b.pdf' },
    ]);
    mockPrismaDocumentDelete = vi.fn(async () => ({ id: 'doc_123' }));
    mockPrismaChatMessageFindMany = vi.fn(async () => [
      { id: 'msg_1', role: 'USER', content: 'Hello', createdAt: 't1' },
      { id: 'msg_2', role: 'ASSISTANT', content: 'Hi', createdAt: 't2' },
    ]);
    deletePdf.mockImplementation(async () => undefined);

    const mockPrisma = {
      document: {
        findFirst: mockPrismaDocumentFindFirst,
        findMany: mockPrismaDocumentFindMany,
        delete: mockPrismaDocumentDelete,
      },
      chatMessage: { findMany: mockPrismaChatMessageFindMany },
    };

    res = createMockRes();

    req = {
      userId: 'user_123',
      params: { id: 'doc_123' },
      query: {},
      _mockPrisma: mockPrisma,
    };
  });

  async function callDeleteDocument() {
    try {
      await deleteDocument(req, res, req._mockPrisma);
    } catch (e) {
      res.status(e.status).json({ error: e.message });
    }
  }

  it('listDocuments returns only the authenticated user documents', async () => {
    await listDocuments(req, res, req._mockPrisma);
    expect(mockPrismaDocumentFindMany).toHaveBeenCalledWith({
      where: { userId: 'user_123' },
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        originalName: true,
        fileSize: true,
        status: true,
        errorMessage: true,
        createdAt: true,
        updatedAt: true,
        _count: { select: { chunks: true } },
      },
    });
    expect(res._jsonBody.documents.length).toBe(2);
  });

  it('getDocument returns document details for owner', async () => {
    await getDocument(req, res, req._mockPrisma);
    expect(mockPrismaDocumentFindFirst).toHaveBeenCalledWith({
      where: { id: 'doc_123', userId: 'user_123' },
      include: { _count: { select: { chunks: true, messages: true } } },
    });
    expect(res._jsonBody.document).toBeDefined();
    expect(res._jsonBody.document.chunkCount).toBe(3);
    expect(res._jsonBody.document.messageCount).toBe(2);
  });

  it('getDocument returns 404 when document does not belong to user', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve(null));
    try {
      await getDocument(req, res, req._mockPrisma);
    } catch (e) {
      res.status(e.status).json({ error: e.message });
    }
    expect(res._statusCode).toBe(404);
  });

  it('getDocumentMessages returns chat history', async () => {
    await getDocumentMessages(req, res, req._mockPrisma);
    expect(mockPrismaChatMessageFindMany).toHaveBeenCalledWith({
      where: { documentId: 'doc_123', userId: 'user_123' },
      orderBy: { createdAt: 'asc' },
      take: 200,
      select: { id: true, role: true, content: true, createdAt: true },
    });
    expect(res._jsonBody.messages.length).toBe(2);
  });

  it('deleteDocument deletes document and storage file for owner', async () => {
    await callDeleteDocument();
    expect(mockPrismaDocumentDelete).toHaveBeenCalledWith({
      where: { id: 'doc_123' },
    });
    expect(deletePdf).toHaveBeenCalledWith('user_123/doc_123.pdf');
    expect(res._statusCode).toBe(204);
  });

  it('deleteDocument returns 404 when document does not belong to user', async () => {
    mockPrismaDocumentFindFirst.mockImplementation(() => Promise.resolve(null));
    await callDeleteDocument();
    expect(res._statusCode).toBe(404);
    expect(mockPrismaDocumentDelete).not.toHaveBeenCalled();
  });
});
