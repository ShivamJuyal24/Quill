import { describe, it, beforeEach, expect, vi } from 'vitest';

// ---- Mock external services ----
vi.mock('../../src/queues/documentQueue.js', () => ({
  enqueueDocumentJob: vi.fn(),
}));

vi.mock('../../src/services/storage.service.js', () => ({
  uploadPdf: vi.fn(),
  deletePdf: vi.fn(),
}));

import { enqueueDocumentJob } from '../../src/queues/documentQueue.js';
import { uploadPdf, deletePdf } from '../../src/services/storage.service.js';
import { uploadFile } from '../../src/controllers/upload.controller.js';

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
let mockPrismaDocumentCreate;
let mockPrismaDocumentUpdate;

const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

describe('upload.controller', () => {
  let req, res;

  beforeEach(() => {
    mockPrismaDocumentCreate = vi.fn(async () => ({
      id: 'doc_123',
      userId: 'user_123',
      originalName: 'test.pdf',
      createdAt: '2026-01-01T00:00:00Z',
    }));
    mockPrismaDocumentUpdate = vi.fn(async () => ({ id: 'doc_123', status: 'FAILED' }));

    enqueueDocumentJob.mockImplementation(async () => ({ id: 'job_123' }));
    uploadPdf.mockImplementation(async () => undefined);
    deletePdf.mockImplementation(async () => undefined);

    const mockPrisma = {
      document: {
        create: mockPrismaDocumentCreate,
        update: mockPrismaDocumentUpdate,
      },
    };

    res = createMockRes();

    req = {
      auth: { userId: 'user_123' },
      headers: {},
      file: {
        buffer: Buffer.from('%PDF-1.4 fake pdf content'),
        originalname: 'test.pdf',
        size: 100,
      },
      _mockPrisma: mockPrisma,
    };
  });

  async function callUploadFile() {
    try {
      await uploadFile(req, res, req._mockPrisma);
    } catch (e) {
      res.status(e.status).json({ error: e.message });
    }
  }

  it('returns 400 when no file is uploaded', async () => {
    req.file = undefined;
    await callUploadFile();
    expect(res._statusCode).toBe(400);
    expect(res._jsonBody.error).toContain('No file uploaded');
  });

  it('returns 400 when file is not a valid PDF (bad magic bytes)', async () => {
    req.file = { buffer: Buffer.from('not a pdf file'), originalname: 'test.txt', size: 100 };
    await callUploadFile();
    expect(res._statusCode).toBe(400);
    expect(res._jsonBody.error).toContain('not a valid PDF');
  });

  it('returns 201 on successful upload', async () => {
    await uploadFile(req, res, req._mockPrisma);
    expect(res._statusCode).toBe(201);
    expect(res._jsonBody.status).toBe('UPLOADED');
  });

  it('calls uploadPdf with correct storage path', async () => {
    await uploadFile(req, res, req._mockPrisma);
    expect(uploadPdf).toHaveBeenCalledTimes(1);
    expect(uploadPdf).toHaveBeenCalledWith(
      expect.stringMatching(/^user_123\/[0-9a-f-]{36}\.pdf$/),
      req.file.buffer
    );
  });

  it('creates document record with correct data', async () => {
    await uploadFile(req, res, req._mockPrisma);
    expect(mockPrismaDocumentCreate).toHaveBeenCalledTimes(1);
    expect(mockPrismaDocumentCreate).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({
          userId: 'user_123',
          originalName: 'test.pdf',
          fileSize: 100,
          id: expect.stringMatching(uuidRegex),
          fileName: expect.stringMatching(/^[0-9a-f-]{36}\.pdf$/),
          filePath: expect.stringMatching(/^user_123\/[0-9a-f-]{36}\.pdf$/),
        }),
      })
    );
  });

  it('enqueues document processing job', async () => {
    await uploadFile(req, res, req._mockPrisma);
    expect(enqueueDocumentJob).toHaveBeenCalledTimes(1);
    expect(enqueueDocumentJob).toHaveBeenCalledWith({
      userId: 'user_123',
      documentId: expect.stringMatching(uuidRegex),
      storagePath: expect.stringMatching(/^user_123\/[0-9a-f-]{36}\.pdf$/),
    });
  });

  it('returns 503 when job queue fails', async () => {
    enqueueDocumentJob.mockImplementation(() => Promise.reject(new Error('queue down')));
    await callUploadFile();
    expect(res._statusCode).toBe(503);
  });

  it('deletes stored PDF when upload fails after storage', async () => {
    mockPrismaDocumentCreate.mockImplementation(() => Promise.reject(new Error('db error')));
    await uploadFile(req, res, req._mockPrisma).catch(() => {});
    expect(deletePdf).toHaveBeenCalledTimes(1);
    expect(deletePdf).toHaveBeenCalledWith(
      expect.stringMatching(/^user_123\/[0-9a-f-]{36}\.pdf$/)
    );
  });
});
