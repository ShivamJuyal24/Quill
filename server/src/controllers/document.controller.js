import { prisma } from '../config/prisma.js';
import { deletePdf } from '../services/storage.service.js';
import { ApiError, notFoundError } from '../utils/errors.js';

/** Every query is scoped by userId — one user can never read another user's documents. */
async function getOwnedDocument(userId, documentId, prismaInstance = prisma) {
  // Express invokes route handlers as fn(req, res, next), so the third
  // argument is the `next` function in production. Fall back to the module
  // Prisma client unless a test explicitly injected a real client object.
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  const document = await prismaInstance.document.findFirst({
    where: { id: documentId, userId },
    include: { _count: { select: { chunks: true, messages: true } } },
  });
  if (!document) throw notFoundError();
  return document;
}

export const listDocuments = async (req, res, prismaInstance = prisma) => {
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  const documents = await prismaInstance.document.findMany({
    where: { userId: req.userId },
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
  res.json({ documents });
};

export const getDocument = async (req, res, prismaInstance = prisma) => {
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  const document = await getOwnedDocument(req.userId, req.params.id, prismaInstance);
  const { _count, ...rest } = document;
  res.json({ document: { ...rest, chunkCount: _count.chunks, messageCount: _count.messages } });
};

export const getDocumentMessages = async (req, res, prismaInstance = prisma) => {
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  await getOwnedDocument(req.userId, req.params.id, prismaInstance);

  const messages = await prismaInstance.chatMessage.findMany({
    where: { documentId: req.params.id, userId: req.userId },
    orderBy: { createdAt: 'asc' },
    take: 200,
    select: { id: true, role: true, content: true, createdAt: true },
  });

  res.json({ messages });
};

export const deleteDocument = async (req, res, prismaInstance = prisma) => {
  if (typeof prismaInstance === 'function') {
    prismaInstance = prisma;
  }
  const document = await getOwnedDocument(req.userId, req.params.id, prismaInstance);

  // Chunks and chat messages are removed by ON DELETE CASCADE.
  await prismaInstance.document.delete({ where: { id: document.id } });

  // Storage cleanup is best-effort; the DB row is already gone.
  await deletePdf(document.filePath).catch(() => {});

  res.status(204).send();
};
