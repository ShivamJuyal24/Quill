const serviceName = 'quill-server';
const isProduction = process.env.NODE_ENV === 'production';

const timestamp = () => new Date().toISOString();

/**
 * Build a structured log object.
 * Always includes: timestamp, level, service, message.
 * Merges optional meta fields (requestId, userId, label, durationMs, etc.) at top level.
 * If meta is not a plain object, it is appended to the message string instead.
 */
function makeLogEntry(level, message, meta = {}) {
  const entry = {
    timestamp: timestamp(),
    level,
    service: serviceName,
    message,
  };
  if (meta && typeof meta === 'object' && !Array.isArray(meta)) {
    Object.assign(entry, meta);
  } else if (meta !== undefined) {
    entry.message = `${message} ${meta}`;
  }
  return entry;
}

/**
 * Write a log entry as a single JSON line to stdout/stderr.
 * Error-level goes to stderr; everything else to stdout.
 */
function write(entry) {
  const line = JSON.stringify(entry);
  const stream = entry.level === 'error' || entry.level === 'fatal' ? process.stderr : process.stdout;
  stream.write(line + '\n');
}

/**
 * Logger with structured JSON output.
 * Usage:
 *   logger.info('User uploaded file', { userId, documentId, fileSize });
 *   logger.error('Failed to enqueue job', { documentId, error: err.message });
 *   logger.warn('Rate limiter unavailable', { error: err.message });
 */
export const logger = {
  info: (message, meta = {}) => {
    if (!isProduction || levelEnabled('info')) write(makeLogEntry('info', message, meta));
  },
  warn: (message, meta = {}) => write(makeLogEntry('warn', message, meta)),
  error: (message, meta = {}) => write(makeLogEntry('error', message, meta)),
  debug: (message, meta = {}) => {
    if (!isProduction) write(makeLogEntry('debug', message, meta));
  },
};

/**
 * Create a child logger with fixed fields merged into every entry.
 * Useful for per-request logging (e.g., attach requestId once).
 */
export function createLogger(bindings = {}) {
  return {
    info: (message, meta = {}) => {
      if (!isProduction || levelEnabled('info')) write(makeLogEntry('info', message, mergeMeta(bindings, meta)));
    },
    warn: (message, meta = {}) => write(makeLogEntry('warn', message, mergeMeta(bindings, meta))),
    error: (message, meta = {}) => write(makeLogEntry('error', message, mergeMeta(bindings, meta))),
    debug: (message, meta = {}) => {
      if (!isProduction) write(makeLogEntry('debug', message, mergeMeta(bindings, meta)));
    },
  };
}

function mergeMeta(bindings, meta) {
  if (meta && typeof meta === 'object' && !Array.isArray(meta)) {
    return { ...bindings, ...meta };
  }
  return bindings;
}

function levelEnabled(level) {
  const order = ['debug', 'info', 'warn', 'error'];
  const min = isProduction ? 'info' : 'debug';
  return order.indexOf(level) >= order.indexOf(min);
}