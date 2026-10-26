import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.js'],
    exclude: ['node_modules', 'dist', '.next'],
    environment: 'node',
    globals: false,
    // no global setup - mocks are per-test-file
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
