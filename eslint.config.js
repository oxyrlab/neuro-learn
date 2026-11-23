import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'coverage', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.strict,
  {
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      'no-restricted-exports': [
        'error',
        {
          restrictDefaultExports: {
            direct: true,
            named: true,
            defaultFrom: true,
            namedFrom: true,
            namespaceFrom: true,
          },
        },
      ],
    },
  },
  {
    // Determinism: simulation code must never read ambient randomness or time.
    files: ['src/**/*.ts'],
    ignores: ['src/**/*.test.ts'],
    rules: {
      'no-restricted-properties': [
        'error',
        { object: 'Math', property: 'random', message: 'Use an injected Rng (see docs/adr/0003).' },
        { object: 'Date', property: 'now', message: 'Time must not influence learning results.' },
        {
          object: 'performance',
          property: 'now',
          message: 'Time must not influence learning results.',
        },
      ],
    },
  },
  {
    files: ['*.config.ts', '*.config.js'],
    rules: { 'no-restricted-exports': 'off' },
  },
);
