// ESLint flat config.
//
// Next 16 removed the `next lint` command and ESLint 10 dropped .eslintrc, so
// linting runs `eslint .` against this file.
//
// `eslint-config-next` is deliberately NOT used: version 16.2.12 declares
// `eslint >= 9` but bundles eslint-plugin-react@7.37.5, which still calls the
// pre-ESLint-10 rule-context API and crashes on load. The Next-specific rules
// that matter come from @next/eslint-plugin-next directly, which has no such
// transitive problem. React-specific linting is covered by TypeScript's own
// checks plus the react-hooks rules Next ships.
import nextPlugin from '@next/eslint-plugin-next';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['.next/**', 'node_modules/**', 'out/**', 'next-env.d.ts'] },

  ...tseslint.configs.recommended,

  {
    plugins: { '@next/next': nextPlugin },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs['core-web-vitals'].rules,
    },
  },

  {
    // The dashboard is a pure API client; `any` is never needed to talk to a
    // typed client, so this stays an error rather than a warning.
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
);
