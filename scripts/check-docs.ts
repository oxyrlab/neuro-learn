/**
 * Documentation gate (`npm run docs:check`). Fails when:
 * 1. a source file under `src/` (other than `index.ts`) has no sibling `.test.ts`;
 * 2. a public export lacks a TSDoc block directly above it;
 * 3. a relative markdown link under `docs/` or in the root `*.md` files points to a missing file;
 * 4. a required docs file is missing.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const problems: string[] = [];

function walk(directory: string): string[] {
  return readdirSync(directory).flatMap((entry) => {
    const full = join(directory, entry);
    return statSync(full).isDirectory() ? walk(full) : [full];
  });
}

const requiredDocs = [
  'docs/README.md',
  'docs/STATUS.md',
  'docs/CHANGELOG.md',
  'docs/glossary.md',
  'AGENTS.md',
  'CLAUDE.md',
  'README.md',
];
for (const file of requiredDocs) {
  if (!existsSync(join(root, file))) problems.push(`Missing required docs file: ${file}`);
}

const sourceFiles = walk(join(root, 'src')).filter(
  (file) => file.endsWith('.ts') && !file.endsWith('.test.ts') && !file.endsWith('index.ts'),
);
for (const file of sourceFiles) {
  const relative = file.slice(root.length + 1);
  if (!existsSync(file.replace(/\.ts$/, '.test.ts'))) {
    problems.push(`${relative}: no sibling test file (${relative.replace(/\.ts$/, '.test.ts')}).`);
  }
  const lines = readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, index) => {
    if (/^export\s/.test(line) && !/^export\s+(type\s+)?\{/.test(line)) {
      const previous = lines[index - 1]?.trim() ?? '';
      if (!previous.endsWith('*/')) {
        problems.push(`${relative}:${index + 1}: public export without a TSDoc block.`);
      }
    }
  });
}

const markdownFiles = [
  ...walk(join(root, 'docs')).filter((file) => file.endsWith('.md')),
  ...readdirSync(root)
    .filter((file) => file.endsWith('.md'))
    .map((file) => join(root, file)),
];
const linkPattern = /\[[^\]]*\]\(([^)]+)\)/g;
for (const file of markdownFiles) {
  const text = readFileSync(file, 'utf8');
  for (const match of text.matchAll(linkPattern)) {
    const target = match[1]?.split('#')[0] ?? '';
    if (target === '' || /^[a-z]+:/i.test(target)) continue;
    if (!existsSync(resolve(dirname(file), target))) {
      problems.push(`${file.slice(root.length + 1)}: broken link "${target}".`);
    }
  }
}

if (problems.length > 0) {
  console.error(
    `docs:check failed with ${problems.length} problem(s):\n- ${problems.join('\n- ')}`,
  );
  process.exit(1);
}
console.log('docs:check passed.');
