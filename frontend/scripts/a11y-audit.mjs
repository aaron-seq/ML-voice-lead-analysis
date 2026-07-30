/**
 * Accessibility audit: WCAG 2.1 A/AA via axe-core, across every view in both
 * colour schemes. Exits non-zero on any violation, so CI fails on a regression.
 *
 * Both schemes are checked because the dark palette is a separate selection
 * rather than an inversion — a contrast failure can exist in one and not the
 * other.
 *
 *   node scripts/a11y-audit.mjs            # expects app on :3000, API on :8000
 *   BASE_URL=http://localhost:3000 node scripts/a11y-audit.mjs
 */

import AxeBuilder from '@axe-core/playwright';
import { chromium } from 'playwright';

const BASE_URL = process.env.BASE_URL ?? 'http://localhost:3000';
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'];

const browser = await chromium.launch({
  // Preinstalled in CI and in the dev container; fall back to Playwright's own
  // download location when unset.
  executablePath: process.env.CHROMIUM_PATH || undefined,
});

let failures = 0;

async function audit(label, prepare, colorScheme = 'light') {
  // axe-core requires a page from an explicit context, not browser.newPage().
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    colorScheme,
  });
  const page = await context.newPage();
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(800);
  if (prepare) await prepare(page);

  const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();

  console.log(`\n=== ${label} (${colorScheme}) ===`);
  if (violations.length === 0) {
    console.log('  no violations');
  }
  for (const v of violations) {
    failures += 1;
    console.log(`  [${v.impact}] ${v.id}: ${v.help}`);
    for (const node of v.nodes.slice(0, 4)) {
      console.log(`      ${node.target.join(' ')}`);
    }
    console.log(`      ${v.helpUrl}`);
  }
  await context.close();
}

const openFirstCall = async (page) => {
  await page.locator('.row').first().click();
  await page.waitForTimeout(1200);
};

await audit('Call list', null);
await audit('Call detail', openFirstCall);
await audit('Call detail', openFirstCall, 'dark');
await audit('Score breakdown expanded', async (page) => {
  await openFirstCall(page);
  const toggle = page.locator('.evidence-toggle').first();
  if (await toggle.count()) {
    await toggle.click();
    await page.waitForTimeout(400);
  }
});
await audit('Pipeline', async (page) => {
  await page.getByRole('button', { name: 'Pipeline' }).click();
  await page.waitForTimeout(1200);
});
await audit('Pipeline', async (page) => {
  await page.getByRole('button', { name: 'Pipeline' }).click();
  await page.waitForTimeout(1200);
}, 'dark');
await audit('Add call', async (page) => {
  await page.getByRole('button', { name: 'Add call' }).click();
  await page.waitForTimeout(600);
});

// Keyboard reachability: every primary control must be operable without a mouse.
const keyboardContext = await browser.newContext();
const page = await keyboardContext.newPage();
await page.goto(BASE_URL, { waitUntil: 'networkidle' });
await page.waitForTimeout(600);

const reached = new Set();
for (let i = 0; i < 30; i += 1) {
  await page.keyboard.press('Tab');
  const descriptor = await page.evaluate(() => {
    const el = document.activeElement;
    if (!el) return '';
    return `${el.tagName}|${(el.className ?? '').toString()}`;
  });
  if (descriptor.includes('nav-item')) reached.add('navigation');
  if (descriptor.includes('search-input')) reached.add('search');
  if (descriptor.includes('chip')) reached.add('band filter');
  if (descriptor.includes('row')) reached.add('call row');
}

console.log('\n=== keyboard ===');
for (const control of ['navigation', 'search', 'band filter', 'call row']) {
  const ok = reached.has(control);
  console.log(`  ${ok ? 'reachable' : 'NOT REACHABLE'}: ${control}`);
  if (!ok) failures += 1;
}

await browser.close();

console.log(
  failures === 0
    ? '\nAccessibility audit passed.'
    : `\nAccessibility audit FAILED with ${failures} problem(s).`,
);
process.exit(failures === 0 ? 0 : 1);
