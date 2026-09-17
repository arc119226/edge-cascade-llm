/**
 * 建置：把要上線的東西集中到 dist/。
 *
 * 刻意不用打包工具。這個專案的 JS 本來就是 ES module，瀏覽器直接吃得下，
 * 而 onnxruntime-web 的 .wasm 檔本來就得原樣複製。多一層打包只會讓
 * 「線上跑的東西」和「repo 裡看到的東西」對不起來，除錯更難。
 *
 * ── Cloudflare Pages 的 25 MiB 單檔上限 ──────────────────────────────
 * onnxruntime-web 的 WebGPU 執行檔（ort-wasm-simd-threaded.jsep.wasm）
 * 是 27 MB，超過 Pages 的單檔上限，傳不上去。
 *
 * 解法：超過上限的檔案不放進 dist/，改放到 dist-assets/ 由使用者上傳到 R2，
 * 執行期再從那裡載入。反正模型權重（數百 MB 到數 GB）本來就得放 R2，
 * 讓「所有大檔都在 R2、Pages 只放程式碼」是一致而且好理解的分工。
 *
 * 用法：
 *   npm run build                          # 全部放 dist/（本機開發、或沒有單檔限制的主機）
 *   npm run build -- --assets-url=https://assets.example.com/ort/
 *                                          # 大檔改放 dist-assets/，程式指向該網址
 *
 * Cloudflare Pages 設定：
 *   Root directory          web
 *   Build command           npm ci && npm run build -- --assets-url=<你的 R2 網址>
 *   Build output directory  dist
 */
import { cp, mkdir, rm, readdir, writeFile, readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const MAX_PAGES_FILE_BYTES = 25 * 1024 * 1024; // Cloudflare Pages 單檔上限

const argv = process.argv.slice(2);
const assetsUrl = argv.find((a) => a.startsWith('--assets-url='))?.split('=').slice(1).join('=');
// 預設「不」開 cross-origin isolation（WASM 多執行緒的前提）。
//
// 已查證：jsDelivr 的 .mjs 與 .wasm 都有帶
// `cross-origin-resource-policy: cross-origin`，所以 CORP 本身不是問題。
// 保守不開的理由是另一個：ORT 在多執行緒模式下會用 wasm 模組的 URL 去建
// Worker，而跨來源建 Worker 是受限的（transformers.js #1527 有同樣症狀）。
// 這一點尚未實測確認，所以預設走安全的單執行緒。
//
// 影響有限：多執行緒只加速 WASM 後端，WebGPU 不受影響，
// 而這個量測工具要量的正是 WebGPU。
// 要試多執行緒就加 --cross-origin-isolated（會自動改成同源 wasm，
// 屆時得自己處理 25 MiB 單檔上限，見 docs/DEPLOY.md）。
const crossOriginIsolated = argv.includes('--cross-origin-isolated');
// 把 wasm 放在同源。測試要用（不能依賴外部網路），
// 開 cross-origin isolation 時也會自動開啟。
const selfHostFlag = argv.includes('--self-host-wasm');

const root = path.resolve(import.meta.dirname, '..');
const dist = path.join(root, 'dist');
const assetsOut = path.join(root, 'dist-assets');

await rm(dist, { recursive: true, force: true });
await rm(assetsOut, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

// 1. 靜態頁面與原始碼
for (const f of ['index.html', 'bench.html', 'manifest.webmanifest', 'sw.js']) {
  if (existsSync(path.join(root, f))) await cp(path.join(root, f), path.join(dist, f));
}
await cp(path.join(root, 'src'), path.join(dist, 'src'), { recursive: true });
if (existsSync(path.join(root, 'public'))) {
  await cp(path.join(root, 'public'), dist, { recursive: true });
}

// 2. onnxruntime-web 的執行檔。
//    自己 host 而不是走 CDN：PWA 要能離線運作，而且 CDN 一掛整個節點就死。
const ortDist = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
if (!existsSync(ortDist)) {
  console.error('找不到 onnxruntime-web，請先執行 npm ci');
  process.exit(1);
}

const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const ortVersion = JSON.parse(
  await readFile(path.join(root, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8'),
).version;

// wasm 執行檔的來源，三選一：
//   1. --assets-url 指定的位置（例如 R2）
//   2. 沒指定時走 jsDelivr —— 零設定，但與 COOP/COEP 互斥
//   3. --cross-origin-isolated 時強制同源（就得自己解決 25 MiB 問題）
const jsdelivr = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
const selfHostWasm = !assetsUrl && (crossOriginIsolated || selfHostFlag);
const wasmSource = assetsUrl ?? (selfHostWasm ? './ort/' : jsdelivr);

const ortOut = path.join(dist, 'ort');
await mkdir(ortOut, { recursive: true });

const oversized = [];
let inDist = 0, distBytes = 0;

for (const f of await readdir(ortDist)) {
  if (!/\.(wasm|mjs)$/.test(f)) continue;
  if (f.endsWith('.mjs') && !/^ort\.webgpu\.mjs$|^ort-wasm-simd-threaded\./.test(f)) continue;

  const src = path.join(ortDist, f);
  const size = (await stat(src)).size;

  // 走 CDN 時 .wasm 不必進 dist（入口 .mjs 仍要，因為我們是 import 它）
  if (!selfHostWasm && !assetsUrl && f.endsWith('.wasm')) continue;

  if ((assetsUrl || !selfHostWasm) && size > MAX_PAGES_FILE_BYTES) {
    await mkdir(assetsOut, { recursive: true });
    await cp(src, path.join(assetsOut, f));
    oversized.push({ f, size });
  } else {
    await cp(src, path.join(ortOut, f));
    inDist++;
    distBytes += size;
  }
}

// _headers 由建置產生，而不是直接複製 —— COOP/COEP 是條件性的
const headerLines = [
  '# 由 scripts/build.mjs 產生，不要手動改。',
  '',
];
if (crossOriginIsolated) {
  headerLines.push(
    '# cross-origin isolation：WASM 多執行緒（SharedArrayBuffer）的前提。',
    '# 代價是所有跨來源資源都必須帶 CORP 標頭。',
    '/*',
    '  Cross-Origin-Opener-Policy: same-origin',
    '  Cross-Origin-Embedder-Policy: require-corp',
    '',
  );
}
headerLines.push(
  '# wasm 與模型權重都是不可變內容，讓瀏覽器與 CDN 盡量長期保存',
  '/ort/*',
  '  Cache-Control: public, max-age=31536000, immutable',
  '',
  '/model/*',
  '  Cache-Control: public, max-age=31536000, immutable',
  '',
  '# service worker 不能被快取，否則更新推不出去',
  '/sw.js',
  '  Cache-Control: no-cache',
  '',
);
await writeFile(path.join(dist, '_headers'), headerLines.join('\n'));

// 3. 執行期設定：告訴前端去哪裡載 wasm。
//    寫成獨立檔案而不是編進 JS，這樣同一份建置可以換不同的 assets 來源。
await writeFile(
  path.join(dist, 'config.json'),
  JSON.stringify({
    wasmPaths: wasmSource,
    ortVersion,
    crossOriginIsolated,
    builtAt: new Date().toISOString(),
  }, null, 2),
);

// 4. 給 service worker 的預快取清單（只含程式碼，不含模型權重）
const shell = [];
async function walk(dir, base = '') {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const rel = base ? `${base}/${e.name}` : e.name;
    if (e.isDirectory()) await walk(path.join(dir, e.name), rel);
    else if (/\.(html|js|mjs|css|webmanifest|png|svg|json)$/.test(e.name)) shell.push('./' + rel);
  }
}
await walk(dist);
await writeFile(path.join(dist, 'shell.json'), JSON.stringify(shell, null, 2));

// 5. 建置後自檢：dist/ 裡不該有超過 Pages 上限的檔案
const tooBig = [];
async function checkSizes(dir, base = '') {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const rel = base ? `${base}/${e.name}` : e.name;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) await checkSizes(p, rel);
    else {
      const s = (await stat(p)).size;
      if (s > MAX_PAGES_FILE_BYTES) tooBig.push({ rel, s });
    }
  }
}
await checkSizes(dist);

console.log('dist/ 建置完成');
console.log(`  ORT 執行檔 ${inDist} 個（${(distBytes / 1e6).toFixed(1)} MB）`);
console.log(`  app shell ${shell.length} 個檔案`);
console.log(`  wasm 載入來源：${selfHostWasm ? './ort/（同源）' : wasmSource}`);
console.log(`  cross-origin isolation：${crossOriginIsolated ? '開啟（WASM 可多執行緒）' : '關閉（WASM 單執行緒，WebGPU 不受影響）'}`);

if (oversized.length) {
  const mb = oversized.reduce((a, o) => a + o.size, 0) / 1e6;
  console.log(`\ndist-assets/ 有 ${oversized.length} 個超過 Pages 25 MiB 上限的檔案（${mb.toFixed(1)} MB）：`);
  for (const o of oversized) console.log(`  ${o.f}  ${(o.size / 1e6).toFixed(1)} MB`);
  console.log(`\n  請把 dist-assets/ 的內容上傳到 ${assetsUrl}`);
  console.log('  例：npx wrangler r2 object put <bucket>/ort/<檔名> --file dist-assets/<檔名>');
}

if (tooBig.length) {
  // 有給 --assets-url 代表使用者是在做部署建置 —— 那就一定要乾淨，否則直接失敗。
  // 沒給的話是本機開發，讓它能跑比較重要，只要警告清楚就好。
  const deploying = Boolean(assetsUrl);
  const log = deploying ? console.error : console.warn;
  log(`\n${deploying ? '✗' : '⚠'} 有 ${tooBig.length} 個檔案超過 Cloudflare Pages 的 25 MiB 單檔上限：`);
  for (const t of tooBig) log(`    ${t.rel}  ${(t.s / 1e6).toFixed(1)} MB`);
  if (deploying) {
    console.error('\n  部署會失敗。請確認 --assets-url 有正確指向 R2。');
    process.exit(1);
  }
  console.warn('\n  這份建置可以在本機跑，但傳不上 Cloudflare Pages。');
  console.warn('  要部署請加上 --assets-url=<R2 網址>，大檔會改放 dist-assets/ 由你上傳到 R2。');
}
