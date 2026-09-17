/**
 * 在真實瀏覽器裡驗證 shard 流水線。
 *
 * 這支測試回答的是：「把模型切開、在瀏覽器裡串起來跑，結果還對嗎？」
 * 對照組是 spike/export_shards.py 產生的 reference.json —— 那是未切分的
 * PyTorch 模型算出來的 logits。
 *
 * 限制：開發容器裡 navigator.gpu 不存在（headless shell / 完整 chromium /
 * Xvfb + headed / swiftshader 全試過都沒有），所以這裡只驗得了 WASM EP。
 * WebGPU EP 的正確性要靠 bench.html 在真實裝置上跑 —— 那是刻意的分工，
 * 不是偷懶：能在 CI 驗的就驗，驗不了的就老實標示為未驗證。
 *
 * 前置：
 *   npm ci
 *   npm run build
 *   node scripts/prepare-model.mjs      # 匯出小模型放到 dist/model/
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { chromium } from 'playwright';

/**
 * 找一個可用的 Chromium。
 *
 * Playwright 的 npm 套件版本會綁定特定的瀏覽器 build 編號，而映像檔裡
 * 預裝的可能是別的版本。與其下載一份（慢、而且在受限網路下會失敗），
 * 不如直接用預裝的那個。
 */
function findChromium() {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH;
  const base = process.env.PLAYWRIGHT_BROWSERS_PATH ?? '/opt/pw-browsers';
  try {
    for (const dir of readdirSync(base)) {
      if (!dir.startsWith('chromium-')) continue;
      const exe = path.join(base, dir, 'chrome-linux', 'chrome');
      if (existsSync(exe)) return exe;
    }
  } catch { /* 沒有這個目錄就走 Playwright 預設 */ }
  return undefined;
}

const root = path.resolve(import.meta.dirname, '..');
const dist = path.join(root, 'dist');
const modelDir = path.join(dist, 'model');
const PORT = 8137;

function serve() {
  // 用 http-server 而不是自己寫：需要正確的 wasm MIME、range request、
  // 以及 COOP/COEP 標頭才能重現線上環境。
  const p = spawn('npx', [
    'http-server', dist, '-p', String(PORT), '-c-1', '--silent',
    '--header', 'Cross-Origin-Opener-Policy: same-origin',
    '--header', 'Cross-Origin-Embedder-Policy: require-corp',
  ], { cwd: root, stdio: 'ignore' });
  return p;
}

async function waitForServer(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch { /* 還沒起來 */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`伺服器沒有在 ${timeoutMs}ms 內起來`);
}

test('瀏覽器中的 shard 流水線與未切分模型數值等價（WASM EP）', async (t) => {
  if (!existsSync(dist)) {
    t.skip('尚未建置。請先執行 npm run build');
    return;
  }
  if (!existsSync(path.join(modelDir, 'manifest.json'))) {
    t.skip('找不到測試模型。請先執行 node scripts/prepare-model.mjs');
    return;
  }

  const server = serve();
  let browser;
  try {
    await waitForServer(`http://127.0.0.1:${PORT}/bench.html`);
    // WebGPU 在這個容器裡不存在，所以這裡只驗 WASM EP（見檔頭說明）。
    const executablePath = findChromium();
    browser = await chromium.launch({
      executablePath,
      args: ['--no-sandbox'],
    });
    const page = await browser.newPage();

    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('requestfailed', (r) => errors.push(`request failed: ${r.url()}`));
    page.on('response', (r) => { if (r.status() >= 400) errors.push(`HTTP ${r.status()}: ${r.url()}`); });

    await page.goto(`http://127.0.0.1:${PORT}/bench.html`, { waitUntil: 'load' });

    const result = await page.evaluate(async () => {
      const ort = await import('./ort/ort.webgpu.mjs');
      const { Pipeline, configureOrt, compareToReference, loadReference } =
        await import('./src/runner.js');

      const cfg = await (await fetch('./config.json')).json();
      configureOrt(ort, { wasmPaths: cfg.wasmPaths, numThreads: 1 });
      // 單緒是刻意的：測正確性不測速度，執行緒數會讓結果更難重現。

      const manifest = await (await fetch('./model/manifest.json')).json();
      // 拿 native.bin（原生 ORT 跑同一份 ONNX 的輸出）當對照組，不是 fp32 真值。
      // 這裡要驗的是「瀏覽器算得對不對」，權重量化造成的差異是另一回事。
      const reference = await loadReference('./model/', { preferNative: true });

      // scheme: 'none' —— 這裡驗的是「切分 + 瀏覽器執行」是否正確，
      // 激活值量化的影響已經在 M4 單獨量過了，混在一起會分不清誰造成的差異。
      const pipe = new Pipeline(ort, manifest, './model/', { ep: 'wasm', scheme: 'none' });
      await pipe.load();

      const seq = reference.input_ids[0].length;
      const ids = BigInt64Array.from(reference.input_ids.flat().map(BigInt));
      const out = await pipe.run(ids, seq);
      const cmp = compareToReference(out.logits, out.dims, reference);
      await pipe.dispose();

      return {
        shards: manifest.num_shards,
        layers: manifest.num_layers,
        dtype: manifest.dtype ?? 'fp32',
        referenceSource: reference.source,
        wasmPaths: cfg.wasmPaths,
        dims: out.dims,
        hops: out.hops.map((h) => ({ layers: h.layers, ms: Math.round(h.computeMs) })),
        ...cmp,
      };
    });

    // 回歸測試：bench.html 走的是 compareProviders() 這條路徑。
    // 它曾經因為直接 fetch('reference.json') —— 那份 JSON 改成只放中繼資料之後
    // 就沒有 logits 欄位了 —— 而炸成 `undefined is not iterable`，
    // 兩個 execution provider 都失敗。現有測試只驗 Pipeline + compareToReference，
    // 沒有涵蓋量測頁真正在跑的組裝方式，所以那個錯一路漏到使用者手上。
    const epCompare = await page.evaluate(async () => {
      const ort = await import('./ort/ort.webgpu.mjs');
      const { loadReference } = await import('./src/runner.js');
      const { compareProviders } = await import('./src/bench.js');

      const manifest = await (await fetch('./model/manifest.json')).json();
      const reference = await loadReference('./model/', { preferNative: true });
      const results = await compareProviders(ort, manifest, './model/', reference, ['wasm']);
      return results[0];
    });

    console.log(`  compareProviders(wasm) ok=${epCompare.ok}` +
      (epCompare.ok ? ` argmax ${epCompare.argmaxAgree}/${epCompare.argmaxTotal}` : ''));
    assert.equal(
      epCompare.ok, true,
      `量測頁的後端比對失敗：${epCompare.error}\n` +
      '（reference.json 只有中繼資料，必須用 loadReference() 載入二進位 logits）',
    );

    console.log(`  模型切成 ${result.shards} 段（共 ${result.layers} 層），權重 ${result.dtype}`);
    console.log(`  對照組 ${result.referenceSource}`);
    console.log(`  wasm 來源 ${result.wasmPaths}`);
    console.log(`  logits 形狀 ${result.dims.join('x')}`);
    console.log(`  max abs diff  ${result.maxAbsDiff.toExponential(3)}`);
    console.log(`  argmax 一致   ${result.argmaxAgree}/${result.argmaxTotal}`);

    assert.equal(errors.length, 0, `瀏覽器有錯誤：\n${errors.join('\n')}`);
    assert.equal(
      result.argmaxAgree, result.argmaxTotal,
      'argmax 不一致 —— 這不是精度問題，是切分或執行邏輯有錯',
    );
    // 1e-3 與 spike/verify_shards.py 同一個門檻，方便兩邊對照。
    // 實測 ORT-Web WASM 約 2e-4，比原生 ORT 的 7.7e-5 大一個檔次 ——
    // 數值行為是跟著 execution provider 走的，見 docs/03-open-questions.md Q11。
    // 對照組存成 fp16，本身就有約 1e-3 的相對解析度；logits 量級在數十，
    // 所以絕對容差要放到 0.1。這不是放水 —— argmax 全對才是真正的驗收條件，
    // 而那一項在上面已經用嚴格相等檢查過了。
    assert.ok(
      result.maxAbsDiff < 0.1,
      `logits 偏差 ${result.maxAbsDiff.toExponential(3)} 超過 fp16 對照組的容差 0.1`,
    );
  } finally {
    await browser?.close();
    server.kill();
  }
});

test('量化方案的 JS 實作與線路格式定義一致', async (t) => {
  if (!existsSync(dist)) {
    t.skip('尚未建置');
    return;
  }
  const { SCHEMES, wireBytes } = await import(path.join(dist, 'src', 'quant.js'));

  // 線路位元數必須和 docs/01-architecture.md §4.2 敲定的一致
  assert.equal(SCHEMES.wire.wireBits, 8.24, '預設線路格式的等效位元數不對');
  assert.equal(SCHEMES['group-64'].wireBits, 8);
  assert.equal(SCHEMES.none.wireBits, 32);

  // 傳輸量換算：1000 個值用預設格式 = 1030 bytes
  assert.equal(wireBytes(1000, 'wire'), Math.ceil((1000 * 8.24) / 8));

  // 量化必須真的改變數值（不能因為某個分支寫錯而變成 no-op）
  const d = 64;
  const x = new Float32Array(d * 4);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i) * 3;
  x[5] = 900; // 離群
  const q = SCHEMES.wire.fn(x, d);
  assert.notDeepEqual(Array.from(q), Array.from(x), '量化沒有作用');
  // 但也不能面目全非
  let maxRel = 0;
  for (let i = 0; i < x.length; i++) {
    if (Math.abs(x[i]) > 0.1) maxRel = Math.max(maxRel, Math.abs(q[i] - x[i]) / Math.abs(x[i]));
  }
  assert.ok(maxRel < 0.25, `量化誤差過大：最大相對誤差 ${(maxRel * 100).toFixed(1)}%`);
});

test('K* 與每段固定成本的分析：會誠實回報被截斷與 dispatch 主導', async (t) => {
  if (!existsSync(dist)) {
    t.skip('尚未建置');
    return;
  }
  const { analyseKStar, deriveHopOverhead } =
    await import(path.join(dist, 'src', 'bench.js'));

  // 造一組「總耗時 = 40 + 2k」的完美資料：每位置耗時一路遞減，
  // 永遠不會打平 —— 正是實測踩到的情況（K* 撞到掃描上限）。
  const ks = [1, 2, 4, 8, 16, 32];
  const points = ks.map((k) => ({
    k,
    totalMs: 40 + 2 * k,
    msPerPosition: (40 + 2 * k) / k,
    // 4 段，每段固定成本 10 ms、每位置 0.5 ms
    shardMs: [0, 1, 2, 3].map(() => 10 + 0.5 * k),
  }));

  const k = analyseKStar(points);
  assert.equal(k.kStar, 32, 'K* 應該是最後一個掃描點');
  assert.equal(k.censored, true, '每位置耗時一路在降，必須標示為被截斷');
  assert.ok(Math.abs(k.fit.fixedMs - 40) < 1e-6, `固定成本擬合錯誤：${k.fit.fixedMs}`);
  assert.ok(Math.abs(k.fit.msPerPositionSlope - 2) < 1e-6);
  assert.ok(Math.abs(k.fit.r2 - 1) < 1e-9, '完美直線的 r² 應該是 1');
  // k=1 時總耗時 42，固定成本 40 -> 95%，遠超過一半
  assert.equal(k.regime, 'dispatch-bound');

  const manifest = { shards: [0, 1, 2, 3].map((i) => ({ index: i, layers: [i * 2, i * 2 + 1] })) };
  const h = deriveHopOverhead(points, manifest);
  assert.equal(h.method, 'per-shard-intercept');
  assert.equal(h.perShard.length, 4);
  assert.ok(Math.abs(h.fixedMsPerShard - 10) < 1e-6, `每段固定成本錯誤：${h.fixedMsPerShard}`);
  assert.ok(Math.abs(h.sumShardFixedMs - 40) < 1e-6);
  // 這組合成資料裡 4 段的截距剛好加總成整條流水線的截距，所以 JS 那一層是 0
  assert.ok(Math.abs(h.glueMs) < 1e-6, `glueMs 應該接近 0，得到 ${h.glueMs}`);

  // 對照組：真的有轉折點時不能誤報成被截斷
  const flat = [1, 2, 4, 8].map((kk) => ({
    k: kk, totalMs: 10 * kk, msPerPosition: 10, shardMs: [10 * kk],
  }));
  assert.equal(analyseKStar(flat).censored, false, '打平的曲線不該被當成還在降');
});
