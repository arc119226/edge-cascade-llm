/**
 * bench.html 的 UI 接線。
 *
 * 量測邏輯全在 bench.js，這裡只負責把結果畫出來，
 * 並且盡量用一般人看得懂的話說明每個數字代表什麼。
 */

import * as ort from '../ort/ort.webgpu.mjs';
import { Pipeline, configureOrt } from './runner.js';
import {
  probeEnvironment, measureKStar, measureHopOverhead,
  compareProviders, buildReport,
} from './bench.js';
import { storageQuota, requestPersistence, cacheStats } from './cache.js';

const $ = (id) => document.getElementById(id);
const fmtBytes = (b) =>
  b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB'
  : b >= 1e6 ? (b / 1e6).toFixed(0) + ' MB'
  : (b / 1e3).toFixed(0) + ' KB';

let environment = null;
let report = null;

// ---------------------------------------------------------------- 環境檢查

$('btn-env').onclick = async () => {
  const btn = $('btn-env');
  btn.disabled = true;
  btn.textContent = '檢查中…';
  try {
    environment = await probeEnvironment();
    const q = await storageQuota();
    const persist = await requestPersistence();
    const cached = await cacheStats();
    environment.storagePersisted = persist.granted;

    const p = environment.providers;
    const rows = [];

    if (p.webgpu) {
      const a = p.adapter;
      rows.push(['WebGPU', `<span class="ok">可用</span>`]);
      rows.push(['顯示卡', [a.vendor, a.architecture, a.description].filter(Boolean).join(' / ') || '（未提供）']);
      rows.push(['單一緩衝區上限', fmtBytes(a.maxBufferSize) +
        `<span class="dim">　決定單一層的權重能不能塞下</span>`]);
      rows.push(['半精度 shader-f16', a.shaderF16
        ? '<span class="ok">支援</span>'
        : '<span class="warn">不支援</span><span class="dim">　會慢一些</span>']);
    } else {
      rows.push(['WebGPU', `<span class="bad">不可用</span>`]);
      rows.push(['原因', `<span class="dim">${p.reason ?? '未知'}</span>`]);
      rows.push(['影響', '<span class="dim">還是可以量測，但會走比較慢的 WASM 模式</span>']);
    }

    rows.push(['CPU 核心數', environment.hardwareConcurrency ?? '（未提供）']);
    rows.push(['多執行緒', environment.crossOriginIsolated
      ? '<span class="ok">已啟用</span>'
      : '<span class="ok">未啟用（正常）</span>'
        + '<span class="dim">　這是預設設定，只影響 WASM 後端，不影響 WebGPU</span>']);
    if (q) {
      rows.push(['可用儲存空間', `${fmtBytes(q.free)} <span class="dim">／ 共 ${fmtBytes(q.quota)}</span>`]);
    }
    rows.push(['儲存是否持久', persist.granted
      ? '<span class="ok">是</span>'
      : '<span class="warn">否</span><span class="dim">　空間不足時快取可能被清掉</span>']);
    if (cached.available && cached.files) {
      rows.push(['已快取權重', `${cached.files} 個檔案　${fmtBytes(cached.bytes)}`]);
    }

    $('env-out').innerHTML =
      '<table style="margin-top:12px">' +
      rows.map(([k, v]) => `<tr><th style="width:38%">${k}</th><td>${v}</td></tr>`).join('') +
      '</table>';
    $('btn-bench').disabled = false;
    btn.textContent = '重新檢查';
  } catch (e) {
    $('env-out').innerHTML = `<p class="bad">檢查失敗：${e}</p>`;
    btn.textContent = '重試';
  } finally {
    btn.disabled = false;
  }
};

// ---------------------------------------------------------------- 效能量測

$('btn-bench').onclick = async () => {
  const btn = $('btn-bench');
  btn.disabled = true;
  $('bench-out').innerHTML = '';
  const prog = $('bench-progress');
  const say = (msg, pct) => {
    prog.innerHTML = `<div class="dim" style="font-size:14px">${msg}</div>` +
      (pct == null ? '' : `<progress value="${pct}" max="100"></progress>`);
  };

  try {
    const baseUrl = $('model-url').value.trim();
    say('載入模型清單…', 2);
    const manifest = await (await fetch(baseUrl + 'manifest.json')).json();
    const reference = await (await fetch(baseUrl + 'reference.json')).json();

    // wasmPaths 由建置時產生的 config.json 決定。
    // 超過 Cloudflare Pages 25 MiB 上限的 wasm 會放在 R2，這裡指過去。
    const cfg = await fetch('./config.json').then((r) => r.json()).catch(() => ({}));
    const ortInfo = configureOrt(ort, { wasmPaths: cfg.wasmPaths ?? '../ort/' });
    const ep = environment?.providers?.webgpu ? 'webgpu' : 'wasm';

    // --- 主流水線（用 manifest 指定的切分數）---
    say(`載入 ${manifest.num_shards} 個模型分段（第一次要下載，請耐心等）…`, 8);
    const pipeline = new Pipeline(ort, manifest, baseUrl, { ep, scheme: 'none' });
    const loadStats = await pipeline.load();
    const loadedBytes = loadStats.reduce((a, s) => a + s.bytes, 0);

    // --- Q2：K* ---
    say('量測平行視窗 K*（看一次算多個位置划不划算）…', 30);
    const kstar = await measureKStar(pipeline, manifest, { maxK: 32, repeats: 3 });

    // --- Q1：每 hop 固定開銷 ---
    // 用同一份 manifest 的不同「前 N 段」子集合來製造不同 hop 數，
    // 避免要求使用者下載多份切分不同的模型。
    say('量測每個節點交接的固定成本…', 55);
    const subsets = [];
    for (const n of [2, manifest.num_shards].filter((v, i, a) =>
      v >= 2 && v <= manifest.num_shards && a.indexOf(v) === i)) {
      subsets.push({
        shards: n,
        manifest,
        pipeline: n === manifest.num_shards
          ? pipeline
          : await (async () => {
              const sub = { ...manifest, num_shards: n, shards: collapse(manifest.shards, n) };
              const pl = new Pipeline(ort, sub, baseUrl, { ep, scheme: 'none' });
              await pl.load();
              return pl;
            })(),
      });
    }
    const hop = await measureHopOverhead(subsets, { seq: 8, repeats: 5 });

    // --- Q11：EP 之間的數值差異 ---
    say('比對不同執行後端的數值差異…', 78);
    const eps = environment?.providers?.webgpu ? ['wasm', 'webgpu'] : ['wasm'];
    const epCompare = await compareProviders(ort, manifest, baseUrl, reference, eps);

    say('整理結果…', 95);
    report = buildReport({
      environment,
      ort: { version: ort.env.versions?.web ?? null, ...ortInfo, executionProvider: ep },
      model: {
        name: manifest.model, layers: manifest.num_layers,
        hiddenSize: manifest.hidden_size, shards: manifest.num_shards,
        loadedBytes,
      },
      kStar: kstar,
      hopOverhead: hop,
      providerComparison: epCompare,
    });

    renderResults(report);
    $('report-json').textContent = JSON.stringify(report, null, 2);
    $('report-card').hidden = false;
    say('完成。', 100);
  } catch (e) {
    $('bench-out').innerHTML =
      `<p class="bad">量測失敗：${e}</p>` +
      `<p class="dim" style="font-size:13px">把這段訊息貼回來也有幫助 —— 失敗本身就是有用的資料。</p>`;
    console.error(e);
  } finally {
    btn.disabled = false;
  }
};

/** 把 N 個 shard 併成 n 個（只改 manifest，不重新匯出模型）。 */
function collapse(shards, n) {
  // 只取前 n-1 個 shard 加上最後一個 —— 這樣層數會不完整，
  // 所以僅用於量「交接成本」，不用於正確性驗證。
  return [...shards.slice(0, n - 1), shards[shards.length - 1]];
}

function renderResults(r) {
  const k = r.kStar;
  const h = r.hopOverhead;
  const out = [];

  out.push('<h2>平行視窗 K*</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    一次算多個位置時，前面幾個幾乎是「免費」的（權重只讀一次）。
    K* 就是免費到哪裡為止 —— 超過之後每多算一個位置就真的要多花時間。
    這決定投機解碼一次能猜幾個字。</p>`);
  out.push(`<p><strong>K* = ${k.kStar}</strong>
    <span class="dim">在 K=${k.kStar} 時每個位置只要 ${k.msPerPositionAtKStar.toFixed(2)} ms，
    比 K=1 的 ${k.msPerPositionAtK1.toFixed(2)} ms 快
    ${k.speedupAtKStar.toFixed(1)} 倍</span></p>`);
  out.push('<table><tr><th>K</th><th class="num">總耗時 (ms)</th><th class="num">每位置 (ms)</th></tr>' +
    k.points.map((p) => `<tr><td>${p.k}</td><td class="num">${p.totalMs.toFixed(1)}</td>` +
      `<td class="num">${p.msPerPosition.toFixed(2)}</td></tr>`).join('') + '</table>');

  out.push('<h2>每個節點交接的固定成本</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    每多經過一個節點，就算網路零延遲也要多花一筆固定時間。
    節點越多這筆成本累積越兇，所以它決定了「模型最多該切幾段」。</p>`);
  if (h.overheadMsPerHop != null) {
    out.push(`<p><strong>約 ${h.overheadMsPerHop.toFixed(1)} ms／次交接</strong></p>`);
  }
  out.push('<table><tr><th>分段數</th><th>交接次數</th><th class="num">總耗時 (ms)</th></tr>' +
    h.rows.map((x) => `<tr><td>${x.shards}</td><td>${x.hops}</td>` +
      `<td class="num">${x.totalMs.toFixed(1)}</td></tr>`).join('') + '</table>');

  out.push('<h2>不同執行後端的數值差異</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    同一個模型用 WASM 和 WebGPU 算，結果會有極小的差異。
    差異太大的話，不同裝置組成的網路可能算出不一致的結果。</p>`);
  out.push('<table><tr><th>後端</th><th class="num">耗時 (ms)</th>' +
    '<th class="num">與參考值最大差</th><th class="num">選字一致率</th></tr>' +
    r.providerComparison.map((p) => p.ok
      ? `<tr><td>${p.ep}</td><td class="num">${p.ms.toFixed(0)}</td>` +
        `<td class="num">${p.maxAbsDiff.toExponential(2)}</td>` +
        `<td class="num">${p.argmaxAgreePct.toFixed(2)}%</td></tr>`
      : `<tr><td>${p.ep}</td><td colspan="3" class="bad">失敗：${p.error}</td></tr>`,
    ).join('') + '</table>');

  const cross = r.providerComparison.find((p) => p.maxDiffVsFirstEp != null);
  if (cross) {
    out.push(`<p class="dim" style="font-size:14px">
      兩個後端之間的最大差異：<code>${cross.maxDiffVsFirstEp.toExponential(2)}</code></p>`);
  }

  $('bench-out').innerHTML = out.join('');
}

// ---------------------------------------------------------------- 匯出

$('btn-copy').onclick = async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    $('copy-msg').textContent = '已複製 ✓';
  } catch {
    $('copy-msg').textContent = '複製失敗，請手動選取下方文字';
  }
  setTimeout(() => ($('copy-msg').textContent = ''), 3000);
};

$('btn-download').onclick = () => {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `edge-cascade-bench-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
};

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}
