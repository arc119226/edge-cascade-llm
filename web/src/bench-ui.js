/**
 * bench.html 的 UI 接線。
 *
 * 量測邏輯全在 bench.js，這裡只負責把結果畫出來，
 * 並且盡量用一般人看得懂的話說明每個數字代表什麼。
 *
 * 「說明」包含誠實講出數字不可信的時候 —— K* 撞到搜尋上限、
 * 或是這台機器上小模型根本不是頻寬受限，都要講，不能只報一個漂亮的數字。
 */

import * as ort from '../ort/ort.webgpu.mjs';
import { Pipeline, configureOrt, loadReference, actualAdapterInfo } from './runner.js';
import {
  probeEnvironment, sweepSeqLen, analyseKStar, deriveHopOverhead,
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

/**
 * 要用哪一張顯示卡。
 *
 * 混合顯卡的筆電上，不指定的話瀏覽器給的是內顯 —— 使用者回報
 * 「我的 NVIDIA 沒被偵測到」就是這個原因。預設改成高效能。
 *
 * 一旦跑過量測就不能再改：ORT 的 WebGPU device 每個頁面載入只建立一次，
 * 要換卡只能重新整理。所以量測跑完就把選單鎖起來並說明原因，
 * 而不是讓使用者改了之後拿到一份其實沒換卡的報告。
 */
let gpuPreference = 'high-performance';
let gpuLocked = false;

const adapterLabel = (a) =>
  a ? ([a.vendor, a.architecture, a.device, a.description].filter(Boolean).join(' / ') || '（瀏覽器未提供型號）')
    : '（取不到）';

// ---------------------------------------------------------------- 環境檢查

async function runEnvCheck() {
  environment = await probeEnvironment({ powerPreference: gpuPreference });
  const q = await storageQuota();
  const persist = await requestPersistence();
  const cached = await cacheStats();
  environment.storagePersisted = persist.granted;

  const p = environment.providers;
  const rows = [];

  if (p.webgpu) {
    const a = p.adapter;
    rows.push(['WebGPU', `<span class="ok">可用</span>`]);
    rows.push(['顯示卡', adapterLabel(a) +
      `<span class="dim">　量測會跑在這一張</span>`]);
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

  // 有兩張卡就兩張都列出來 —— 使用者才看得出報告裡的是哪一張。
  if (p.adapters?.distinct) {
    rows.push(['另一張顯示卡',
      `<span class="dim">${adapterLabel(p.adapters.other)}　未使用</span>`]);
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

  updateGpuPicker(p);
  $('btn-bench').disabled = false;
}

/** 只有真的有兩張卡才顯示選單，不然是多一個沒意義的選項。 */
function updateGpuPicker(providers) {
  const box = $('gpu-pick');
  const sel = $('gpu-pref');
  const note = $('gpu-pick-note');
  if (!providers.webgpu || !providers.adapters?.distinct) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  sel.value = gpuPreference;
  sel.disabled = gpuLocked;
  note.innerHTML = gpuLocked
    ? '已經量測過了。要換另一張卡請<strong>重新整理頁面</strong>再選 —— '
      + '顯示卡在第一次量測時就固定住了，現在改選不會真的換卡。'
    : '偵測到你有兩張顯示卡。改選之後會重新檢查，然後才開始量測。';
}

$('gpu-pref').onchange = async () => {
  if (gpuLocked) return;
  gpuPreference = $('gpu-pref').value;
  $('env-out').innerHTML = '<p class="dim">重新檢查中…</p>';
  try {
    await runEnvCheck();
  } catch (e) {
    $('env-out').innerHTML = `<p class="bad">檢查失敗：${e}</p>`;
  }
};

$('btn-env').onclick = async () => {
  const btn = $('btn-env');
  btn.disabled = true;
  btn.textContent = '檢查中…';
  try {
    await runEnvCheck();
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

    // 參考 logits 存在二進位旁檔裡，reference.json 只有中繼資料。
    // 一定要走 loadReference()：直接 fetch 會拿到沒有 logits 的物件，
    // 然後在比對時炸成看不出根因的 TypeError（這個錯真的發生過）。
    //
    // preferNative 是刻意的，和 test/browser.test.mjs 一致：這裡要驗的是
    // 「瀏覽器算得對不對」，對照組該是原生 ORT 跑同一份 ONNX 的輸出，
    // 而不是 PyTorch fp32 真值 —— 後者量的是權重量化品質，是另一個問題。
    const reference = await loadReference(baseUrl, { preferNative: true });

    // wasmPaths 由建置時產生的 config.json 決定。
    const cfg = await fetch('./config.json').then((r) => r.json()).catch(() => ({}));
    const ep = environment?.providers?.webgpu ? 'webgpu' : 'wasm';
    const ortInfo = configureOrt(ort, {
      wasmPaths: cfg.wasmPaths ?? '../ort/',
      // 用偵測時實際生效的那個值：detectProviders 若退回過預設 adapter，
      // 它會回報 null，這裡就不要再硬指定，免得兩邊挑到不同的卡。
      powerPreference: ep === 'webgpu' ? environment?.providers?.powerPreference : undefined,
    });
    gpuLocked = true;   // ORT 的 device 即將建立，之後換卡只能重新整理
    if (environment?.providers) updateGpuPicker(environment.providers);

    // --- 主流水線（用 manifest 指定的切分數）---
    say(`載入 ${manifest.num_shards} 個模型分段（第一次要下載，請耐心等）…`, 8);
    const pipeline = new Pipeline(ort, manifest, baseUrl, { ep, scheme: 'none' });
    const loadStats = await pipeline.load();
    const loadedBytes = loadStats.reduce((a, s) => a + s.bytes, 0);

    // --- Q1 + Q2：一次掃描，兩個答案 ---
    //
    // 舊版分開量，而且 Q1 是用「把 4 段收成 2 段」製造不同 hop 數 ——
    // 那樣收起來的流水線會跳過中間的層，時間差裡混了「少幾個 hop」和
    // 「少算幾層」，兩者無法分離。現在改成固定完整流水線、只掃序列長度，
    // 從每個 shard 的擬合截距導出固定成本。順帶少載一整條流水線，快不少。
    say('掃描序列長度（同時量平行視窗與每段的固定成本）…', 30);
    const points = await sweepSeqLen(pipeline, manifest, {
      maxK: 64,
      repeats: 3,
      onProgress: ({ done, total, k }) =>
        say(`掃描序列長度 K=${k}（${done}/${total}）…`, 30 + (done / total) * 45),
    });
    const kstar = analyseKStar(points);
    const hop = deriveHopOverhead(points, manifest);

    // --- Q11：EP 之間的數值差異 ---
    say('比對不同執行後端的數值差異…', 80);
    const eps = environment?.providers?.webgpu ? ['wasm', 'webgpu'] : ['wasm'];
    const epCompare = await compareProviders(ort, manifest, baseUrl, reference, eps);

    say('整理結果…', 95);
    report = buildReport({
      environment,
      ort: {
        version: ort.env.versions?.web ?? null,
        ...ortInfo,
        executionProvider: ep,
        // ORT 自己建 device 時挑到的那張卡。這是唯一能證明
        // 「報告寫的卡 == 真的在算的卡」的欄位。
        actualAdapter: ep === 'webgpu' ? actualAdapterInfo(ort) : null,
      },
      model: {
        name: manifest.model, layers: manifest.num_layers,
        hiddenSize: manifest.hidden_size, shards: manifest.num_shards,
        dtype: manifest.dtype ?? null,
        loadedBytes,
        loadStats,
      },
      reference: { source: reference.source, dtype: reference.logits_dtype ?? null },
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

const ms = (v, digits = 1) => (v == null ? '—' : v.toFixed(digits));

function renderResults(r) {
  const k = r.kStar;
  const h = r.hopOverhead;
  const out = [];

  out.push('<h2>載入的權重</h2>');
  out.push(`<p><strong>${fmtBytes(r.model.loadedBytes)}</strong>
    <span class="dim">　${r.model.shards} 段、共 ${r.model.layers} 層${
      r.model.dtype ? `、權重 ${r.model.dtype}` : ''}</span></p>`);

  out.push('<h2>平行視窗 K*</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    一次算多個位置時，前面幾個幾乎是「免費」的（權重只讀一次）。
    K* 就是免費到哪裡為止 —— 超過之後每多算一個位置就真的要多花時間。
    這決定投機解碼一次能猜幾個字。</p>`);
  out.push(`<p><strong>K* = ${k.kStar}</strong>
    <span class="dim">在 K=${k.kStar} 時每個位置只要 ${ms(k.msPerPositionAtKStar, 2)} ms，
    比 K=1 的 ${ms(k.msPerPositionAtK1, 2)} ms 快
    ${ms(k.speedupAtKStar, 1)} 倍</span></p>`);

  // 這兩段警告比上面那個數字重要：不講的話，一個被截斷或被 dispatch 主導的
  // K* 看起來和真正的 roofline 轉折點一模一樣。
  if (k.censored) {
    out.push(`<p class="warn" style="font-size:14px">
      ⚠ 每個位置的耗時一直降到 K=${k.maxKTested}（掃描上限）都<strong>還在降</strong>。
      這代表真正的轉折點在 ${k.maxKTested} 以上，
      <strong>${k.kStar} 是量到的上限，不是轉折點</strong>。</p>`);
  }
  if (k.fit) {
    out.push(`<p class="dim" style="font-size:14px">
      擬合：總耗時 ≈ <code>${ms(k.fit.fixedMs)} ms + ${ms(k.fit.msPerPositionSlope, 2)} ms × 位置數</code>
      （r² = ${ms(k.fit.r2, 3)}）。
      其中固定的那一項在 K=1 時就佔了 ${k.fixedFractionAtK1 == null ? '—' :
        (k.fixedFractionAtK1 * 100).toFixed(0)}%。</p>`);
  }
  if (k.regime === 'dispatch-bound') {
    out.push(`<p class="warn" style="font-size:14px">
      ⚠ 固定成本佔了大半，代表這台機器上的瓶頸是<strong>呼叫開銷</strong>
      （每次 session.run 的固定成本），而不是讀權重的頻寬。
      這是因為測試模型只有 135M —— 權重太小，讀權重根本不是瓶頸。
      所以這個 K* 反映的是「呼叫成本被攤掉的速度」，
      <strong>不能直接外推到 70B 那種大模型</strong>。
      要回答那一題需要換 1B 以上的模型重測。</p>`);
  }

  out.push('<table><tr><th>K</th><th class="num">總耗時 (ms)</th><th class="num">每位置 (ms)</th></tr>' +
    k.points.map((p) => `<tr><td>${p.k}</td><td class="num">${p.totalMs.toFixed(1)}</td>` +
      `<td class="num">${p.msPerPosition.toFixed(2)}</td></tr>`).join('') + '</table>');

  out.push('<h2>每個節點交接的固定成本</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    每多一個節點，就算網路零延遲也要多花一筆固定時間：張量搬進搬出、
    呼叫一次推論、量化編解碼。注意這是<strong>每一段</strong>都要付一次，
    不是每個交接 —— 切成 4 段要付 4 次。節點越多累積越兇，
    所以它決定了「模型最多該切幾段」。
    <br>作法：固定整條流水線不變，只改一次餵幾個位置，對每一段擬合
    <code>耗時 = 固定成本 + 斜率 × 位置數</code>，截距就是固定成本。</p>`);
  if (h.fixedMsPerShard != null) {
    out.push(`<p><strong>約 ${ms(h.fixedMsPerShard)} ms／段</strong>
      <span class="dim">　整條流水線固定成本 ${ms(h.pipelineFixedMs)} ms，
      其中 ${ms(h.glueMs)} ms 在 JS 這一層（建張量、量化、排程）</span></p>`);
  }
  out.push('<table><tr><th>分段</th><th>負責的層</th><th class="num">固定成本 (ms)</th>' +
    '<th class="num">每位置 (ms)</th><th class="num">r²</th></tr>' +
    h.perShard.map((x) => `<tr><td>${x.index}</td>` +
      `<td class="dim">${x.layers ? x.layers.join('–') : '—'}</td>` +
      `<td class="num">${ms(x.fixedMs)}</td>` +
      `<td class="num">${ms(x.slopeMsPerPosition, 2)}</td>` +
      `<td class="num">${ms(x.r2, 3)}</td></tr>`).join('') + '</table>');
  out.push(`<p class="dim" style="font-size:13px">
    r² 越接近 1 代表這條直線越貼合實測點。這裡量的是<strong>本機</strong>成本 ——
    真實部署還要再加上網路往返，那一段由 <code>bench/model.py</code> 另外算。</p>`);

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

  if (r.ort.actualAdapter) {
    out.push(`<p class="dim" style="font-size:13px">
      實際執行的顯示卡：<code>${adapterLabel(r.ort.actualAdapter)}</code>
      （由 ORT 回報，不是我們猜的）</p>`);
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
