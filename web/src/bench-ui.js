/**
 * bench.html 的 UI 接線。
 *
 * 量測邏輯全在 bench.js，這裡只負責把結果畫出來，
 * 並且盡量用一般人看得懂的話說明每個數字代表什麼。
 *
 * 「說明」包含誠實講出數字不可信的時候：曲線上根本沒有轉折點、
 * 這台機器的瓶頸不是我們想量的那個、WASM 那欄只是單執行緒 ——
 * 這些不講，讀的人就會從一個漂亮的數字得出錯誤結論。
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
const ms = (v, digits = 1) => (v == null ? '—' : v.toFixed(digits));

let environment = null;
let report = null;

/**
 * 量測要用的顯示卡偏好。
 *
 * ⚠ 這是個「盡量」而已，網頁改不了實際用哪張卡。Chrome 在 Windows 上
 * 只用啟動時分配到的那張顯示卡（筆電通常是內顯），requestAdapter 的
 * powerPreference 依官方文件「doesn't have any impact」；ORT 1.30 也
 * 把這個設定要來的 adapter 丟掉了（見 runner.js configureOrt 的說明）。
 *
 * 所以這裡不做成下拉選單 —— 給使用者一個選不動的選項，比不給更糟。
 * 真正能換卡的做法寫在環境表格裡，那是瀏覽器與作業系統層級的設定。
 */
const GPU_PREFERENCE = 'high-performance';

const adapterLabel = (a) =>
  a ? ([a.vendor, a.architecture, a.device, a.description].filter(Boolean).join(' / ') || '（瀏覽器未提供型號）')
    : '（取不到）';

/** 只有一張卡可用時，告訴使用者怎麼在瀏覽器／系統層級換。 */
const SINGLE_ADAPTER_HELP = `
  <span class="dim">Chrome 一次只用得到一張顯示卡，而且是它啟動時分配到的那張
  （筆電上通常是內顯）。如果你還有一張獨立顯卡，要換得在瀏覽器或系統層級設定 ——
  網頁這邊改不了。</span>
  <ol class="dim" style="font-size:13px; margin:6px 0 0; padding-left:20px">
    <li>網址列輸入 <code>chrome://flags/#force-high-performance-gpu</code>
        → 設成 Enabled → 重啟 Chrome</li>
    <li>或 Windows 設定 → 系統 → 顯示 → 顯示卡 → 加入 <code>chrome.exe</code>
        → 選項 → 高效能</li>
    <li>改完用 <code>chrome://gpu</code> 確認哪一張標成 active</li>
  </ol>`;

// ---------------------------------------------------------------- 環境檢查

async function runEnvCheck() {
  environment = await probeEnvironment({ powerPreference: GPU_PREFERENCE });
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
    if (p.adapters?.distinct) {
      rows.push(['另一張顯示卡',
        `<span class="dim">${adapterLabel(p.adapters.other)}　未使用</span>`]);
    } else if (p.singleAdapter) {
      // 兩種偏好拿到同一張卡（連 limits 都相同）—— 瀏覽器只給得出一張。
      rows.push(['只偵測到一張顯示卡', SINGLE_ADAPTER_HELP]);
    }
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
    : '<span class="warn">否</span>'
      + '<span class="dim">　空間不足時快取可能被清掉。把這頁加入書籤可提高瀏覽器授予的機率</span>']);
  if (cached.available && cached.files) {
    rows.push(['已快取權重', `${cached.files} 個檔案　${fmtBytes(cached.bytes)}`]);
  }

  $('env-out').innerHTML =
    '<table style="margin-top:12px">' +
    rows.map(([k, v]) => `<tr><th style="width:38%">${k}</th><td>${v}</td></tr>`).join('') +
    '</table>';

  $('btn-bench').disabled = false;
}

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
      powerPreference: ep === 'webgpu' ? environment?.providers?.powerPreference : undefined,
    });

    // --- 主流水線（用 manifest 指定的切分數）---
    say(`載入 ${manifest.num_shards} 個模型分段（第一次要下載，請耐心等）…`, 8);
    const pipeline = new Pipeline(ort, manifest, baseUrl, { ep, scheme: 'none' });
    const loadStats = await pipeline.load();
    const loadedBytes = loadStats.reduce((a, s) => a + s.bytes, 0);

    // --- Q1 + Q2：一次掃描，兩個答案 ---
    // 固定完整流水線、只掃序列長度。層數全程不變，所以每一段的擬合截距
    // 就是它的固定 dispatch 成本，不會和「少算幾層」混在一起。
    say('掃描序列長度（同時量邊際成本與每段的固定成本）…', 30);
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
        // 「報告寫的卡 == 真的在算的卡」的欄位（讀 env.webgpu.device，
        // 不是 env.webgpu.adapter —— 後者 ORT 1.30 從不寫回）。
        actualAdapter: ep === 'webgpu' ? await actualAdapterInfo(ort) : null,
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

function renderResults(r) {
  const k = r.kStar;
  const h = r.hopOverhead;
  const out = [];

  out.push('<h2>載入的權重</h2>');
  out.push(`<p><strong>${fmtBytes(r.model.loadedBytes)}</strong>
    <span class="dim">　${r.model.shards} 段、共 ${r.model.layers} 層${
      r.model.dtype ? `、權重 ${r.model.dtype}` : ''}</span></p>`);

  // ---- 一次多算幾個位置划不划算（取代原本的 K*）----
  out.push('<h2>一次多算幾個位置，划算嗎？</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    投機解碼的作法是先用小模型猜 γ 個字，再<strong>一次</strong>送進流水線驗證。
    划不划算只取決於一件事：多驗一個位置要多花多少時間。</p>`);
  out.push(`<p><strong>第一個位置 ${ms(k.firstPositionMs)} ms，
    之後每多一個位置只要 ${ms(k.marginalMsPerPosition, 2)} ms</strong></p>`);

  if (k.verifyWindow.length) {
    out.push('<table><tr><th>一次猜幾個字 (γ)</th><th class="num">總耗時 (ms)</th>' +
      '<th class="num">相對單一 token</th><th class="num">每位置 (ms)</th></tr>' +
      k.verifyWindow.map((v) => `<tr><td>${v.gamma}</td>` +
        `<td class="num">${v.totalMs.toFixed(1)}</td>` +
        `<td class="num">${v.vsSingleToken.toFixed(2)}×</td>` +
        `<td class="num">${v.msPerPosition.toFixed(2)}</td></tr>`).join('') + '</table>');
    const g8 = k.verifyWindow.find((v) => v.gamma === 8);
    if (g8) {
      out.push(`<p class="dim" style="font-size:14px">
        讀法：猜 8 個字一次驗證只花單一 token 的
        <strong>${g8.vsSingleToken.toFixed(2)} 倍</strong>時間，卻驗完了 8 個位置。
        只要小模型猜對的比例夠高，這筆交易就划算。</p>`);
    }
  }

  // 這幾段警告比上面的數字重要：不講的話，一條沒有轉折點的曲線
  // 和真正的 roofline 轉折點看起來一模一樣。
  if (k.knee === 'none-observed') {
    out.push(`<p class="warn" style="font-size:14px">
      ⚠ 在 K=1…${k.maxKTested} 之間，總耗時是一條<strong>直線</strong>
      （r² = ${ms(k.fit?.r2, 3)}），沒有任何轉折點。
      所以傳統的「平行視窗 K*」在這台機器上<strong>沒有意義</strong> ——
      不是掃描不夠遠（拉到 1024 也一樣），而是這個模型太小。</p>`);
  } else if (k.knee === 'observed') {
    out.push(`<p class="dim" style="font-size:14px">
      每位置耗時在 K=${k.kStar} 達到最低（${ms(k.msPerPositionAtKStar, 2)} ms），
      之後開始回升 —— 那是真正的轉折點。</p>`);
  }
  if (k.fit) {
    out.push(`<p class="dim" style="font-size:14px">
      擬合：總耗時 ≈ <code>${ms(k.fit.fixedMs)} ms + ${ms(k.fit.msPerPositionSlope, 2)} ms × 位置數</code>
      （r² = ${ms(k.fit.r2, 3)}）。固定的那一項在 K=1 時佔
      ${k.fixedFractionAtK1 == null ? '—' : (k.fixedFractionAtK1 * 100).toFixed(0)}%。
      ${k.amortisation.map((a) =>
        `要把每位置成本壓到 ${a.withinPct}% 以內需要 K ≥ ${a.k}。`).join('')}</p>`);
  }
  if (k.regime === 'dispatch-bound') {
    out.push(`<p class="warn" style="font-size:14px">
      ⚠ 固定成本佔了大半，代表這台機器上的瓶頸是<strong>呼叫開銷</strong>
      （每次推論的固定成本），而不是讀權重的頻寬。
      因為測試模型只有 1.35 億參數 —— 權重太小，讀權重根本不是瓶頸。
      所以這些數字<strong>不能直接外推到 700 億那種大模型</strong>。
      想真正回答那一題，在本機用大模型跑：
      <code>MODEL=HuggingFaceTB/SmolLM2-1.7B npm run dev</code>
      （只在你自己電腦上，不會影響線上的量測頁）。</p>`);
  }

  out.push('<details style="margin-top:8px"><summary class="dim" style="font-size:13px;cursor:pointer">' +
    '展開完整的掃描資料</summary>' +
    '<table><tr><th>K</th><th class="num">總耗時 (ms)</th><th class="num">每位置 (ms)</th></tr>' +
    k.points.map((p) => `<tr><td>${p.k}</td><td class="num">${p.totalMs.toFixed(1)}</td>` +
      `<td class="num">${p.msPerPosition.toFixed(2)}</td></tr>`).join('') + '</table></details>');

  // ---- 每段的固定成本 ----
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

  // ---- 後端數值差異 ----
  out.push('<h2>不同執行後端的數值差異</h2>');
  out.push(`<p class="dim" style="font-size:14px">
    同一個模型用 WASM 和 WebGPU 算，結果會有極小的差異。
    差異太大的話，不同裝置組成的網路可能算出不一致的結果。</p>`);
  out.push('<table><tr><th>後端</th><th class="num">首次 (ms)</th><th class="num">穩態 (ms)</th>' +
    '<th class="num">與參考值最大差</th><th class="num">選字一致率</th></tr>' +
    r.providerComparison.map((p) => p.ok
      ? `<tr><td>${p.ep}</td>` +
        `<td class="num">${ms(p.firstRunMs, 0)}</td>` +
        `<td class="num">${ms(p.ms, 0)}</td>` +
        `<td class="num">${p.maxAbsDiff.toExponential(2)}</td>` +
        `<td class="num">${p.argmaxAgreePct.toFixed(2)}%</td></tr>`
      : `<tr><td>${p.ep}</td><td colspan="4" class="bad">失敗：${p.error}</td></tr>`,
    ).join('') + '</table>');
  out.push(`<p class="dim" style="font-size:13px">
    「首次」包含 shader 編譯等一次性成本，「穩態」才是之後每次的速度 ——
    兩者混在一起會讓人誤以為 WebGPU 比較慢。首次那一欄本身也有用：
    那是一個節點加入網路後要付的暖機成本。
    ${r.ort.numThreads === 1
      ? '另外，這裡的 WASM 是<strong>單執行緒</strong>（沒有 cross-origin isolation），'
        + '不要把它當成 WASM 的速度上限。'
      : ''}</p>`);

  const cross = r.providerComparison.find((p) => p.maxDiffVsFirstEp != null);
  if (cross) {
    out.push(`<p class="dim" style="font-size:14px">
      兩個後端之間的最大差異：<code>${cross.maxDiffVsFirstEp.toExponential(2)}</code>
      —— 這才是「不同裝置混用不同後端會不會分歧」的直接答案。</p>`);
  }

  if (r.ort.actualAdapter) {
    out.push(`<p class="dim" style="font-size:13px">
      實際執行的顯示卡：<code>${adapterLabel(r.ort.actualAdapter)}</code>
      （由 ORT 建立的 device 回報，不是我們猜的）</p>`);
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
