/**
 * Shard 流水線執行器。
 *
 * 這是 EdgeCascadeLLM 節點端的核心：載入一段模型（若干層），
 * 吃進上一個節點送來的 hidden state，算完之後送給下一個節點。
 *
 * 目前這一版把所有 shard 跑在同一個瀏覽器分頁裡（等於「本機模擬整條流水線」）。
 * M3 才會把 shard 之間換成真的 WebRTC DataChannel。先確定計算部分對，
 * 再加網路，否則出錯時分不清是算錯還是傳錯。
 */

import { SCHEMES, DEFAULT_SCHEME, wireBytes } from './quant.js';
import { fetchShard } from './cache.js';

/**
 * ORT 的全域設定。必須在建立任何 session 之前呼叫。
 *
 * `powerPreference` 特別重要，而且是踩過坑學到的：ORT-Web 的 WebGPU 後端
 * **自己**會呼叫一次 `navigator.gpu.requestAdapter()`，用的是
 * `ort.env.webgpu.powerPreference`，完全不理會我們在 detectProviders()
 * 裡挑到的那個 adapter。所以混合顯卡的機器上，只修偵測而不設這個值，
 * 會變成「報告上寫獨立顯卡、實際跑內顯」—— 比沒偵測到還糟。
 *
 * ORT 1.30 把這個屬性標成 deprecated，建議改成自己建 GPUDevice 塞進
 * `env.webgpu.device`。那條路現在不走：自建 device 就得自己複製 ORT
 * 對 features（shader-f16、subgroups）與各項 limits 的整套要求，
 * 漏一項就是變慢或直接建不起來。等 powerPreference 真的被移除再改。
 */
export function configureOrt(ort, { wasmPaths, numThreads, powerPreference } = {}) {
  if (powerPreference && ort.env.webgpu) {
    // 只在第一個 WebGPU session 建立之前設定才有效果。
    ort.env.webgpu.powerPreference = powerPreference;
  }
  if (wasmPaths) {
    // ORT 是相對於「它自己的模組網址」去解析 wasmPaths，不是相對於頁面。
    // 直接傳 './ort/' 會變成 /ort/ort/... 而找不到檔案，所以這裡先解成絕對網址。
    ort.env.wasm.wasmPaths = /^https?:\/\//.test(wasmPaths)
      ? wasmPaths
      : new URL(wasmPaths, globalThis.location?.href ?? 'http://localhost/').href;
  }
  // 多執行緒需要 cross-origin isolation（COOP/COEP）。沒有的話 ORT 會自己退回單緒，
  // 但先明確偵測，才能在 UI 上告訴使用者為什麼比較慢。
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  ort.env.wasm.numThreads = numThreads ?? (isolated ? navigator.hardwareConcurrency || 4 : 1);
  return {
    crossOriginIsolated: isolated,
    numThreads: ort.env.wasm.numThreads,
    powerPreference: powerPreference ?? null,
  };
}

/**
 * ORT 實際拿到的是哪一張顯示卡。
 *
 * 只有在第一個 WebGPU session 建立之後才問得到。這個欄位存在的理由很單純：
 * 沒有它，我們只是在「相信」上面設的 powerPreference 生效了。
 * 報告要能自己證明它量的是哪張卡。
 */
export function actualAdapterInfo(ort) {
  const adapter = ort?.env?.webgpu?.adapter;
  if (!adapter?.info) return null;
  return describeAdapter(adapter);
}

/** 把一個 GPUAdapter 攤平成可以塞進 JSON 報告的樣子。 */
function describeAdapter(adapter) {
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor ?? null,
    architecture: info.architecture ?? null,
    device: info.device ?? null,
    description: info.description ?? null,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    shaderF16: adapter.features.has('shader-f16'),
    subgroups: adapter.features.has('subgroups'),
  };
}

/** 兩個 adapter 是不是同一張卡。Chrome 會把型號遮罩掉，所以只能靠這幾欄比。 */
function sameAdapter(a, b) {
  if (!a || !b) return false;
  return a.vendor === b.vendor
    && a.architecture === b.architecture
    && a.device === b.device
    && a.description === b.description
    && a.maxBufferSize === b.maxBufferSize;
}

async function tryAdapter(options) {
  try {
    return (await navigator.gpu.requestAdapter(options)) ?? null;
  } catch {
    return null;
  }
}

/**
 * 偵測這台裝置支援哪些 execution provider。
 *
 * 混合顯卡（筆電內顯 + 獨立顯卡）要特別處理：不帶 `powerPreference` 呼叫
 * `requestAdapter()` 時，瀏覽器回的是**內顯**。使用者回報「我的 NVIDIA
 * 沒被偵測到」就是這個原因 —— 那張卡一直都在，只是我們沒問對問題。
 *
 * 所以這裡把 high-performance 與 low-power 兩個都探一次：
 *   - `out.adapter`  實際會拿去跑的那一張
 *   - `out.adapters` `{ chosen, other, distinct }`，兩張都列出來
 *
 * 單顯卡機器上兩次會拿到同一張，`distinct: false`，UI 就不用多講什麼。
 *
 * @param {{powerPreference?: 'high-performance'|'low-power'}} opts
 */
export async function detectProviders({ powerPreference = 'high-performance' } = {}) {
  const out = {
    wasm: true,
    webgpu: false,
    adapter: null,
    adapters: null,
    powerPreference,
    reason: null,
  };
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    out.reason = '這個瀏覽器沒有 WebGPU（navigator.gpu 不存在）';
    return out;
  }
  try {
    // 要的那張優先；拿不到就退回瀏覽器預設，總比直接報「不支援」誠實。
    let chosen = await tryAdapter({ powerPreference });
    let fellBack = false;
    if (!chosen) {
      chosen = await tryAdapter(undefined);
      fellBack = chosen != null;
    }
    if (!chosen) {
      out.reason = '有 WebGPU API 但取不到 adapter（通常是顯卡被列入黑名單或驅動太舊）';
      return out;
    }

    out.webgpu = true;
    out.adapter = describeAdapter(chosen);
    if (fellBack) {
      out.reason = `取不到 ${powerPreference} adapter，已退回瀏覽器預設的那一張`;
      out.powerPreference = null;
    }

    // 另一張卡：只為了「告訴使用者他有幾張」，取不到就當作沒有。
    const otherPreference = powerPreference === 'high-performance' ? 'low-power' : 'high-performance';
    const otherAdapter = await tryAdapter({ powerPreference: otherPreference });
    const otherInfo = otherAdapter ? describeAdapter(otherAdapter) : null;
    out.adapters = {
      chosen: out.adapter,
      // 退回過預設 adapter 的話，就說不準拿到的是哪一種偏好了，照實記 null。
      chosenPreference: fellBack ? null : powerPreference,
      other: otherInfo,
      otherPreference,
      // 探不到另一張時不算「有兩張」—— 不然單顯卡機器會被誤報成雙顯卡。
      distinct: otherInfo != null && !sameAdapter(out.adapter, otherInfo),
    };
  } catch (e) {
    out.reason = `requestAdapter 失敗：${e}`;
  }
  return out;
}

export class Pipeline {
  /**
   * @param {object} ort           onnxruntime-web 模組
   * @param {object} manifest      export_shards.py 產出的 manifest.json
   * @param {string} baseUrl       shard 檔案所在的位置
   * @param {object} opts          { ep, scheme, onProgress }
   */
  constructor(ort, manifest, baseUrl, opts = {}) {
    this.ort = ort;
    this.manifest = manifest;
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
    this.ep = opts.ep ?? 'wasm';
    this.scheme = opts.scheme ?? DEFAULT_SCHEME;
    this.onProgress = opts.onProgress ?? (() => {});
    this.sessions = [];
    this.loadStats = [];
  }

  /** 載入全部 shard。回傳每個 shard 的載入統計。 */
  async load() {
    this.sessions = [];
    this.loadStats = [];

    for (const meta of this.manifest.shards) {
      this.onProgress({ phase: 'load', index: meta.index, total: this.manifest.shards.length });
      const t0 = performance.now();

      const model = await fetchShard(this.baseUrl + meta.file);
      const options = { executionProviders: [this.ep] };

      // torch.onnx 對較大的模型會把權重放進 .onnx.data 旁檔。
      // ORT-Web 不會自己去抓那個檔，必須明確餵進來 ——
      // 少了這段，session 建立就會失敗（"Module.MountedFiles is not available"）。
      const dataName = meta.file + '.data';
      let externalBytes = 0;
      try {
        const ext = await fetchShard(this.baseUrl + dataName, { optional: true });
        if (ext) {
          options.externalData = [{ path: dataName, data: ext }];
          externalBytes = ext.byteLength;
        }
      } catch {
        /* 沒有旁檔就是單檔模型，正常 */
      }

      const session = await this.ort.InferenceSession.create(model, options);
      this.sessions.push({ meta, session });
      this.loadStats.push({
        index: meta.index,
        ms: performance.now() - t0,
        // 權重幾乎全在旁檔裡：只算 .onnx 的話，一個幾百 MB 的模型會被報成
        // 幾 MB。實測資料裡 loadedBytes = 2.4 MB 就是漏了這一項。
        bytes: model.byteLength + externalBytes,
        modelBytes: model.byteLength,
        externalBytes,
      });
    }
    return this.loadStats;
  }

  /**
   * 跑一次完整的流水線 traversal。
   *
   * @param {BigInt64Array} inputIds  [1, seq]
   * @returns {{logits: Float32Array, dims: number[], hops: object[], totalMs: number}}
   */
  async run(inputIds, seqLen) {
    const { Tensor } = this.ort;
    const d = this.manifest.hidden_size;
    const posData = new BigInt64Array(seqLen);
    for (let i = 0; i < seqLen; i++) posData[i] = BigInt(i);

    let current = new Tensor('int64', inputIds, [1, seqLen]);
    const hops = [];
    const tStart = performance.now();

    for (let i = 0; i < this.sessions.length; i++) {
      const { meta, session } = this.sessions[i];
      const isLast = i === this.sessions.length - 1;

      const feeds = { position_ids: new Tensor('int64', posData, [1, seqLen]) };
      feeds[meta.input] = current;

      const t0 = performance.now();
      const out = await session.run(feeds);
      const computeMs = performance.now() - t0;

      let next = out[meta.output];

      // 中間的 shard 輸出就是要走網路的東西 —— 在這裡量化，
      // 模擬它真的被壓成線路格式送出去再解回來。
      let quantMs = 0;
      let bytes = 0;
      if (!isLast) {
        const q0 = performance.now();
        const scheme = SCHEMES[this.scheme] ?? SCHEMES[DEFAULT_SCHEME];
        const quantised = scheme.fn(next.data, d);
        quantMs = performance.now() - q0;
        bytes = wireBytes(next.data.length, this.scheme);
        next = new Tensor('float32', quantised, next.dims);
      }

      hops.push({
        index: meta.index,
        layers: meta.layers,
        computeMs,
        quantMs,
        wireBytes: bytes,
        dims: next.dims.slice(),
      });
      current = next;
      this.onProgress({ phase: 'run', index: i, total: this.sessions.length });
    }

    return {
      logits: current.data,
      dims: current.dims.slice(),
      hops,
      totalMs: performance.now() - tStart,
    };
  }

  async dispose() {
    for (const { session } of this.sessions) {
      await session.release?.();
    }
    this.sessions = [];
  }
}

/** 逐位置取 argmax —— 貪婪解碼真正在意的東西。 */
export function argmaxPerPosition(logits, dims) {
  const [, seq, vocab] = dims;
  const out = new Int32Array(seq);
  for (let t = 0; t < seq; t++) {
    let best = 0;
    const base = t * vocab;
    for (let v = 1; v < vocab; v++) if (logits[base + v] > logits[base + best]) best = v;
    out[t] = best;
  }
  return out;
}

/**
 * 載入參考 logits。
 *
 * 有兩組對照值，用途完全不同，不可混用：
 *   - reference.bin  未量化的 PyTorch fp32 真值 -> 用來衡量「量化品質」
 *   - native.bin     原生 ORT 跑同一份 ONNX 的輸出 -> 用來驗證「瀏覽器流水線正確性」
 *
 * 驗證瀏覽器時必須拿 native.bin 比。拿 fp32 真值比的話，量化誤差會被
 * 誤判成流水線的 bug —— 那是兩個不同的問題，混在一起就沒有一個數字說得清楚。
 *
 * 都存成 fp16 二進位而非 JSON：786,432 個浮點數存 JSON 是 14.7 MB，
 * 存 fp16 只要 1.5 MB，而且 JSON 版本本身就超過 Cloudflare 的單檔上限。
 */
export async function loadReference(baseUrl, { preferNative = true } = {}) {
  const base = baseUrl.endsWith('/') ? baseUrl : baseUrl + '/';
  const meta = await (await fetch(base + 'reference.json')).json();

  const file = preferNative && meta.native_file ? meta.native_file : meta.logits_file;
  if (!file) {
    // 舊格式把 logits 直接放在 JSON 裡
    return { ...meta, logits: Float32Array.from(meta.logits), source: 'legacy-json' };
  }
  const buf = await (await fetch(base + file)).arrayBuffer();
  const half = new Uint16Array(buf);
  const logits = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) logits[i] = fp16ToFloat(half[i]);

  return { ...meta, logits, source: file };
}

/** fp16 位元樣式轉 JS number。 */
function fp16ToFloat(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/** 與參考 logits 比對，回傳可讀的差異報告。 */
export function compareToReference(logits, dims, reference) {
  // 直接 fetch('reference.json') 會拿到「只有中繼資料、沒有 logits」的物件，
  // 接著在下面炸成 `undefined is not iterable` —— 看不出根因。
  // 這個錯真的發生過（bench-ui 漏改），所以在這裡講清楚該怎麼做。
  if (!reference?.logits?.length) {
    throw new Error(
      '參考 logits 沒有載入。reference.json 只放中繼資料，實際數值在 ' +
      'logits_file / native_file 指到的二進位檔裡 —— 請用 loadReference() 載入，' +
      '不要直接 fetch reference.json。',
    );
  }
  const expected = reference.logits instanceof Float32Array
    ? reference.logits
    : Float32Array.from(reference.logits);
  let maxAbsDiff = 0;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(logits[i] - expected[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
  }
  const got = argmaxPerPosition(logits, dims);
  const exp = argmaxPerPosition(expected, reference.logits_shape);
  let agree = 0;
  for (let i = 0; i < got.length; i++) if (got[i] === exp[i]) agree++;
  return {
    maxAbsDiff,
    argmaxAgree: agree,
    argmaxTotal: got.length,
    argmaxAgreePct: (agree / got.length) * 100,
  };
}
