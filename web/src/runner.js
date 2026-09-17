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

/** ORT 的全域設定。必須在建立任何 session 之前呼叫。 */
export function configureOrt(ort, { wasmPaths, numThreads } = {}) {
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
  return { crossOriginIsolated: isolated, numThreads: ort.env.wasm.numThreads };
}

/** 偵測這台裝置支援哪些 execution provider。 */
export async function detectProviders() {
  const out = { wasm: true, webgpu: false, adapter: null, reason: null };
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    out.reason = '這個瀏覽器沒有 WebGPU（navigator.gpu 不存在）';
    return out;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      out.reason = '有 WebGPU API 但取不到 adapter（通常是顯卡被列入黑名單或驅動太舊）';
      return out;
    }
    out.webgpu = true;
    const info = adapter.info ?? {};
    out.adapter = {
      vendor: info.vendor ?? null,
      architecture: info.architecture ?? null,
      description: info.description ?? null,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      shaderF16: adapter.features.has('shader-f16'),
      subgroups: adapter.features.has('subgroups'),
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
      try {
        const ext = await fetchShard(this.baseUrl + dataName, { optional: true });
        if (ext) options.externalData = [{ path: dataName, data: ext }];
      } catch {
        /* 沒有旁檔就是單檔模型，正常 */
      }

      const session = await this.ort.InferenceSession.create(model, options);
      this.sessions.push({ meta, session });
      this.loadStats.push({
        index: meta.index,
        ms: performance.now() - t0,
        bytes: model.byteLength,
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
