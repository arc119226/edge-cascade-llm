/**
 * 激活值量化 —— 必須與 `spike/quant_schemes.py` 逐位元對應。
 *
 * 這是節點之間的「線路格式」。如果 JS 端與 Python 端算出不同的結果，
 * 流水線就會在節點之間產生不一致，而且會是那種很難查的錯：
 * 輸出看起來合理、只是慢慢偏掉。所以這裡的實作刻意寫得囉嗦而直白，
 * 寧可慢一點也要和 Python 版一模一樣。
 *
 * M4 實測結論（見 docs/00-feasibility.md §6）：
 *   - per-tensor int8 不可用 —— PPL 暴增 30444%
 *   - scale 必須「逐 token」適應，因為巨值集中在特定 token
 *   - 預設格式：per-channel int8 + 前 3% 通道保留 fp16（等效 8.24 bits）
 */

/** 對稱量化的 quantize -> dequantize 往返。scale 需能廣播。 */
function qdq(value, scale, qmax) {
  if (scale === 0) return 0; // 整片都是 0
  const q = Math.max(-qmax - 1, Math.min(qmax, Math.round(value / scale)));
  return q * scale;
}

/** fp16 往返 —— 模擬用半精度傳輸造成的精度損失。 */
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
export function fp16Roundtrip(x) {
  _f32[0] = x;
  const bits = _u32[0];
  const sign = bits >>> 31;
  let exp = (bits >>> 23) & 0xff;
  let mant = bits & 0x7fffff;

  if (exp === 0xff) return x; // Inf / NaN 原樣通過
  // 轉成 fp16 的指數範圍後再轉回來
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign ? -Infinity : Infinity; // 溢位
  if (e <= 0) return sign ? -0 : 0; // 下溢（不處理非正規數，足夠近似）
  // fp16 只有 10 bits 尾數：截掉低 13 bits，並做最近偶數捨入
  const round = (mant + 0x1000) >>> 13;
  mant = (round & 0x3ff) << 13;
  if (round > 0x3ff) e += 1;
  _u32[0] = (sign << 31) | ((e - 15 + 127) << 23) | mant;
  return _f32[0];
}

/**
 * 每個 hidden 維度一個 scale，scale 跨所有 token 共用。
 *
 * ⚠ M4 實測這個方案單獨使用時表現不佳（PPL +6.23%），比 group-wise 還差。
 * 原因是 scale 跨 token 共用：一個 token 的巨值會把該 channel 的 scale
 * 對所有 token 都撐大。保留它是為了與 Python 版對齊、以及當作
 * per-channel+outlier 的基底。
 */
export function perChannel(x, dModel, bits = 8) {
  const qmax = (1 << (bits - 1)) - 1;
  const n = x.length;
  const nTok = n / dModel;
  const out = new Float32Array(n);

  const scale = new Float32Array(dModel);
  for (let c = 0; c < dModel; c++) {
    let m = 0;
    for (let t = 0; t < nTok; t++) m = Math.max(m, Math.abs(x[t * dModel + c]));
    scale[c] = m / qmax;
  }
  for (let t = 0; t < nTok; t++) {
    for (let c = 0; c < dModel; c++) {
      const i = t * dModel + c;
      out[i] = qdq(x[i], scale[c], qmax);
    }
  }
  return out;
}

/**
 * 把 channel 切成固定大小的組，每組、每個 token 各一個 scale。
 *
 * M4 實測：這是最好的「純 int8、零額外頻寬」選項（PPL +0.48%）。
 * d_model 不是 group 整數倍時，尾巴自成一組（與 Python 版行為一致 ——
 * 早期版本這裡會靜默退回 per-channel，是個真實踩過的 bug）。
 */
export function groupWise(x, dModel, bits = 8, group = 64) {
  const qmax = (1 << (bits - 1)) - 1;
  const nTok = x.length / dModel;
  const out = new Float32Array(x.length);

  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let g0 = 0; g0 < dModel; g0 += group) {
      const g1 = Math.min(g0 + group, dModel);
      let m = 0;
      for (let c = g0; c < g1; c++) m = Math.max(m, Math.abs(x[base + c]));
      const scale = m / qmax;
      for (let c = g0; c < g1; c++) out[base + c] = qdq(x[base + c], scale, qmax);
    }
  }
  return out;
}

/**
 * 預設線路格式：per-channel int8 + 數值最大的一小撮 channel 保留 fp16。
 *
 * M4 實測最佳：PPL 退化 0.03%、argmax 一致 99.32%，
 * 線路成本僅比純 int8 多 3%。
 */
export function perChannelOutlierFp16(x, dModel, bits = 8, outlierFrac = 0.03) {
  const nTok = x.length / dModel;

  // 先算每個 channel 的最大絕對值，用來挑離群
  const mag = new Float32Array(dModel);
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (let c = 0; c < dModel; c++) {
      const a = Math.abs(x[base + c]);
      if (a > mag[c]) mag[c] = a;
    }
  }
  const nOut = Math.max(1, Math.floor(dModel * outlierFrac));
  // 取前 nOut 大的 channel（用索引排序，dModel 不大所以直接排）
  const idx = Array.from({ length: dModel }, (_, i) => i)
    .sort((a, b) => mag[b] - mag[a])
    .slice(0, nOut);
  const isOutlier = new Uint8Array(dModel);
  for (const i of idx) isOutlier[i] = 1;

  const out = perChannel(x, dModel, bits);
  for (let t = 0; t < nTok; t++) {
    const base = t * dModel;
    for (const c of idx) out[base + c] = fp16Roundtrip(x[base + c]);
  }
  return out;
}

/** 名稱 -> { fn, wireBits }。wireBits 是每個值的等效傳輸位元數。 */
export const SCHEMES = {
  none: { fn: (x) => x, wireBits: 32, label: 'fp32（不量化）' },
  'per-channel': {
    fn: (x, d) => perChannel(x, d, 8),
    wireBits: 8,
    label: 'per-channel int8',
  },
  'group-64': {
    fn: (x, d) => groupWise(x, d, 8, 64),
    wireBits: 8,
    label: 'group-64 int8（純 int8 最佳）',
  },
  wire: {
    fn: (x, d) => perChannelOutlierFp16(x, d, 8, 0.03),
    wireBits: 8.24,
    label: 'per-channel + 3% 通道 fp16（預設）',
  },
};

export const DEFAULT_SCHEME = 'wire';

/** 這個張量若用該方案傳輸，實際要送多少位元組。 */
export function wireBytes(numel, schemeName) {
  const s = SCHEMES[schemeName] ?? SCHEMES[DEFAULT_SCHEME];
  return Math.ceil((numel * s.wireBits) / 8);
}
